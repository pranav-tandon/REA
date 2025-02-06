# backend/flow.py
# ------------------------------------------------------------------------------
# Flow Documentation and Data Flow Summary
# ------------------------------------------------------------------------------
#
# 1. Collect Constraints:
#    - User hits the "CollectConstraints" UI which calls POST /flow/collect-constraints.
#    - This endpoint returns a profileId.
#    - The user's constraints are stored in the profiles collection with status "COLLECTED".
#
# 2. Confirmation:
#    - The next page calls GET /flow/confirm?profileId=xxx.
#    - The server fetches city information and recommended neighborhoods (sourced from the DB or an LLM).
#    - This information is displayed to the user.
#
# 3. Neighborhood Selection:
#    - The UI allows the user to pick from the recommended neighborhoods.
#    - A POST request to /flow/select-neighborhoods updates the profile document with the chosen neighborhoods.
#    - The profile status is updated to "NEIGHBORHOODS_SELECTED".
#
# 4. Scrape & RAG:
#    - The final page triggers a POST /flow/finalize-search?profileId=xxx.
#    - This endpoint collects (or scrapes) about 100 listings from the listings_collection.
#    - It creates embeddings using SentenceTransformer and builds an in-memory FAISS index.
#    - A similarity search is performed on the FAISS index to extract the top 5 listings along with short justifications.
#
# 5. Chat:
#    - The final page includes a chat text box that connects to the existing POST /chat endpoint.
#    - Users can refine or re-run constraints, optionally linking back to the "CollectConstraints" step.
#
# ------------------------------------------------------------------------------

import logging
import sys
import os
import pymongo
from bson import ObjectId, errors as bson_errors
from dotenv import load_dotenv
import json
import datetime

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -----------------
# KOR for Fallback
# -----------------
from kor import from_pydantic, create_extraction_chain

# Importing chat_llm and models from house_search
from house_search import HouseRequest, chat_llm, ActionType, extraction_chain

# Import SystemMessage and HumanMessage needed for the LLM call
from langchain.schema import HumanMessage, SystemMessage

from fastapi import APIRouter, HTTPException
from rag import get_top_10_with_profile_rag
from bson import ObjectId

load_dotenv()

# ------------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------------
logger = logging.getLogger("flow_logger")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s: %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

# ------------------------------------------------------------------------------
# MongoDB Setup
# ------------------------------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = pymongo.MongoClient(MONGO_URI)
db = client["real_estate_db"]

profiles_collection = db["profiles"]         # new user "profile" constraints
listings_collection = db["listings"]         # your existing listings
neighborhood_stats = db["neighborhood_stats"]  # optional

# ------------------------------------------------------------------------------
# Router
# ------------------------------------------------------------------------------
router = APIRouter()

# ------------------------------------------------------------------------------
# Pydantic Schemas
# ------------------------------------------------------------------------------
class CollectConstraintsRequest(BaseModel):
    user_input: Optional[str] = Field(
        None, 
        description="The user's free-form real-estate query input for LLM extraction"
    )
    city: str = Field(..., description="City name")
    state: str = Field(..., description="State code")
    price: float = Field(..., description="Target price")
    actionType: str = Field(..., description="Buy or Sell")
    notes: Optional[str] = None
    zipCode: Optional[str] = None
    schoolDistrict: Optional[str] = None
    maxCommuteTime: Optional[int] = None
    yearBuilt: Optional[int] = None
    parkingSpaces: Optional[int] = None
    stories: Optional[int] = None
    basement: Optional[bool] = None
    lotSize: Optional[str] = None
    kitchenPreferences: Optional[str] = None
    flooringType: Optional[str] = None
    layoutStyle: Optional[str] = None
    exteriorMaterial: Optional[str] = None
    hasPool: Optional[bool] = None
    outdoorSpace: Optional[str] = None

class CollectConstraintsResponse(BaseModel):
    profileId: str
    success: bool

class NeighborhoodSelectionRequest(BaseModel):
    profileId: str
    neighborhoods: List[str]

class NeighborhoodSelectionResponse(BaseModel):
    profileId: str
    selected_neighborhoods: List[str]

class AugmentListingsRequest(BaseModel):
    profileId: str
    listingIds: List[str] = Field(..., description="Which listings to augment via LLM")
    buyerQuery: Optional[str] = Field(None, description="Buyer preferences or question")

class FlowRequest(BaseModel):
    user_input: str
    context: Optional[str] = None

class FinalizeSearchRequest(BaseModel):
    profileId: str = Field(..., description="Profile ID to finalize search for")

class FinalizeSearchResponse(BaseModel):
    profileId: str
    top_recommendations: List[dict]
    results_count: int

# ------------------------------------------------------------------------------
# KOR Fallback Schema: NeighborhoodExtraction
# ------------------------------------------------------------------------------
class NeighborhoodExtraction(BaseModel):
    """
    Simple schema for extracting a list of neighborhood strings.
    """
    neighborhoods: List[str]

# Build Kor node + chain for fallback
# 1) from_pydantic(...) now returns (node, validator)
fallback_node, fallback_validator = from_pydantic(
    NeighborhoodExtraction,
    description="Extract neighborhoods from text. Neighborhoods should be a list of strings.",
    examples=[
        (
            "Here are the top areas: Ballard, Fremont, Capitol Hill, Queen Anne, Magnolia",
            {"neighborhoods": ["Ballard", "Fremont", "Capitol Hill", "Queen Anne", "Magnolia"]}
        )
    ],
    many=False
)

# 2) create_extraction_chain with node=fallback_node AND encoder_or_encoder_class="json"
fallback_extraction_chain = create_extraction_chain(
    llm=chat_llm,
    node=fallback_node,
    encoder_or_encoder_class="json"  # Use JSON encoder to handle embedded lists
)

def extract_neighborhoods_with_kor(raw_text: str) -> List[str]:
    """
    Use the fallback_extraction_chain to parse neighborhoods from raw_text
    using chain.run(...) instead of chain.invoke(...).
    """
    chain_output = fallback_extraction_chain.run(raw_text)
    neighborhoods = chain_output.get("data", {}).get("neighborhoods", [])
    return neighborhoods

# ------------------------------------------------------------------------------
# Route 1: Collect Constraints (Using HouseSearch-style LLM extraction via Kor)
# ------------------------------------------------------------------------------
@router.post("/flow/collect-constraints", response_model=CollectConstraintsResponse)
async def collect_constraints(request: CollectConstraintsRequest):
    """
    1) Use the Kor extraction chain (like in /house-search) to parse constraints
       (city, state, max_price, beds, baths, action) from user_input.
    2) Store them (and any additional fields) in a profile doc with status='COLLECTED'.
    3) Return profileId so the front-end can proceed (we skip steps 2 & 3).
    """
    logger.info("STEP 1: Collect Constraints (via House LLM extraction)")

    try:
        # 1) Invoke the same LLM extraction chain you used in house-search.
        extraction_result = extraction_chain.invoke(request.user_input)
        house_req: HouseRequest = extraction_result.get("validated_data")

        if not house_req or not house_req.city:
            logger.warning("City name was not extracted from user_input.")
            raise HTTPException(
                status_code=400,
                detail="City name is required or was not extracted."
            )
        
        # 2) Build your profile object based on extracted fields + any extras
        #    from CollectConstraintsRequest.
        profile = {
            "city": house_req.city,
            "state": house_req.state,
            "price": house_req.max_price,              # from house_req
            "actionType": house_req.action.lower(),    # e.g. "buy" or "rent"
            "beds": house_req.beds,
            "baths": house_req.baths,
            # Optionally store notes from LLM if you have a 'notes' field,
            # otherwise fall back to request.notes.
            "notes": getattr(house_req, "notes", None) or request.notes,

            # Optionally store the entire original user_input.
            "original_user_input": request.user_input,

            # Then store all additional request fields as needed.
            "zipCode": request.zipCode,
            # etc... (store other additional fields if required)
            "status": "COLLECTED",
            "createdAt": datetime.datetime.utcnow(),
            "updatedAt": datetime.datetime.utcnow()
        }
        
        result = profiles_collection.insert_one(profile)
        profile_id = str(result.inserted_id)
        
        logger.info(f"Created profile with ID: {profile_id}")
        
        return CollectConstraintsResponse(
            profileId=profile_id,
            success=True
        )

    except Exception as e:
        logger.error(f"Error in collect_constraints: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# ------------------------------------------------------------------------------
# Route 2: Confirm (City Stats, Recommended Neighborhoods)
# ------------------------------------------------------------------------------
@router.get("/flow/confirm")
async def confirm_profile(profileId: str):
    """
    2) Return city info & recommended neighborhoods for user confirmation.
       They can see data about the city, then proceed to pick neighborhoods.
    """
    logger.info("STEP 2: Confirm - profileId=%s", profileId)

    try:
        profile = profiles_collection.find_one({"_id": ObjectId(profileId)})
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")

        city = profile.get("city")
        state = profile.get("state")

        # Default data if LLM fails or we can't parse
        default_data = {
            "city": city,
            "state": state,
            "city_stats": {
                "population": "Unknown",
                "median_home_price": "Contact local realtor",
                "climate": "Varies by season",
                "average_income": "Data not available"
            },
            "recommended_neighborhoods": [
                f"Downtown {city}",
                f"North {city}",
                f"South {city}",
                f"East {city}",
                f"West {city}"
            ]
        }

        # Make the LLM call to get city info in JSON
        messages = [
            SystemMessage(content="""You are a real estate expert AI. Return ONLY a JSON object with city 
            statistics and recommended neighborhoods. Format must be valid JSON."""),
            HumanMessage(content=f"""Return information about {city}, {state} in this exact JSON format:
            {{
                "city_stats": {{
                    "population": "number with commas",
                    "median_home_price": "dollar amount",
                    "climate": "climate type",
                    "average_income": "dollar amount"
                }},
                "recommended_neighborhoods": [
                    "neighborhood1" -Describe the neighborhood in one sentence,
                    "neighborhood2" -Describe the neighborhood in one sentence,
                    "neighborhood3" -Describe the neighborhood in one sentence,
                    "neighborhood4" -Describe the neighborhood in one sentence,
                    "neighborhood5" -Describe the neighborhood in one sentence
                ]
            }}""")
        ]

        logger.info(f"Requesting data for city: {city}, {state}")

        response = await chat_llm.ainvoke(messages)
        content = response.content
        logger.info(f"Raw LLM response: {content}")

        if content:
            try:
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                city_data = json.loads(content)
                return {
                    "city": city,
                    "state": state,
                    "city_stats": city_data["city_stats"],
                    "recommended_neighborhoods": city_data["recommended_neighborhoods"],
                    "source": "llm"
                }
            except Exception as parse_err:
                logger.warning(f"Failed to parse LLM response: {parse_err}")
                # -------------------------------
                # KOR Fallback: Extract neighborhoods from raw text
                # -------------------------------
                extracted = extract_neighborhoods_with_kor(content)
                return {
                    **default_data,
                    "source": "default+kor_fallback",
                    "kor_extracted_neighborhoods": extracted
                }
        else:
            logger.warning("Empty LLM response, using default data")
            return {**default_data, "source": "default", "error": "Empty LLM response"}

    except Exception as e:
        logger.error(f"Error in confirm_profile: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# ------------------------------------------------------------------------------
# Route 3: Select Neighborhoods
# ------------------------------------------------------------------------------
@router.post("/flow/select-neighborhoods", response_model=NeighborhoodSelectionResponse)
async def select_neighborhoods(profileId: str, neighborhoods: List[str]):
    """
    3) User picks up to 5 neighborhoods from recommended.
       Store them in the profile doc, set status='NEIGHBORHOODS_SELECTED'.
    """
    logger.info("STEP 3: Select Neighborhoods - profileId=%s", profileId)
    
    try:
        # Validate ObjectId format
        try:
            profile_obj_id = ObjectId(profileId)
        except bson_errors.InvalidId:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid profile ID format"
            )
        
        # Update profile
        result = profiles_collection.update_one(
            {"_id": profile_obj_id},
            {
                "$set": {
                    "neighborhoods": neighborhoods,
                    "status": "NEIGHBORHOODS_SELECTED",
                    "updatedAt": datetime.datetime.utcnow()
                }
            }
        )
        
        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Profile not found"
            )
            
        logger.info(f"Successfully updated neighborhoods for profile {profileId}")
        return NeighborhoodSelectionResponse(profileId=profileId, selected_neighborhoods=neighborhoods)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in select_neighborhoods: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# ------------------------------------------------------------------------------
# Route 4: Finalize Search (Scrape + RAG)
# ------------------------------------------------------------------------------
@router.post("/flow/finalize-search", response_model=FinalizeSearchResponse)
async def finalize_search(request: FinalizeSearchRequest):
    """
    Finalize search for a profile by gathering and ranking listings.
    """
    logger.info("STEP 4: Finalize Search - profileId=%s", request.profileId)
    
    try:
        # Validate ObjectId format
        try:
            profile_obj_id = ObjectId(request.profileId)
        except bson_errors.InvalidId:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid profile ID format"
            )
        
        # Find the profile
        profile = profiles_collection.find_one({"_id": profile_obj_id})
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Profile not found"
            )

        logger.info("Profile info => city=%s, neighborhoods=%s, price=%s, actionType=%s",
                   profile.get("city"), profile.get("neighborhoods"), 
                   profile.get("price"), profile.get("actionType"))

        # Gather listings from MongoDB
        listings = gather_top_100_listings(profile)
        logger.info("Gathered %d listings to re-rank", len(listings))

        # Run RAG approach
        final_recommendations = run_rag_and_rerank(listings, profile)
        logger.info("RAG complete. Found top %d listings", len(final_recommendations))

        # Build response with justifications
        results = []
        for listing in final_recommendations:
            justification = (
                f"Matches your budget ({profile['price']}) and is in "
                f"{listing.get('neighborhood', 'Unknown')}."
            )
            results.append({
                "listingId": str(listing.get("_id", "")),
                "address": listing.get("address", "No address"),
                "price": listing.get("price", 0),
                "justification": justification
            })

        # Update profile status
        profiles_collection.update_one(
            {"_id": profile_obj_id},
            {
                "$set": {
                    "status": "RECOMMENDATIONS_READY",
                    "updatedAt": datetime.datetime.utcnow()
                }
            }
        )

        return {
            "profileId": request.profileId,
            "top_recommendations": results,
            "results_count": len(results)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in finalize_search: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/rag/top10")
def rag_top10(profile_id: str):
    """
    Returns the top 10 listings for the given profile, 
    factoring in the user's conversation history.
    """
    try:
        # Validate object id
        if not ObjectId.is_valid(profile_id):
            raise HTTPException(status_code=400, detail="Invalid profile_id")

        top_10 = get_top_10_with_profile_rag(profile_id)
        return {"profile_id": profile_id, "top_10": top_10}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------------------
# OPTIONAL: Route 5 (LLM Augmentation)
# ------------------------------------------------------------------------------
@router.post("/flow/augment-listings")
def augment_listings(payload: AugmentListingsRequest):
    """
    5) (Optional) If you want to further personalize listings for a buyer,
       referencing user preferences. This draws from your snippet's 'Step 6' idea,
       but it's entirely optional.
    """
    logger.info("STEP 5: Augment Listing Descriptions for profileId=%s", payload.profileId)

    profile = profiles_collection.find_one({"_id": pymongo.ObjectId(payload.profileId)})
    if not profile:
        logger.error("Profile not found: %s", payload.profileId)
        raise HTTPException(status_code=404, detail="Profile not found")

    # Retrieve the listing docs
    listing_docs = list(listings_collection.find(
        {"_id": {"$in": [pymongo.ObjectId(lid) for lid in payload.listingIds]}}
    ))
    if not listing_docs:
        logger.warning("No listings found to augment for profileId=%s", payload.profileId)

    # Example augmentation code (pseudocode):
    # for doc in listing_docs:
    #     combined_text = doc["description"] + " " + payload.buyerQuery
    #     doc["augmented_description"] = <call your LLM to summarize combined_text>
    #     ...
    # return them

    augmented_results = []
    for doc in listing_docs:
        augmented_text = f"{doc.get('description','')} (Augmented for user query: {payload.buyerQuery})"
        augmented_results.append({
            "listingId": str(doc["_id"]),
            "original_description": doc.get("description",""),
            "augmented_description": augmented_text
        })

    return {"augmented_listings": augmented_results}

# ------------------------------------------------------------------------------
# Helper: gather_top_100_listings
# ------------------------------------------------------------------------------
def gather_top_100_listings(profile: dict) -> List[dict]:
    """
    Gather top 100 listings based on the profile. Builds a HouseRequest from the
    profile fields to leverage the same logic used by house_search.py.
    Then attempts a cached listing query; if none are found, tries a fallback query.
    """
    try:
        house_req = HouseRequest.parse_obj({
            "action": profile.get("actionType"),
            "city": profile.get("city"),
            "state": profile.get("state"),
            "beds": profile.get("beds"),
            "baths": profile.get("baths"),
            "max_price": profile.get("price")
        })
        logger.info("Extracted HouseRequest from profile: %s", house_req)

        # First, try to find cached listings (using the same fields as house_search)
        query = {
            "query_city": house_req.city,
            "search_type": house_req.action
        }
        if house_req.beds is not None:
            query["query_beds"] = house_req.beds
        if house_req.baths is not None:
            query["query_baths"] = house_req.baths
        if house_req.max_price is not None:
            query["query_max_price"] = house_req.max_price

        results = list(listings_collection.find(query).limit(100))
        if results:
            logger.info("Found %d cached listings using query: %s", len(results), query)
            return results

        # If no cached results, try fallback (search by city in "hdpData.homeInfo.city")
        fallback_query = {
            "hdpData.homeInfo.city": {"$regex": f"^{house_req.city}$", "$options": "i"}
        }
        if house_req.max_price is not None:
            # Allow for ~20% flexibility around the target price
            price = house_req.max_price
            fallback_query["hdpData.homeInfo.price"] = {"$gte": price * 0.8, "$lte": price * 1.2}

        results = list(listings_collection.find(fallback_query).limit(100))
        logger.info("Found %d listings using fallback query: %s", len(results), fallback_query)
        return results

    except Exception as e:
        logger.error("Error in gather_top_100_listings: %s", str(e), exc_info=True)
        return []

# ------------------------------------------------------------------------------
# Helper: run_rag_and_rerank
# ------------------------------------------------------------------------------
def run_rag_and_rerank(listings: List[dict], profile: dict) -> List[dict]:
    """Run RAG process to rank listings."""
    if not listings:
        return []

    # Convert each listing to text
    listing_texts = []
    for doc in listings:
        text = f"{doc.get('address', '')}, {doc.get('neighborhood', '')}, {doc.get('city', '')}. {doc.get('description', '')}"
        listing_texts.append(text)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(listing_texts, convert_to_numpy=True).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Build the query from the profile
    query = f"{profile.get('actionType')} in {profile.get('city')} for {profile.get('price')}. {profile.get('notes', '')}"
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")

    k = min(5, len(listings))
    D, I = index.search(query_embedding, k)

    # Return top k results
    return [listings[i] for i in I[0]]

@router.post("/flow/search")
async def flow_search(request: FlowRequest):
    try:
        # Your flow search logic here
        return {
            "message": "Flow search endpoint",
            "user_input": request.user_input
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/flow/status")
async def flow_status():
    """Check if the flow service is running."""
    return {"status": "active"}
