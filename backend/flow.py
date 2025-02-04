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
# Visual Summary:
#
# Collect Constraints:
#    • City (required)
#    • State (required)
#    • Price (required)
#    • Action (Buy / Sell)
#    • Notes (optional)
#    [Submit → store in profiles → next page]
#
# Confirmation:
#    • Display city & state.
#    • Show city statistics (population, weather, etc.).
#    • Present recommended neighborhoods (sourced from LLM or DB).
#    [Next → move to Neighborhood selection]
#
# Neighborhood Selection:
#    • Allow multi-select up to 5 neighborhoods.
#    [Finalize → update selection → next page]
#
# Final Recommendations:
#    • Execute the RAG process on approximately 100 listings to retrieve the top 5 matches.
#    • Provide short justifications with each listing.
#    • Include a chat text box for additional Q&A.
#    [Optionally include a "Start Over" or "Adjust Constraints" button]
#
# ------------------------------------------------------------------------------


from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional
import datetime
import logging
import sys
import os
import pymongo
from bson import ObjectId, errors as bson_errors
from dotenv import load_dotenv

# For the RAG step
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# This references your existing house_search logic
# e.g., "house_search" might have functions that scrape or fetch listings
from house_search import HouseRequest

load_dotenv()

# ------------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------------
logger = logging.getLogger("flow_logger")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '[%(levelname)s] %(asctime)s - %(name)s: %(message)s'
)
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
listings_collection = db["listings"]         # your existing listings, scraped from Zillow/Redfin
neighborhood_stats = db["neighborhood_stats"]  # optional

# ------------------------------------------------------------------------------
# Router
# ------------------------------------------------------------------------------
router = APIRouter()

# ------------------------------------------------------------------------------
# Pydantic Schemas
# ------------------------------------------------------------------------------
class CollectConstraintsRequest(BaseModel):
    city: str = Field(..., description="City name")
    state: str = Field(..., description="State code")
    price: float = Field(..., description="Target price")
    actionType: str = Field(..., description="Buy or Sell")
    notes: Optional[str] = None

class CollectConstraintsResponse(BaseModel):
    profileId: str
    success: bool

class NeighborhoodSelectionRequest(BaseModel):
    profileId: str
    neighborhoods: List[str]

class NeighborhoodSelectionResponse(BaseModel):
    profileId: str
    selected_neighborhoods: List[str]

# If you want an LLM-based augmentation step:
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
# Route 1: Collect Constraints
# ------------------------------------------------------------------------------
@router.post("/flow/collect-constraints", response_model=CollectConstraintsResponse)
async def collect_constraints(request: CollectConstraintsRequest):
    """
    1) Create a new profile with the user's basic constraints:
       - City, State, Price, ActionType (Buy/Sell), optional notes
       - Set status='COLLECTED'
       - Return profileId so front-end can move to next step
    """
    logger.info("STEP 1: Collect Constraints: city=%s, state=%s, price=%s, action=%s",
                request.city, request.state, request.price, request.actionType)

    try:
        profile = {
            "city": request.city,
            "state": request.state,
            "price": request.price,
            "actionType": request.actionType,
            "notes": request.notes,
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
       They can see data about the city (population, stats, etc.), then proceed
       to pick neighborhoods.
    """
    logger.info("STEP 2: Confirm - profileId=%s", profileId)

    try:
        # Validate ObjectId format
        try:
            profile_obj_id = ObjectId(profileId)
        except bson_errors.InvalidId:
            logger.error(f"Invalid profile ID format: {profileId}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid profile ID format - must be 24-character hex string"
            )
        
        # Find the profile
        profile = profiles_collection.find_one({"_id": profile_obj_id})
        if not profile:
            logger.error(f"Profile not found: {profileId}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Profile not found"
            )

        city = profile.get("city")
        state = profile.get("state")
        logger.info("Found profile: city=%s, state=%s, price=%s, actionType=%s",
                   city, state, profile.get("price"), profile.get("actionType"))

        # Get city info and recommended neighborhoods
        city_info_doc = neighborhood_stats.find_one({"city": city})
        
        if city_info_doc:
            logger.info(f"Found city info for {city}")
            city_stats = city_info_doc.get("stats", {})
            recommended_neighborhoods = city_info_doc.get("neighborhoods", [])
        else:
            logger.info(f"No city info found for {city}, using defaults")
            city_stats = {
                "population": "Data not available",
                "median_home_price": "Data not available",
                "climate": "Data not available",
                "average_income": "Data not available"
            }
            recommended_neighborhoods = [
                "Downtown",
                "Suburban Area",
                "Historic District",
                "Waterfront",
                "University District"
            ]
        
        return {
            "profileId": profileId,
            "city": city,
            "state": state,
            "city_stats": city_stats,
            "recommended_neighborhoods": recommended_neighborhoods
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in confirm_profile: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

# ------------------------------------------------------------------------------
# Route 3: Select Neighborhoods
# ------------------------------------------------------------------------------
@router.post("/flow/select-neighborhoods", response_model=NeighborhoodSelectionResponse)
async def select_neighborhoods(profileId: str, neighborhoods: List[str]):
    """
    3) User picks up to 5 neighborhoods from recommended.
       We store them in the profile doc, set status='NEIGHBORHOODS_SELECTED'.
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

# ------------------------------------------------------------------------------
# OPTIONAL: Route 5 (LLM Augmentation) - Taking Inspiration from Step 6
# ------------------------------------------------------------------------------
@router.post("/flow/augment-listings")
def augment_listings(payload: AugmentListingsRequest):
    """
    5) (Optional) If you want to further personalize listings for a buyer,
       referencing user preferences. This draws from your snippet's 'Step 6' idea,
       but it's entirely optional. You might use a ChatOpenAI or GPT to highlight
       features relevant to 'buyerQuery'.
    """
    logger.info("STEP 5: Augment Listing Descriptions for profileId=%s", payload.profileId)

    profile = profiles_collection.find_one({"_id": pymongo.ObjectId(payload.profileId)})
    if not profile:
        logger.error("Profile not found: %s", payload.profileId)
        raise HTTPException(status_code=404, detail="Profile not found")

    # Retrieve the listing docs
    listing_docs = list(listings_collection.find({"_id": {"$in": [pymongo.ObjectId(lid) for lid in payload.listingIds]}}))
    if not listing_docs:
        logger.warning("No listings found to augment for profileId=%s", payload.profileId)

    # "buyerQuery" might be something like "Looking for a cozy 3-bedroom with a big yard"
    # You can pass that + the listing's existing description into an LLM prompt
    # For example (pseudo-code):
    # 
    # for doc in listing_docs:
    #     combined_text = ...
    #     # call your LLM (like ChatOpenAI) to produce an augmented summary
    #     doc["augmented_description"] = ...
    #     # store it back or return it

    # Return them
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
    """Gather top 100 listings based on profile criteria."""
    try:
        city = profile.get("city")
        logger.info(f"Searching for listings in city: {city}")
        
        # Build query for nested city field (case-insensitive)
        query = {
            "hdpData.homeInfo.city": {"$regex": f"^{city}$", "$options": "i"}
        }
        
        # Add price filter if specified (also nested in hdpData.homeInfo)
        if price := profile.get("price"):
            # Allow for 20% price flexibility
            query["hdpData.homeInfo.price"] = {
                "$gte": price * 0.8,
                "$lte": price * 1.2
            }
        
        # Debug counts
        total_docs = listings_collection.count_documents({})
        city_matches = listings_collection.count_documents({
            "hdpData.homeInfo.city": {"$regex": f"^{city}$", "$options": "i"}
        })
        logger.info(f"Total documents: {total_docs}")
        logger.info(f"Documents matching city '{city}': {city_matches}")
        
        # Try to find listings with the full query
        results = list(listings_collection.find(query).limit(100))
        logger.info(f"Found {len(results)} results with full query: {query}")
        
        # If no results, try with just city
        if not results:
            logger.info("No results with price filter, trying city-only query...")
            results = list(listings_collection.find({
                "hdpData.homeInfo.city": {"$regex": f"^{city}$", "$options": "i"}
            }).limit(100))
            logger.info(f"Found {len(results)} results with city-only query")
        
        # If still no results, try partial city match
        if not results:
            logger.info("No results with exact city match, trying partial match...")
            results = list(listings_collection.find({
                "hdpData.homeInfo.city": {"$regex": city, "$options": "i"}
            }).limit(100))
            logger.info(f"Found {len(results)} results with partial city match")
        
        return results

    except Exception as e:
        logger.error(f"Error in gather_top_100_listings: {str(e)}", exc_info=True)
        return []

# ------------------------------------------------------------------------------
# Helper: run_rag_and_rerank
# ------------------------------------------------------------------------------
def run_rag_and_rerank(listings: List[dict], profile: dict) -> List[dict]:
    """Run RAG process to rank listings."""
    if not listings:
        return []

    # Convert listings to text
    listing_texts = []
    for doc in listings:
        text = f"{doc.get('address', '')}, {doc.get('neighborhood', '')}, {doc.get('city', '')}. {doc.get('description', '')}"
        listing_texts.append(text)

    # Embed texts
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(listing_texts, convert_to_numpy=True)
    embeddings = np.array(embeddings, dtype="float32")

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Create and embed query
    query = f"{profile.get('actionType')} in {profile.get('city')} for {profile.get('price')}. {profile.get('notes', '')}"
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")

    # Search
    k = min(5, len(listings))
    D, I = index.search(query_embedding, k)
    
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

@router.get("/flow/debug-listings/{city}")
async def debug_listings(city: str):
    """Debug endpoint to check listings for a city."""
    try:
        # Count total documents
        total_count = listings_collection.count_documents({})
        
        # Get unique cities (from nested structure)
        unique_cities = listings_collection.distinct("hdpData.homeInfo.city")
        
        # Count documents for the specified city (case-insensitive)
        city_count = listings_collection.count_documents({
            "hdpData.homeInfo.city": {"$regex": f"^{city}$", "$options": "i"}
        })
        
        # Get a sample listing
        sample = listings_collection.find_one({
            "hdpData.homeInfo.city": {"$regex": f"^{city}$", "$options": "i"}
        })
        if sample:
            sample["_id"] = str(sample["_id"])
        
        return {
            "total_documents": total_count,
            "city_documents": city_count,
            "unique_cities": unique_cities,
            "sample_listing": sample,
            "collection_name": listings_collection.name,
            "database_name": db.name
        }
    except Exception as e:
        logger.error(f"Error in debug_listings: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
