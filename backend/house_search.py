"""
house_search.py

Demonstrates how to parse user real-estate queries using Kor + LangChain,
invoking a DeepSeek-based LLM endpoint (either via a local Ollama server or via DeepSeek API) 
and then scrape Zillow listings with pyzill. It also stores results in MongoDB (optional).
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from enum import Enum
from typing import Optional
import logging
import sys
import os
import json
import datetime

from dotenv import load_dotenv

# Kor + LangChain
from kor import from_pydantic, create_extraction_chain
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Zillow scraping
import pyzill
from geopy.geocoders import Nominatim
import pymongo

# New approach:
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama

# === Load environment variables ===
load_dotenv()

# === Model Configuration ===
# For the local mode we use these:
DEEPSEEK_MODEL_ENDPOINT = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
DEEPSEEK_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "deepseek-r1")

# === MongoDB connection (optional) ===
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = pymongo.MongoClient(MONGO_URI)
db = client["real_estate_db"]
listings_collection = db["listings"]

# === Create a logger for our module ===
logger = logging.getLogger("house_search_logger")
logger.setLevel(logging.DEBUG)  # DEBUG, INFO, WARNING, etc.
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# === FastAPI router ===
router = APIRouter()

# === Action enum + request schema ===
class ActionType(str, Enum):
    BUY = "buy"
    RENT = "rent"

class HouseRequest(BaseModel):
    """
    User's real-estate query schema.
    """
    action: Optional[ActionType] = None
    city: Optional[str] = None
    state: Optional[str] = None
    beds: Optional[int] = None
    baths: Optional[int] = None
    max_price: Optional[int] = None

class ChatRequest(BaseModel):
    """
    User's chat query schema.
    """
    user_input: str
    context: Optional[str] = None

# === Create Kor schema + validator ===
schema, validator = from_pydantic(
    HouseRequest,
    description="Extract real-estate search parameters from natural language queries",
    examples=[
        (
            "I want to buy a 2 bedroom 2 bath house in Miami under 500000",
            {"action": "buy", "city": "Miami", "state": None, "beds": 2, "baths": 2, "max_price": 500000}
        ),
        (
            "Looking to rent a 1 bed apartment in Seattle for 2000",
            {"action": "rent", "city": "Seattle", "state": None, "beds": 1, "baths": None, "max_price": 2000}
        ),
        (
            "Show me 3 bedroom houses in Albany, NY under 300000",
            {"action": "buy", "city": "Albany", "state": "NY", "beds": 3, "baths": None, "max_price": 300000}
        )
    ],
    many=False
)

# === Initialize the LLM, extraction chain, and chat model based on the MODE ===
# Default mode is "local" (DeepSeek via Ollama). When MODEL_MODE is set to "api", we use the DeepSeek API.
MODEL_MODE = os.getenv("MODEL_MODE", "local").lower()
if MODEL_MODE not in ("local", "api"):
    logger.warning("Invalid MODEL_MODE provided. Defaulting to local.")
    MODEL_MODE = "local"

if MODEL_MODE == "local":
    try:
        logger.info(f"Initializing local model {DEEPSEEK_MODEL_NAME} via Ollama...")
        # (Optional) Attempt to pull model if missing
        import requests
        try:
            response = requests.post(
                f"{DEEPSEEK_MODEL_ENDPOINT}/api/generate",
                json={"model": DEEPSEEK_MODEL_NAME, "prompt": "test"}
            )
            if response.status_code == 404:
                logger.info(f"Model {DEEPSEEK_MODEL_NAME} not found. Attempting to pull...")
                pull_response = requests.post(
                    f"{DEEPSEEK_MODEL_ENDPOINT}/api/pull",
                    json={"name": DEEPSEEK_MODEL_NAME}
                )
                if pull_response.status_code == 200:
                    logger.info(f"Successfully pulled model {DEEPSEEK_MODEL_NAME}")
                else:
                    raise Exception(f"Failed to pull model: {pull_response.text}")
        except Exception as pull_err:
            logger.error(f"Error pulling model: {str(pull_err)}")
            raise

        llm = Ollama(
            base_url=DEEPSEEK_MODEL_ENDPOINT,
            model=DEEPSEEK_MODEL_NAME,
            temperature=0
        )
        extraction_chain = create_extraction_chain(
            llm,
            schema,
            validator=validator,
            encoder_or_encoder_class="json"
        )
        chat_llm = ChatOllama(
            base_url=DEEPSEEK_MODEL_ENDPOINT,
            model=DEEPSEEK_MODEL_NAME,
            temperature=0.7
        )
        logger.info("Local DeepSeek LLM and extraction chain initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing local LLM: {str(e)}", exc_info=True)
        raise
else:  # MODEL_MODE == "api"
    try:
        # Get the API key from the environment. If you set this key to "gpt", the code will use GPT integration.
        api_key = os.getenv("DEEPSEEK_API_KEY", "<YOUR_API_KEY_HERE>")
        if api_key.strip().lower() == "gpt":
            logger.info("Initializing GPT API LLM via ChatOpenAI...")
            llm = ChatOpenAI(
                openai_api_key=os.getenv("OPENAI_API_KEY", "<OPENAI_API_KEY_HERE>"),
                openai_api_base="https://api.openai.com/v1",
                model_name="gpt-3.5-turbo",
                temperature=0
            )
        else:
            logger.info("Initializing DeepSeek API LLM via ChatOpenAI...")
            llm = ChatOpenAI(
                openai_api_key=api_key,
                openai_api_base="https://api.deepseek.com/v1",  # hypothetical DeepSeek API URL
                model_name="deepseek-chat",
                temperature=0
            )
        extraction_chain = create_extraction_chain(
            llm,
            schema,
            validator=validator,
            encoder_or_encoder_class="json"
        )
        # For chat, we reuse the same LLM instance:
        chat_llm = llm
        logger.info("API-based LLM and extraction chain initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing API-based LLM: {str(e)}", exc_info=True)
        raise

# === Helper: bounding box from city name ===
def get_bounding_box_for_city(city_name: str, state_name: Optional[str] = None):
    geolocator = Nominatim(user_agent="my_zillow_scraper")
    full_location_string = city_name if not state_name else f"{city_name}, {state_name}"
    location = geolocator.geocode(full_location_string, addressdetails=True)
    if not location or "boundingbox" not in location.raw:
        return None
    south_lat, north_lat, west_long, east_long = location.raw["boundingbox"]
    return (float(north_lat), float(east_long), float(south_lat), float(west_long))

# === Pydantic model for incoming user text ===
class SearchRequest(BaseModel):
    user_input: str
    force_refresh: bool = False

def sanitize_keys(d):
    """Recursively replace dots and dollar signs in dictionary keys."""
    if isinstance(d, dict):
        return {k.replace('.', '_').replace('$', '_'): sanitize_keys(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [sanitize_keys(i) for i in d]
    else:
        return d

# === Main route (POST /house-search) ===
@router.post("/house-search")
async def house_search(request: SearchRequest):
    user_input = request.user_input.strip()
    logger.info(f"Received POST /house-search with user_input: {user_input}")
    try:
        # 1) Invoke the Kor extraction chain
        result = extraction_chain.invoke(user_input)
        house_req: HouseRequest = result.get("validated_data")
        if not house_req or not house_req.city:
            logger.warning("City name was not extracted from user_input; returning 400.")
            raise HTTPException(status_code=400, detail="City name is required or was not extracted.")

        # 2) Check for existing listings (cache)
        query = {
            "query_city": house_req.city,
            "query_beds": house_req.beds,
            "query_baths": house_req.baths,
            "query_max_price": house_req.max_price,
            "search_type": house_req.action or ActionType.BUY
        }
        query = {k: v for k, v in query.items() if v is not None}
        existing_count = listings_collection.count_documents(query)
        if not request.force_refresh and existing_count > 0:
            logger.info(f"Found {existing_count} existing listings for {house_req.city}")
            existing_docs = list(listings_collection.find(query))
            for doc in existing_docs:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
            return {
                "parsed_query": house_req.dict(),
                "results_count": len(existing_docs),
                "results": existing_docs,
                "stored_count": 0,
                "from_cache": True
            }

        # 3) If no existing listings, proceed with Zillow scraping
        logger.info(f"No existing listings found for {house_req.city}. Proceeding with Zillow scrape.")
        bb = get_bounding_box_for_city(house_req.city, house_req.state)
        logger.debug(f"Bounding box for city '{house_req.city}': {bb}")
        if bb:
            ne_lat, ne_long, sw_lat, sw_long = bb
            zoom_value = 10
        else:
            logger.info(f"No bounding box found for '{house_req.city}', defaulting to entire US.")
            ne_lat, ne_long = 49.3457868, -66.9513812
            sw_lat, sw_long = 24.7433195, -124.7844079
            zoom_value = 5

        search_params = {
            "pagination": 1,
            "search_value": house_req.city,
            "min_beds": house_req.beds,
            "max_beds": house_req.beds,
            "min_bathrooms": house_req.baths,
            "max_bathrooms": house_req.baths,
            "min_price": None,
            "max_price": house_req.max_price,
            "ne_lat": ne_lat,
            "ne_long": ne_long,
            "sw_lat": sw_lat,
            "sw_long": sw_long,
            "zoom_value": zoom_value
        }
        logger.info(f"Zillow search params: {search_params}")

        try:
            if house_req.action == ActionType.RENT:
                results = pyzill.for_rent(**search_params)
            else:
                results = pyzill.for_sale(**search_params)
            logger.debug(f"Zillow raw results: {json.dumps(results, indent=2)}")
        except Exception as e:
            logger.error(f"Zillow search error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Zillow search failed: {str(e)}")

        map_results = results.get("mapResults", [])
        logger.info(f"Returning {len(map_results)} results to the client.")

        try:
            stored_count = 0
            for listing in map_results:
                zpid = listing.get("zpid")
                if not zpid:
                    logger.warning("Skipping listing without zpid.")
                    continue
                sanitized_listing = sanitize_keys(listing)
                sanitized_listing.update({
                    "source": "zillow",
                    "query_city": house_req.city,
                    "query_state": house_req.state,
                    "query_beds": house_req.beds,
                    "query_baths": house_req.baths,
                    "query_max_price": house_req.max_price,
                    "search_type": house_req.action or ActionType.BUY,
                    "timestamp": datetime.datetime.utcnow()
                })
                try:
                    result = listings_collection.update_one(
                        {"zpid": zpid},
                        {"$set": sanitized_listing},
                        upsert=True
                    )
                    stored_count += 1
                    logger.debug(
                        f"{'Inserted' if result.upserted_id else 'Updated'} listing with zpid: {zpid}"
                    )
                except Exception as mongo_err:
                    logger.error(f"MongoDB error for zpid {zpid}: {str(mongo_err)}")
                    continue
            logger.info(f"Successfully stored {stored_count}/{len(map_results)} listings in MongoDB")
        except Exception as e:
            logger.error(f"Error storing listings in MongoDB: {str(e)}")
            pass

        return {
            "parsed_query": house_req.dict(),
            "results_count": len(map_results),
            "results": map_results,
            "stored_count": stored_count,
            "from_cache": False
        }

    except HTTPException as http_err:
        logger.error(f"HTTPException in house_search: {http_err.detail}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error in house_search: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# === Chat route remains similar ===
@router.post("/chat")
async def chat_route(request: ChatRequest):
    try:
        if not chat_llm:
            raise HTTPException(
                status_code=503,
                detail="Chat model not initialized. Please ensure the service is running."
            )
        logger.info(f"Received chat request: {request.user_input}")
        system_prompt = (
            "You are REA (Real Estate Assistant), an AI specializing in real estate. "
            "You help users understand property listings, market trends, and answer questions about homes. "
            "Be concise, professional, and helpful. If you're not sure about something, say so."
        )
        if request.context:
            system_prompt += f"\nContext about the current conversation: {request.context}"
        try:
            response = chat_llm.predict(
                f"{system_prompt}\n\nUser: {request.user_input}\nAssistant:"
            )
            logger.info("Generated response for chat request")
            return {
                "response": response,
                "success": True
            }
        except Exception as llm_err:
            logger.error(f"Error generating chat response: {str(llm_err)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate response: {str(llm_err)}"
            )
    except HTTPException as http_err:
        logger.error(f"HTTPException in chat_route: {http_err.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat_route: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Chat error: {str(e)}"
        )