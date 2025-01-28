"""
house_search.py

Demonstrates how to parse user real-estate queries using Kor + LangChain,
invoking a DeepSeek-based LLM endpoint (OpenAI-compatible) and then
scrape Zillow listings with pyzill. It also stores results in MongoDB (optional).
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from enum import Enum
from typing import Optional
import logging
import sys
import os
import json

from dotenv import load_dotenv

# Kor + LangChain
from kor import from_pydantic, create_extraction_chain
from langchain.chat_models import ChatOpenAI

# Zillow scraping
import pyzill
from geopy.geocoders import Nominatim
import pymongo

# === Load environment variables ===
load_dotenv()

# === MongoDB connection (optional) ===
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = pymongo.MongoClient(MONGO_URI)
db = client["real_estate_db"]
listings_collection = db["listings"]

# === Create a logger for our module ===
logger = logging.getLogger("house_search_logger")
logger.setLevel(logging.DEBUG)  # DEBUG, INFO, WARNING, etc.

# Create a console handler and set its log level
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
# Optional format
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
    beds: Optional[int] = None
    baths: Optional[int] = None
    max_price: Optional[int] = None

# === Create Kor schema + validator ===
schema, validator = from_pydantic(
    HouseRequest,
    description="Extract real-estate search parameters from natural language queries",
    examples=[
        (
            "I want to buy a 2 bedroom 2 bath house in Miami under 500000",
            {"action": "buy", "city": "Miami", "beds": 2, "baths": 2, "max_price": 500000}
        ),
        (
            "Looking to rent a 1 bed apartment in Seattle for 2000",
            {"action": "rent", "city": "Seattle", "beds": 1, "baths": None, "max_price": 2000}
        ),
        (
            "Show me 3 bedroom houses in Boston",
            {"action": "buy", "city": "Boston", "beds": 3, "baths": None, "max_price": None}
        )
    ],
    many=False
)

# === Instantiate the DeepSeek-based LLM + extraction chain ===
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "<YOUR_API_KEY_HERE>")
try:
    logger.info("Initializing ChatOpenAI with DeepSeek credentials...")
    llm = ChatOpenAI(
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base="https://api.deepseek.com/v1",   # hypothetical
        model_name="deepseek-chat",                      # hypothetical
        temperature=0
    )

    extraction_chain = create_extraction_chain(
        llm,
        schema,
        validator=validator,
        encoder_or_encoder_class="json"
    )
    logger.info("DeepSeek LLM + extraction chain initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing DeepSeek LLM: {str(e)}", exc_info=True)
    raise

# === Helper: bounding box from city name ===
def get_bounding_box_for_city(city_name: str):
    geolocator = Nominatim(user_agent="my_zillow_scraper")
    location = geolocator.geocode(city_name, addressdetails=True)
    if not location or "boundingbox" not in location.raw:
        return None
    south_lat, north_lat, west_long, east_long = location.raw["boundingbox"]
    return (float(north_lat), float(east_long), float(south_lat), float(west_long))

# === Pydantic model for incoming user text ===
class SearchRequest(BaseModel):
    user_input: str

# === Main route (POST /house-search) ===
@router.post("/house-search")
async def house_search(request: SearchRequest):
    user_input = request.user_input.strip()
    logger.info(f"Received POST /house-search with user_input: {user_input}")

    try:
        # 1) Invoke the Kor extraction chain
        result = extraction_chain.invoke(user_input)

        # The chain output typically contains: 
        # {
        #   "raw": "...",               # raw text from LLM
        #   "data": { ... },            # the unvalidated JSON
        #   "errors": [],
        #   "validated_data": HouseRequest(...) 
        # }
        logger.debug(f"Raw chain output: {result.get('raw')}")
        logger.debug(f"Parsed data: {result.get('data')}")
        logger.debug(f"Validation errors: {result.get('errors')}")

        house_req: HouseRequest = result.get("validated_data")
        logger.debug(f"HouseRequest from chain: {house_req}")

        if not house_req or not house_req.city:
            logger.warning("City name was not extracted from user_input; returning 400.")
            raise HTTPException(status_code=400, detail="City name is required or was not extracted.")

        # 2) Derive bounding box
        bb = get_bounding_box_for_city(house_req.city)
        logger.debug(f"Bounding box for city '{house_req.city}': {bb}")
        if bb:
            ne_lat, ne_long, sw_lat, sw_long = bb
            zoom_value = 10
        else:
            logger.info(f"No bounding box found for '{house_req.city}', defaulting to entire US.")
            ne_lat, ne_long = 49.3457868, -66.9513812
            sw_lat, sw_long = 24.7433195, -124.7844079
            zoom_value = 5

        # 3) Build Zillow search params
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

        # 4) Execute the Zillow search
        try:
            if house_req.action == ActionType.RENT:
                results = pyzill.for_rent(**search_params)
            else:
                # default to buy
                results = pyzill.for_sale(**search_params)
            logger.debug(f"Zillow raw results: {json.dumps(results, indent=2)}")
        except Exception as e:
            logger.error(f"Zillow search error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Zillow search failed: {str(e)}")

        # 5) Return final output
        map_results = results.get("mapResults", [])
        logger.info(f"Returning {len(map_results)} results to the client.")
        return {
            "parsed_query": house_req.dict(),
            "results_count": len(map_results),
            "results": map_results
        }

    except HTTPException as http_err:
        logger.error(f"HTTPException in house_search: {http_err.detail}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error in house_search: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
