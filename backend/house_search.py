"""
house_search.py

This file demonstrates how to parse user real-estate queries using Kor + LangChain
and then scrape Zillow listings with pyzill, ignoring other scrapers & FAISS for now.
It also stores results in MongoDB (optional).
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pymongo
from enum import Enum
from typing import Optional
import os
from dotenv import load_dotenv

# Kor + LangChain
from kor import from_pydantic, create_extraction_chain
from langchain_community.chat_models import ChatOpenAI

# Zillow scraping
import pyzill
from geopy.geocoders import Nominatim
import json

# Load environment variables
load_dotenv()

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = pymongo.MongoClient(MONGO_URI)
db = client["real_estate_db"]
listings_collection = db["listings"]

router = APIRouter()

# === Define schema for user queries ===
class ActionType(str, Enum):
    BUY = "buy"
    RENT = "rent"

class HouseRequest(BaseModel):
    action: Optional[ActionType] = None
    city: Optional[str] = None
    beds: Optional[int] = None
    baths: Optional[int] = None
    max_price: Optional[int] = None

# === Create Kor schema + validator ===
schema, validator = from_pydantic(
    HouseRequest,
    description="Real-estate search parameters",
    examples=[
        (
            "I want to buy a 2 bedroom 2 bath house in Miami under 500000",
            {"action": "buy", "city": "Miami", "beds": 2, "baths": 2, "max_price": 500000}
        ),
        (
            "Looking to rent a 1 bed 1 bath in Seattle for 2000",
            {"action": "rent", "city": "Seattle", "beds": 1, "baths": 1, "max_price": 2000}
        )
    ],
    many=False
)

# === Instantiate the LLM + chain (pointing to local Ollama, for example) ===
ollama_llm = ChatOpenAI(
    openai_api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
    openai_api_key="unused",
    model_name=os.getenv("OLLAMA_MODEL_NAME", "llama2"),
    temperature=0
)

extraction_chain = create_extraction_chain(
    ollama_llm,
    schema,
    validator=validator,
    encoder_or_encoder_class="json"
)

# === Helper: bounding box from geopy ===
def get_bounding_box_for_city(city_name: str):
    geolocator = Nominatim(user_agent="my_zillow_scraper")
    location = geolocator.geocode(city_name, addressdetails=True)
    if not location or "boundingbox" not in location.raw:
        return None
    south_lat, north_lat, west_long, east_long = location.raw["boundingbox"]
    return (float(north_lat), float(east_long), float(south_lat), float(west_long))

# === Request model for user input text ===
class SearchRequest(BaseModel):
    user_input: str

# === Main route (POST /house-search) ===
@router.post("/house-search")
async def house_search(request: SearchRequest):
    try:
        # Run the extraction chain with the new syntax
        chain_output = await extraction_chain.arun(request.user_input)
        
        # Parse the JSON output
        try:
            result_dict = json.loads(chain_output) if isinstance(chain_output, str) else chain_output
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Invalid chain output format")
        
        # Extract the parsed data
        parsed_data = result_dict.get("data", {})
        if not parsed_data:
            raise HTTPException(status_code=400, detail="Could not parse user query")

        # Extract search parameters
        city_name = parsed_data.get("city")
        action = parsed_data.get("action")
        beds = parsed_data.get("beds")
        baths = parsed_data.get("baths")
        max_price = parsed_data.get("max_price")

        # Get bounding box
        if city_name:
            bb = get_bounding_box_for_city(city_name)
            if bb:
                ne_lat, ne_long, sw_lat, sw_long = bb
                zoom_value = 10
            else:
                # Fallback to entire US
                ne_lat, ne_long = 49.3457868, -66.9513812
                sw_lat, sw_long = 24.7433195, -124.7844079
                zoom_value = 5
        else:
            raise HTTPException(status_code=400, detail="City name is required")

        # Search parameters
        search_params = {
            "pagination": 1,
            "search_value": city_name,
            "min_beds": beds,
            "max_beds": beds,
            "min_bathrooms": baths,
            "max_bathrooms": baths,
            "min_price": None,
            "max_price": max_price,
            "ne_lat": ne_lat,
            "ne_long": ne_long,
            "sw_lat": sw_lat,
            "sw_long": sw_long,
            "zoom_value": zoom_value
        }

        # Execute search based on action type
        if action == ActionType.BUY:
            results = pyzill.for_sale(**search_params)
        else:
            results = pyzill.for_rent(**search_params)

        return {
            "parsed_query": parsed_data,
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
