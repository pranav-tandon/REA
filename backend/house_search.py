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
from langchain.chat_models import ChatOpenAI

# Zillow scraping
import pyzill
from geopy.geocoders import Nominatim

# MongoDB connection (adjust if needed)
MONGO_URI = "mongodb://localhost:27017/"
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
    openai_api_base="http://localhost:11411",  # Ollama's default endpoint
    openai_api_key="unused",                  # Ollama doesn't require a key
    model_name="llama2",                      # or any model you have installed
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
    ne_lat = float(north_lat)
    sw_lat = float(south_lat)
    ne_long = float(east_long)
    sw_long = float(west_long)
    return ne_lat, ne_long, sw_lat, sw_long

# === Request model for user input text ===
class SearchRequest(BaseModel):
    user_input: str

# === Main route (POST /house-search) ===
@router.post("/house-search")
def house_search(request: SearchRequest):
    user_query = request.user_input

    # 1) Parse user input with Kor
    chain_result = extraction_chain({"input": user_query})
    parsed_data = chain_result["validated_data"]
    if not parsed_data:
        raise HTTPException(status_code=400, detail="Could not parse user query.")

    # Extract fields
    city_name = parsed_data.city
    action = parsed_data.action
    beds = parsed_data.beds
    baths = parsed_data.baths
    max_price = parsed_data.max_price

    # 2) Get bounding box
    if city_name:
        bb = get_bounding_box_for_city(city_name)
    else:
        bb = None

    if bb:
        ne_lat, ne_long, sw_lat, sw_long = bb
        zoom_value = 10
    else:
        # fallback to entire US
        ne_lat, ne_long = 49.3457868, -66.9513812
        sw_lat, sw_long = 24.7433195, -124.7844079
        zoom_value = 5

    pagination = 1
    search_value = city_name or ""
    min_beds = beds
    max_beds = beds
    min_bathrooms = baths
    max_bathrooms = baths
    min_price = None

    # 3) Use pyzill to scrape data
    if action == ActionType.BUY:
        results = pyzill.for_sale(
            pagination,
            search_value,
            min_beds,
            max_beds,
            min_bathrooms,
            max_bathrooms,
            min_price,
            max_price,
            ne_lat,
            ne_long,
            sw_lat,
            sw_long,
            zoom_value
        )
    else:
        results = pyzill.for_rent(
            pagination,
            search_value,
            min_beds,
            max_beds,
            min_bathrooms,
            max_bathrooms,
            min_price,
            max_price,
            ne_lat,
            ne_long,
            sw_lat,
            sw_long,
            zoom_value
        )

    map_results = results.get("mapResults", [])

    # 4) Store in MongoDB (optional)
    for r in map_results:
        doc = {
            "address": r.get("address"),
            "price": r.get("price"),
            "beds": r.get("beds"),
            "baths": r.get("baths"),
            "zpid": r.get("zpid"),
            "source": "zillow_scraper",
        }
        listings_collection.insert_one(doc)

    return {
        "parsed_query": parsed_data.dict(),
        "results_count": len(map_results),
        "results": map_results
    }

load_dotenv()

deepseek_llm = ChatOpenAI(
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
    model_name=os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat"),
    temperature=0
)

@router.post("/house-search")
def house_search_route(user_message: str):
    # Use deepseek_llm to parse the user's real-estate parameters
    response = deepseek_llm.predict(user_message)
    return {"response": response}
