from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from house_search import router as house_search_router
from pydantic import BaseModel
import requests
import pymongo
import faiss
import numpy as np
from dotenv import load_dotenv
import os

from sentence_transformers import SentenceTransformer
from typing import List

from cma import estimate_value
from neighborhood import get_neighborhood_stats

# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include only house_search router (now includes chat functionality)
app.include_router(house_search_router, prefix="")

# Use environment variables
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://localhost:11434/generate")

# Connect to MongoDB
client = pymongo.MongoClient(MONGO_URI)
db = client["real_estate_db"]
listings_collection = db["listings"]

# Load FAISS index (placeholder - handle if not existing)
try:
    index = faiss.read_index("listings_index.faiss")
except:
    index = None

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

@app.get("/")
def read_root():
    return {"message": "REA Backend API is running"}

@app.get("/valuation/{property_id}")
def valuation_endpoint(property_id: str):
    prop = listings_collection.find_one({"_id": property_id})
    if not prop:
        return {"error": "Property not found"}
    estimated = estimate_value(prop, listings_collection)
    return {"estimated_value": estimated}

@app.get("/neighborhood/{zip_code}")
def neighborhood_info(zip_code: str):
    stats = get_neighborhood_stats(zip_code, db)
    if not stats:
        return {"error": "No neighborhood stats found"}
    return stats

@app.get("/test")
async def test_route():
    return {"message": "API is working"}
