from fastapi import FastAPI
from house_search import router as house_search_router
from basic_chat import router as basic_chat_router
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
app.include_router(house_search_router)
app.include_router(basic_chat_router)

# Use environment variables
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://localhost:11411/generate")

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

class ChatRequest(BaseModel):
    user_query: str

@app.get("/")
def root():
    return {"message": "REA Backend is running!"}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    query_text = request.user_query

    # 1. Convert user query to embedding
    if not index:
        # fallback if index not built
        return {"response": "No FAISS index found, can't search listings."}

    query_emb = embed_model.encode([query_text])
    query_emb = np.array(query_emb, dtype='float32')

    # 2. Retrieve top matches from FAISS
    k = 3
    distances, indices = index.search(query_emb, k)

    matched_docs = []
    for idx in indices[0]:
        # 'idx' is the row in your original array used when building the index
        # This might require you to maintain a separate array of doc_ids
        # For placeholder, let's assume you have a global in-memory doc map
        # matched_docs.append(MY_GLOBAL_LISTINGS[idx])
        pass

    # 3. Build context for LLM
    context_text = "..."  # Combine matched docs data into a string

    # 4. Query Ollama
    payload = {
        "prompt": (
            f"You are REA, a real estate assistant.\n"
            f"User Query: {query_text}\n\n"
            f"Context: {context_text}\n\n"
            f"Answer in a helpful manner."
        )
    }

    try:
        response = requests.post(LLM_SERVER_URL, json=payload)
        llm_reply = response.json().get("generated_text", "")
    except Exception as e:
        llm_reply = f"Error calling LLM: {str(e)}"

    return {
        "response": llm_reply,
        "matched_docs": matched_docs
    }

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
