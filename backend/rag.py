# backend/rag.py

import os
import json
import time
import faiss
import openai
import pickle
import numpy as np
import pymongo
import datetime
import logging

from typing import List, Dict, Any
from dotenv import load_dotenv
from datasets import load_dataset

# For building TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# For embeddings
from sentence_transformers import SentenceTransformer

###############################################################################
#                              CONFIG & SETUP
###############################################################################

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = pymongo.MongoClient(MONGO_URI)
db = client["real_estate_db"]

profiles_coll = db["profiles"]         # user constraints
conversation_coll = db["conversations"]
listings_coll = db["listings"]         # if you store any additional listings

# Paths to store index artifacts
RESOURCES_DIR = os.path.join(os.path.dirname(__file__), "resources")
os.makedirs(RESOURCES_DIR, exist_ok=True)

FAISS_INDEX_PATH = os.path.join(RESOURCES_DIR, "faiss_index.bin")
TFIDF_DATA_PATH = os.path.join(RESOURCES_DIR, "tfidf_data.pkl")
DOC_EMBEDDINGS_PATH = os.path.join(RESOURCES_DIR, "doc_embeddings.pkl")

# Default embedding model for FAISS
EMBED_MODEL_NAME = "sentence-transformers/msmarco-distilbert-dot-v5"

###############################################################################
#                      1) BUILD THE INDEX (Doc Embeddings)
###############################################################################
def build_index_from_zillow_dataset():
    """
    One-time function that:
      1) Loads the 'zillow/real_estate_v1' dataset from Hugging Face
      2) Builds TF-IDF vectors (lexical)
      3) Builds a FAISS index with sentence-transformers embeddings (dense)
      4) Saves them to backend/resources
    """
    logging.info("Loading dataset: zillow/real_estate_v1")
    ds = load_dataset("zillow/real_estate_v1", split="train")
    
    # Debug: Print first row to inspect available fields
    logging.info(f"Sample row from dataset: {ds[0]}")
    
    logging.info("Preparing docs from dataset...")
    docs = []
    for i, row in enumerate(ds):
        # More robust text field construction
        description = row.get('description', '').strip()
        address = row.get('address', '').strip()
        city = row.get('city', '').strip()
        state = row.get('state', '').strip()
        price = row.get('price', '')
        beds = row.get('beds', '')
        baths = row.get('baths', '')
        
        # Build a richer text field for better matching
        text_parts = []
        if description:
            text_parts.append(description)
        if address:
            text_parts.append(f"Address: {address}")
        if city and state:
            text_parts.append(f"Location: {city}, {state}")
        if price:
            text_parts.append(f"Price: ${price:,.2f}" if isinstance(price, (int, float)) else f"Price: {price}")
        if beds:
            text_parts.append(f"Beds: {beds}")
        if baths:
            text_parts.append(f"Baths: {baths}")
            
        text = "\n".join(text_parts)
        
        # Skip empty documents
        if not text.strip():
            continue
            
        doc = {
            "id": f"doc_{i}",
            "text": text,
            "address": address or "N/A",
            "city": city,
            "state": state,
            "price": price,
            "beds": beds,
            "baths": baths,
            "raw": row
        }
        docs.append(doc)
    
    logging.info(f"Processed {len(docs)} valid documents")
    if not docs:
        raise ValueError("No valid documents found in dataset!")

    # ------------------- Build TF-IDF -------------------
    texts = [d["text"] for d in docs]
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

    # ------------------- Build FAISS --------------------
    logging.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    logging.info(f"Embedding {len(docs)} docs...")
    doc_embs = embed_model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    dim = doc_embs.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)  # dot-product if normalized
    faiss_index.add(doc_embs)
    logging.info(f"FAISS index size: {faiss_index.ntotal}")

    # ------------------- Save artifacts ------------------
    logging.info("Saving FAISS index to %s", FAISS_INDEX_PATH)
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)

    logging.info("Saving TF-IDF data to %s", TFIDF_DATA_PATH)
    tfidf_data = {
        "vectorizer": tfidf_vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "docs": docs
    }
    with open(TFIDF_DATA_PATH, "wb") as f:
        pickle.dump(tfidf_data, f)

    logging.info("Saving doc_embeddings to %s", DOC_EMBEDDINGS_PATH)
    with open(DOC_EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(doc_embs, f)

    logging.info("Done building index for zillow/real_estate_v1.")

###############################################################################
#                    2) HYBRID SEARCH CLASS (TF-IDF + FAISS)
###############################################################################
class ZillowHybridSearch:
    """
    Combines:
      - TF-IDF for lexical search
      - FAISS for dense search
      - Merges results in a naive way
    """
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self.model_name = model_name
        self.embed_model = None
        self.docs: List[Dict[str, Any]] = []
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.faiss_index = None

    def load_resources(self):
        """
        Load the saved TF-IDF and FAISS index from disk (backend/resources).
        """
        # Load TF-IDF
        with open(TFIDF_DATA_PATH, "rb") as f:
            tfidf_data = pickle.load(f)
            self.tfidf_vectorizer = tfidf_data["vectorizer"]
            self.tfidf_matrix = tfidf_data["tfidf_matrix"]
            self.docs = tfidf_data["docs"]

        # Load FAISS
        self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)

        # Lazy-load embedding model
        self.embed_model = SentenceTransformer(self.model_name)
        logging.info("[ZillowHybridSearch] Loaded all resources successfully.")
        logging.info(f"[ZillowHybridSearch] {len(self.docs)} docs total.")

    def lexical_search(self, query: str, top_k: int = 10):
        if self.tfidf_matrix is None:
            logging.error("[ZillowHybridSearch] No TF-IDF matrix loaded!")
            return []
        q_vec = self.tfidf_vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        top_idx = sims.argsort()[-top_k:][::-1]
        results = []
        for idx in top_idx:
            doc = self.docs[idx]
            score = float(sims[idx])
            results.append({**doc, "score": score, "source": "lexical"})
        return results

    def dense_search(self, query: str, top_k: int = 10):
        if self.faiss_index is None:
            logging.error("[ZillowHybridSearch] No FAISS index loaded!")
            return []
        query_emb = self.embed_model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

        distances, indices = self.faiss_index.search(query_emb, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            score = float(distances[0][i])
            doc = self.docs[idx]
            results.append({**doc, "score": score, "source": "dense"})
        return results

    def hybrid_search(self, query: str, top_k: int = 10, alpha: float = 0.5):
        """
        Naive hybrid:
          - get top_k from lexical
          - get top_k from dense
          - unify by doc['id'], combine scores
        alpha = weighting factor for dense vs lexical
        """
        lex_res = self.lexical_search(query, top_k=top_k)
        den_res = self.dense_search(query, top_k=top_k)
        
        # Debug logging
        logging.debug(f"Lexical results: {len(lex_res)} docs")
        logging.debug(f"Dense results: {len(den_res)} docs")
        
        merged = {}

        # Insert lexical with score normalization
        max_lex_score = max((r["score"] for r in lex_res), default=1.0)
        for r in lex_res:
            _id = r["id"]
            # Normalize lexical score to [0,1]
            norm_score = r["score"] / max_lex_score if max_lex_score > 0 else 0
            merged[_id] = {
                "doc": r,
                "lex_score": norm_score,
                "dense_score": 0.0
            }

        # Insert dense with score normalization
        max_dense_score = max((r["score"] for r in den_res), default=1.0)
        for r in den_res:
            _id = r["id"]
            # Normalize dense score to [0,1]
            norm_score = r["score"] / max_dense_score if max_dense_score > 0 else 0
            if _id not in merged:
                merged[_id] = {
                    "doc": r,
                    "lex_score": 0.0,
                    "dense_score": norm_score
                }
            else:
                merged[_id]["dense_score"] = norm_score

        final_list = []
        for k, v in merged.items():
            doc_data = v["doc"]
            combined_score = alpha * v["dense_score"] + (1 - alpha) * v["lex_score"]
            doc_data["score"] = combined_score
            doc_data["lex_score"] = v["lex_score"]
            doc_data["dense_score"] = v["dense_score"]
            final_list.append(doc_data)

        final_list.sort(key=lambda x: x["score"], reverse=True)
        return final_list[:top_k]

###############################################################################
#         3) UTILITY: RETRIEVE CONVERSATION SNIPPETS for RAG CONTEXT
###############################################################################

embed_model_for_conv = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_relevant_conversation_snippets(user_id: str, top_k: int = 3) -> str:
    """
    Example that fetches conversation from Mongo, 
    does an in-memory FAISS embed & search, returns top_k.
    """
    conv_docs = list(conversation_coll.find({"user_id": user_id}))
    if not conv_docs:
        return ""

    texts = [doc.get("text", "") for doc in conv_docs]
    if not texts:
        return ""

    embeddings = embed_model_for_conv.encode(texts, convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    big_query_text = " ".join(texts)
    query_emb = embed_model_for_conv.encode([big_query_text], convert_to_numpy=True).astype("float32")

    _, I = index.search(query_emb, min(top_k, len(conv_docs)))

    relevant_snippets = []
    for idx in I[0]:
        relevant_snippets.append(texts[idx])

    return "\n".join(relevant_snippets)

###############################################################################
#   4) GPT CLASSIFICATION / RE-RANKING for final top-10 listings
###############################################################################

def classify_properties_with_gpt(
    properties: List[Dict[str, Any]],
    profile_context: str,
    conversation_context: str
) -> List[Dict[str, Any]]:
    """
    Takes the top properties, plus a user profile context and conversation context,
    and calls GPT to re-rank with a numeric score & rationale. Returns top 10.
    """
    BATCH_SIZE = 50
    all_classifications = []
    client = openai.OpenAI()  # Create client instance

    def chunk_list(lst, chunk_size):
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]

    for batch in chunk_list(properties, BATCH_SIZE):
        prompt = f"""
You are an expert real estate analyst. 
Below is the user's profile constraints (the main requirements):
---
{profile_context}
---

Below is conversation context from this user:
---
{conversation_context}
---

Now here is a list of property listings in JSON format. 
Each property has keys: "address", "beds", "baths", "price", "unique_features".

For EACH property, please provide:
- "address": property address
- "score": a numeric score from 1 (poor match) to 10 (excellent match)
- "rationale": short explanation

Output your response as valid JSON array of objects:
[
  {{ "address": "...", "score": 7, "rationale": "...(why)..." }},
  ...
]
        """
        prompt += "\n\nProperties:\n"
        prompt += json.dumps(batch, indent=2)
        prompt += "\n\nOutput (as JSON):"

        messages = [
            {"role": "system", "content": "You are an expert real estate analyst."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = client.chat.completions.create(  # Updated API call
                model="gpt-4",
                messages=messages,
                temperature=0.0,
                max_tokens=2000
            )
            output_text = response.choices[0].message.content.strip()  # Updated response access
            chunk_result = json.loads(output_text)
            all_classifications.extend(chunk_result)
        except Exception as e:
            print("Error in GPT call or JSON parse:", e)
            logging.error(f"GPT API error: {str(e)}", exc_info=True)
        time.sleep(1)  # be polite to rate limits

    for prop in all_classifications:
        try:
            prop['score'] = float(prop.get('score', 0))
        except:
            prop['score'] = 0

    # return top 10 by score
    top_10 = sorted(all_classifications, key=lambda x: x['score'], reverse=True)[:10]
    return top_10

###############################################################################
#   5) RAG LOGIC: GATHER + GPT CLASSIFICATION (Profile + Conversation)
###############################################################################

def build_listing_query_from_profile(profile_doc: dict) -> dict:
    """
    Convert profile doc into a MongoDB query (if you also store listings in DB).
    Adjust as needed. 
    """
    query = {}
    city = profile_doc.get("city")
    state = profile_doc.get("state")
    if city:
        query["city"] = {"$regex": f"^{city}$", "$options": "i"}
    if state:
        query["state"] = {"$regex": f"^{state}$", "$options": "i"}

    price_constraint = profile_doc.get("price")
    action = profile_doc.get("actionType", "").lower()
    if price_constraint and isinstance(price_constraint, (int, float)):
        if action in ("buy", "sell"):
            query["price"] = {"$lte": price_constraint}
        elif action == "rent":
            query["price"] = {"$lte": price_constraint}

    if "beds" in profile_doc and profile_doc["beds"]:
        query["beds"] = {"$gte": profile_doc["beds"]}
    if "baths" in profile_doc and profile_doc["baths"]:
        query["baths"] = {"$gte": profile_doc["baths"]}

    zip_code = profile_doc.get("zipCode")
    if zip_code:
        query["zip_code"] = {"$regex": f"^{zip_code}$", "$options": "i"}

    return query

def get_top_10_with_profile_rag(profile_id: str) -> List[Dict[str,Any]]:
    """
    1) Load profile from 'profiles'
    2) Filter listings in DB or do a hybrid search if you want
    3) Retrieve conversation context
    4) Classify final with GPT
    5) Return top 10
    """
    profile_doc = profiles_coll.find_one({"_id": pymongo.ObjectId(profile_id)})
    if not profile_doc:
        return []

    user_id = profile_doc.get("userId") or profile_doc.get("user_id", "unknown")

    # Summarize constraints
    constraint_text = f"""
Action: {profile_doc.get('actionType', 'N/A')}
City: {profile_doc.get('city', '')}
State: {profile_doc.get('state', '')}
Price: {profile_doc.get('price', '')}
Beds: {profile_doc.get('beds', '')}
Baths: {profile_doc.get('baths', '')}
ZipCode: {profile_doc.get('zipCode', '')}
SchoolDistrict: {profile_doc.get('schoolDistrict', '')}
"""

    # We can either do a hybrid search or do a direct MongoDB query
    query = build_listing_query_from_profile(profile_doc)
    raw_listings = list(listings_coll.find(query).limit(500))

    if not raw_listings:
        return []

    # Prepare for GPT
    props_for_gpt = []
    for lst in raw_listings:
        doc = {
            "address": lst.get("address", "N/A"),
            "beds": lst.get("beds", 0),
            "baths": lst.get("baths", 0),
            "price": lst.get("price", 0),
            "unique_features": lst.get("unique_features", [])
        }
        props_for_gpt.append(doc)

    # Gather conversation
    conversation_context = retrieve_relevant_conversation_snippets(user_id, top_k=3)

    # GPT classify
    top_10 = classify_properties_with_gpt(
        props_for_gpt,
        profile_context=constraint_text,
        conversation_context=conversation_context
    )
    return top_10


###############################################################################
#                                USAGE
###############################################################################
if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # 1) Build index once (comment out if already built):
    # build_index_from_zillow_dataset()  # Commented out to avoid rebuilding

    # 2) Then do a quick test of HybridSearch
    searcher = ZillowHybridSearch()
    searcher.load_resources()

    # Test data - sample listings
    test_listings = [
        {
            "address": "1639 Clarence Ave, Berwyn, IL 60402",
            "price": 309900.0,
            "beds": 2,
            "baths": 1.0,
            "area": 1295,
            "description": "Finished basement",
            "city": "Berwyn",
            "state": "IL",
            "zip_code": "60402"
        },
        {
            "address": "4529.5 S Drexel Ave #2W, Chicago, IL 60653",
            "price": 149900.0,
            "beds": 2,
            "baths": 1.0,
            "area": 900,
            "description": "Price reduced by $5,000",
            "city": "Chicago", 
            "state": "IL",
            "zip_code": "60653"
        },
        {
            "address": "2517 S 4th Ave, North Riverside, IL 60546",
            "price": 329900.0,
            "beds": 2,
            "baths": 1.0,
            "area": 1124,
            "description": "Partial finished basement",
            "city": "North Riverside",
            "state": "IL",
            "zip_code": "60546"
        }
    ]

    # Test queries
    test_queries = [
        "Looking for a 2 bedroom house under $200,000 in Chicago",
        "Need a property with finished basement in Berwyn",
        "Affordable condo in Chicago with 2 beds"
    ]

    print("\n[TESTING HYBRID SEARCH WITH SAMPLE DATA]")
    for query in test_queries:
        print(f"\nQuery: {query}")
        hybrid_results = searcher.hybrid_search(query, top_k=2, alpha=0.6)
        for i, r in enumerate(hybrid_results):
            print(f"\n{i+1}. Score={r.get('score', 0):.3f}")
            print(f"   Address: {r.get('address', 'N/A')}")
            
            # Safe price formatting
            price = r.get('price')
            if price is not None and isinstance(price, (int, float)):
                print(f"   Price: ${price:,.2f}")
            else:
                print("   Price: N/A")
            
            # Safe beds/baths formatting
            beds = r.get('beds', 'N/A')
            baths = r.get('baths', 'N/A')
            print(f"   Beds/Baths: {beds}/{baths}")

    # Test GPT classification
    print("\n[TESTING GPT CLASSIFICATION]")
    profile_context = """
    Action: Buy
    City: Chicago
    State: IL
    Price: 200000
    Beds: 2
    Baths: 1
    """

    conversation_context = "Looking for an affordable property in Chicago. Prefer something with good public transportation access."

    top_matches = classify_properties_with_gpt(
        test_listings,
        profile_context=profile_context,
        conversation_context=conversation_context
    )

    print("\nGPT Classification Results:")
    for i, match in enumerate(top_matches):
        print(f"\n{i+1}. {match.get('address', 'N/A')}")
        print(f"   Score: {match.get('score', 0)}/10")
        print(f"   Rationale: {match.get('rationale', 'N/A')}")