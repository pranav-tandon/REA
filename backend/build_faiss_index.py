# backend/build_faiss_index.py
import pymongo
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from bson import ObjectId
from dotenv import load_dotenv
import os

load_dotenv()

def build_faiss_index(
    mongo_uri=os.getenv("MONGO_URI", "mongodb://localhost:27017/"),
    db_name="real_estate_db",
    index_file="listings_index.faiss",
    doc_ids_file="doc_ids.npy"
):
    # 1. Connect to Mongo
    client = pymongo.MongoClient(mongo_uri)
    db = client[db_name]
    listings_coll = db["listings"]

    # 2. Fetch listings
    listings = list(listings_coll.find())
    if not listings:
        print("No listings found in MongoDB. Add data first!")
        return
    
    # 3. Prepare text data + doc IDs
    text_data = []
    doc_ids = []
    for listing in listings:
        # Combine relevant fields. Adjust as needed.
        address = listing.get("address", "")
        desc = listing.get("description", "")
        combined_text = f"{address}. {desc}"
        text_data.append(combined_text)
        doc_ids.append(str(listing["_id"]))  # store as string for reference

    # 4. Embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")  # or any other
    embeddings = model.encode(text_data)
    embeddings = np.array(embeddings, dtype='float32')

    # 5. Create & train the FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # 6. Save index + doc IDs
    faiss.write_index(index, index_file)
    np.save(doc_ids_file, np.array(doc_ids))
    
    print(f"FAISS index built and saved to {index_file}")
    print(f"Document IDs saved to {doc_ids_file}")

if __name__ == "__main__":
    build_faiss_index()
