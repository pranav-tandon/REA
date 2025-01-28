#!/usr/bin/env bash
#
# apply_deepseek_kor_ollama.sh
#
# Purpose: Demonstration script to add the "DeepSeek + Kor + pyzill" logic
# into an existing REA project, ignoring old scrapers & FAISS for now.
#
# Usage:
#   1) cd /path/to/REA
#   2) chmod +x apply_deepseek_kor_ollama.sh
#   3) ./apply_deepseek_kor_ollama.sh

set -e

echo "=== Step 1: Append dependencies to backend/requirements.txt if not present..."

REQ_FILE="backend/requirements.txt"

# Helper function to append only if the line does not already exist
function add_line {
  local PKG="$1"
  if ! grep -q "${PKG}" "${REQ_FILE}" 2>/dev/null; then
    echo "${PKG}" >> "${REQ_FILE}"
    echo "  Added: ${PKG}"
  else
    echo "  Already in requirements.txt: ${PKG}"
  fi
}

add_line "kor==0.7.1"
add_line "langchain==0.0.103"
add_line "geopy==2.3.0"
add_line "pyzill==0.0.7"
# Additional dependencies you may need:
# add_line "openai==0.27.0"

echo ""
echo "=== Step 2: Create/overwrite backend/house_search.py with Kor + LangChain + pyzill ==="

cat <<'EOF' > backend/house_search.py
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
EOF

echo ""
echo "=== Step 3: Patch backend/app.py to include our new route (house_search.py). ==="

APP_FILE="backend/app.py"

# 1) Ensure there's a line to import our router
if ! grep -q "import house_search" "$APP_FILE" 2>/dev/null; then
  echo "  Patching: adding 'import house_search'"
  sed -i.bak '/from fastapi import FastAPI/a \
from house_search import router as house_search_router
' "$APP_FILE"
fi

# 2) Ensure we include the router in the FastAPI app
if ! grep -q "app.include_router(house_search_router)" "$APP_FILE" 2>/dev/null; then
  echo "  Patching: adding 'app.include_router(house_search_router)'"
  sed -i.bak '/app = FastAPI()/a \
app.include_router(house_search_router)
' "$APP_FILE"
fi

# Cleanup backup
rm -f backend/app.py.bak

echo ""
echo "=== Step 4: Overwrite nextjs/pages/api/houseSearch.ts with a new route. ==="

cat <<'EOF' > nextjs/pages/api/houseSearch.ts
import type { NextApiRequest, NextApiResponse } from "next";

/**
 * Forwards user queries to FastAPI's /house-search endpoint.
 */
export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method === "POST") {
    try {
      const backendRes = await fetch("http://localhost:8000/house-search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req.body),
      });
      const data = await backendRes.json();
      return res.status(200).json(data);
    } catch (error) {
      console.error(error);
      return res.status(500).json({ error: "Backend service error" });
    }
  } else {
    return res.status(405).json({ error: "Method not allowed" });
  }
}
EOF

echo ""
echo "=== Step 5: Overwrite nextjs/pages/index.tsx with a minimal example. ==="

cat <<'EOF' > nextjs/pages/index.tsx
import { useState } from "react";

export default function Home() {
  const [userInput, setUserInput] = useState("");
  const [searchResults, setSearchResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    setLoading(true);
    setSearchResults(null);
    try {
      const res = await fetch("/api/houseSearch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_input: userInput }),
      });
      const data = await res.json();
      setSearchResults(data);
    } catch (err) {
      console.error(err);
      alert("Error searching.");
    }
    setLoading(false);
  };

  return (
    <main style={{ padding: "1rem" }}>
      <h1>REA - House Search</h1>
      <textarea
        rows={3}
        style={{ width: "100%", marginBottom: "0.5rem" }}
        value={userInput}
        onChange={(e) => setUserInput(e.target.value)}
        placeholder="E.g. 'Looking to rent a 2 bedroom in Seattle for 2000'..."
      />
      <div>
        <button onClick={handleSearch} disabled={loading}>
          {loading ? "Searching..." : "Search"}
        </button>
      </div>
      {searchResults && (
        <div style={{ marginTop: "1rem" }}>
          <h3>Parsed Query:</h3>
          <pre>{JSON.stringify(searchResults.parsed_query, null, 2)}</pre>

          <h3>Results:</h3>
          <p>Found {searchResults.results_count} listings</p>
          {searchResults.results.map((r: any, idx: number) => (
            <div
              key={idx}
              style={{ border: "1px solid #ccc", margin: "1rem 0", padding: "1rem" }}
            >
              <p><strong>Address:</strong> {r.address}</p>
              <p><strong>Price:</strong> {r.price}</p>
              <p><strong>Beds:</strong> {r.beds}</p>
              <p><strong>Baths:</strong> {r.baths}</p>
            </div>
          ))}
        </div>
      )}
    </main>
  );
}
EOF

echo ""
echo "=== Done! ==="
echo "Summary of actions:"
echo "1) Updated backend/requirements.txt to include kor, langchain, geopy, pyzill."
echo "2) Created backend/house_search.py containing the new route & logic."
echo "3) Patched backend/app.py to import and include the new router."
echo "4) Created nextjs/pages/api/houseSearch.ts for bridging Next.js -> FastAPI."
echo "5) Overwrote nextjs/pages/index.tsx with a minimal UI."
echo ""
echo "Now you can run something like:"
echo "  cd backend && source venv/bin/activate && pip install -r requirements.txt"
echo "  uvicorn app:app --host 0.0.0.0 --port 8000"
echo "Then in another terminal:"
echo "  cd ../nextjs && npm install && npm run dev"
echo "Navigate to http://localhost:3000 and test your new House Search!"
