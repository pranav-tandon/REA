import requests
from bs4 import BeautifulSoup
import pymongo

def scrape_zillow(search_url: str, mongo_uri: str="mongodb://localhost:27017/"):
    client = pymongo.MongoClient(mongo_uri)
    db = client["real_estate_db"]
    listings_collection = db["listings"]

    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    listing_cards = soup.select(".list-card-info")
    for card in listing_cards:
        address = card.select_one(".list-card-addr").get_text(strip=True)
        price_text = card.select_one(".list-card-price").get_text(strip=True)

        doc = {
            "source": "zillow",
            "address": address,
            "price": price_text,
            # possibly parse beds/baths
        }
        listings_collection.insert_one(doc)

    print("Scraping complete.")
