import requests
from bs4 import BeautifulSoup
import pymongo

def redfin_scraper(search_url: str, mongo_uri: str="mongodb://localhost:27017/"):
    client = pymongo.MongoClient(mongo_uri)
    db = client["real_estate_db"]
    listings_collection = db["listings"]

    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    listing_cards = soup.select(".HomeCardContainer")
    for card in listing_cards:
        address = card.select_one(".homeAddressV2").get_text(strip=True)
        price_text = card.select_one(".homecardV2Price").get_text(strip=True)

        doc = {
            "source": "redfin",
            "address": address,
            "price": price_text,
            # possibly parse beds/baths
        }
        listings_collection.insert_one(doc)

    print("Scraping complete.")