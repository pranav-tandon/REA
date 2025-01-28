# backend/scrapers/scrape_all.py
from zillow_scraper import scrape_zillow
from redfin_scraper import scrape_redfin

def scrape_all():
    # Provide your target URLs
    zillow_url = "https://www.zillow.com/homes/for_sale/"
    scrape_zillow(zillow_url)

    redfin_url = "https://www.redfin.com/city/XXXXX"  # example
    scrape_redfin(redfin_url)

if __name__ == "__main__":
    scrape_all()
