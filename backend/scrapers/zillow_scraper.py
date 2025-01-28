from lxml import html
import requests
import unicodecsv as csv
import argparse

def parse(zipcode, filter=None):
    """
    Fetch property data from Zillow based on a given zipcode and filter.
    Minimally changed from the original example: only the request headers
    are updated to appear more like a real browser.
    """

    if filter == "newest":
        url = "https://www.zillow.com/homes/for_sale/{0}/0_singlestory/days_sort".format(zipcode)
    elif filter == "cheapest":
        url = "https://www.zillow.com/homes/for_sale/{0}/0_singlestory/pricea_sort/".format(zipcode)
    else:
        url = "https://www.zillow.com/homes/for_sale/{0}_rb/?fromHomePage=true&shouldFireSellPageImplicitClaimGA=false&fromHomePageTab=buy".format(zipcode)

    # We only attempt once for simplicity; you can loop multiple times if desired.
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }

    response = requests.get(url, headers=headers)
    print(response.status_code)
    if response.status_code != 200:
        print("Failed to retrieve Zillow page. Status code:", response.status_code)
        return []

    parser = html.fromstring(response.text)
    search_results = parser.xpath("//div[@id='search-results']//article")
    properties_list = []

    for prop in search_results:
        raw_address = prop.xpath(".//span[@itemprop='address']//span[@itemprop='streetAddress']//text()")
        raw_city = prop.xpath(".//span[@itemprop='address']//span[@itemprop='addressLocality']//text()")
        raw_state = prop.xpath(".//span[@itemprop='address']//span[@itemprop='addressRegion']//text()")
        raw_postal_code = prop.xpath(".//span[@itemprop='address']//span[@itemprop='postalCode']//text()")
        raw_price = prop.xpath(".//span[@class='zsg-photo-card-price']//text()")
        raw_info = prop.xpath(".//span[@class='zsg-photo-card-info']//text()")
        raw_broker_name = prop.xpath(".//span[@class='zsg-photo-card-broker-name']//text()")
        url_list = prop.xpath(".//a[contains(@class,'overlay-link')]/@href")
        raw_title = prop.xpath(".//h4//text()")

        address = ' '.join(' '.join(raw_address).split()) if raw_address else None
        city = ''.join(raw_city).strip() if raw_city else None
        state = ''.join(raw_state).strip() if raw_state else None
        postal_code = ''.join(raw_postal_code).strip() if raw_postal_code else None
        price = ''.join(raw_price).strip() if raw_price else None
        info = ' '.join(' '.join(raw_info).split()).replace(u"\xb7", ',')
        broker = ''.join(raw_broker_name).strip() if raw_broker_name else None
        title = ''.join(raw_title) if raw_title else None
        property_url = "https://www.zillow.com" + url_list[0] if url_list else None
        is_forsale = prop.xpath('.//span[@class="zsg-icon-for-sale"]')

        properties = {
            'address': address,
            'city': city,
            'state': state,
            'postal_code': postal_code,
            'price': price,
            'facts and features': info,
            'real estate provider': broker,
            'url': property_url,
            'title': title
        }
        if is_forsale:
            properties_list.append(properties)

    return properties_list

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    argparser.add_argument('zipcode', help='')
    sortorder_help = """
    available sort orders are :
    newest : Latest property details
    cheapest : Properties with cheapest price
    """
    argparser.add_argument('sort', nargs='?', help=sortorder_help, default='Homes For You')
    args = argparser.parse_args()
    zipcode = args.zipcode
    sort = args.sort

    print("Fetching data for %s" % (zipcode))
    scraped_data = parse(zipcode, sort)
    print("Writing data to output file")

    with open("properties-%s.csv" % (zipcode), 'wb') as csvfile:
        fieldnames = [
            'title', 'address', 'city', 'state', 'postal_code',
            'price', 'facts and features', 'real estate provider', 'url'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in scraped_data:
            writer.writerow(row)
