def estimate_value(property_doc, listings_collection):
    """Heuristic approach comparing average price/sqft in the same ZIP code."""
    zip_code = property_doc.get("zip_code")
    if not zip_code:
        return None

    comps = list(listings_collection.find({"zip_code": zip_code}))
    total_price = 0
    total_sqft = 0
    for comp in comps:
        price = comp.get("price")
        sqft = comp.get("sqft")
        if isinstance(price, (int, float)) and isinstance(sqft, (int, float)) and sqft > 0:
            total_price += price
            total_sqft += sqft

    if total_sqft > 0:
        avg_price_per_sqft = total_price / total_sqft
        if property_doc.get("sqft"):
            return round(avg_price_per_sqft * property_doc["sqft"], 2)

    return None
