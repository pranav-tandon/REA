def get_neighborhood_stats(zip_code: str, db):
    stats_collection = db["neighborhood_stats"]
    stats_doc = stats_collection.find_one({"zip_code": zip_code})
    return stats_doc  # Could be None if not found
