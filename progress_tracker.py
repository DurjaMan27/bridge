import time

_bid_counter = {"count": 0, "start_time": time.time()}

def increment_bid_count():
    _bid_counter["count"] += 1

def get_bid_count():
    return _bid_counter["count"]

def reset_counter():
    _bid_counter["count"] = 0
    _bid_counter["start_time"] = time.time()