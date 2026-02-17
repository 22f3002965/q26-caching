import time
from fastapi import FastAPI
from pydantic import BaseModel
from cache import IntelligentCache
from analytics import Analytics

app = FastAPI()

cache = IntelligentCache()
analytics = Analytics()

class QueryRequest(BaseModel):
    query: str
    application: str

@app.post("/")
def handle_query(request: QueryRequest):

    start = time.time()

    # 1️⃣ Exact match
    entry = cache.check_exact(request.query)
    if entry:
        analytics.record_hit()
        latency = int((time.time() - start) * 1000)
        return {
            "answer": entry["answer"],
            "cached": True,
            "latency": latency,
            "cacheKey": "exact"
        }

    # 2️⃣ Semantic match
    entry = cache.check_semantic(request.query)
    if entry:
        analytics.record_hit()
        latency = int((time.time() - start) * 1000)
        return {
            "answer": entry["answer"],
            "cached": True,
            "latency": latency,
            "cacheKey": "semantic"
        }

    # 3️⃣ Miss → simulate LLM call
    analytics.record_miss()

    answer = f"Review result for: {request.query}"

    cache.add(request.query, answer)

    latency = int((time.time() - start) * 1000)

    return {
        "answer": answer,
        "cached": False,
        "latency": latency,
        "cacheKey": "miss"
    }

@app.get("/analytics")
def get_analytics():
    return analytics.get_stats(cache.size())
