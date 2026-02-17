from config import MODEL_COST_PER_MILLION, AVG_TOKENS_PER_REQUEST

class Analytics:
    def __init__(self):
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def record_hit(self):
        self.total_requests += 1
        self.cache_hits += 1

    def record_miss(self):
        self.total_requests += 1
        self.cache_misses += 1

    def get_stats(self, cache_size):
        hit_rate = (
            self.cache_hits / self.total_requests
            if self.total_requests > 0 else 0
        )

        cost_savings = (
            self.cache_hits *
            AVG_TOKENS_PER_REQUEST *
            MODEL_COST_PER_MILLION
        ) / 1_000_000

        return {
            "hitRate": round(hit_rate, 2),
            "totalRequests": self.total_requests,
            "cacheHits": self.cache_hits,
            "cacheMisses": self.cache_misses,
            "cacheSize": cache_size,
            "costSavings": round(cost_savings, 2),
            "savingsPercent": round(hit_rate * 100, 2),
            "strategies": [
                "exact match",
                "semantic similarity",
                "LRU eviction",
                "TTL expiration"
            ]
        }
