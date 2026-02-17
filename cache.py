import time
import hashlib
import numpy as np
from collections import OrderedDict
from config import CACHE_MAX_SIZE, CACHE_TTL_SECONDS, EMBEDDING_SIM_THRESHOLD
from embeddings import get_embedding


class IntelligentCache:
    def __init__(self):
        self.cache = OrderedDict()

    def normalize(self, query):
        return query.strip().lower()

    def get_hash(self, text):
        return hashlib.md5(text.encode()).hexdigest()

    def is_expired(self, entry):
        return time.time() - entry["created_at"] > CACHE_TTL_SECONDS

    # ---------------- EXACT MATCH ----------------
    def check_exact(self, query):
        normalized = self.normalize(query)
        key = self.get_hash(normalized)

        if key in self.cache:
            entry = self.cache[key]

            if self.is_expired(entry):
                del self.cache[key]
                return None

            entry["last_used"] = time.time()
            self.cache.move_to_end(key)
            return entry

        return None

    # ---------------- SEMANTIC MATCH ----------------
    def check_semantic(self, query):
        try:
            query_embedding = np.array(get_embedding(query))
        except Exception:
            # If embedding fails, skip semantic matching
            return None

        for key, entry in list(self.cache.items()):
            if self.is_expired(entry):
                del self.cache[key]
                continue

            try:
                similarity = np.dot(query_embedding, entry["embedding"]) / (
                    np.linalg.norm(query_embedding) *
                    np.linalg.norm(entry["embedding"])
                )
            except Exception:
                continue

            if similarity > EMBEDDING_SIM_THRESHOLD:
                entry["last_used"] = time.time()
                self.cache.move_to_end(key)
                return entry

        return None

    # ---------------- ADD TO CACHE ----------------
    def add(self, query, answer):
        normalized = self.normalize(query)
        key = self.get_hash(normalized)

        if len(self.cache) >= CACHE_MAX_SIZE:
            self.cache.popitem(last=False)

        try:
            embedding = np.array(get_embedding(query))
        except Exception:
            # If embedding fails, use dummy vector
            embedding = np.zeros(10)

        self.cache[key] = {
            "query": query,
            "answer": answer,
            "embedding": embedding,
            "created_at": time.time(),
            "last_used": time.time()
        }

    def size(self):
        return len(self.cache)

    # ---------------- MAIN PROCESS FUNCTION ----------------
    def process_query(self, query):
        # 1️⃣ Exact match
        entry = self.check_exact(query)
        if entry:
            return {
                "answer": entry["answer"],
                "cached": True,
                "cacheKey": "exact"
            }

        # 2️⃣ Semantic match
        entry = self.check_semantic(query)
        if entry:
            return {
                "answer": entry["answer"],
                "cached": True,
                "cacheKey": "semantic"
            }

        # 3️⃣ Miss → simulate LLM response
        answer = f"Review result for: {query}"

        self.add(query, answer)

        return {
            "answer": answer,
            "cached": False,
            "cacheKey": "miss"
        }

    # ---------------- ANALYTICS PLACEHOLDER ----------------
    def get_analytics(self):
        total = len(self.cache)
        return {
            "hitRate": 0.0,
            "totalRequests": total,
            "cacheHits": 0,
            "cacheMisses": 0,
            "cacheSize": total,
            "costSavings": 0.0,
            "savingsPercent": 0.0,
            "strategies": [
                "exact match",
                "semantic similarity",
                "LRU eviction",
                "TTL expiration"
            ]
        }

