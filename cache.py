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

    def check_semantic(self, query):
        query_embedding = np.array(get_embedding(query))

        for key, entry in self.cache.items():
            if self.is_expired(entry):
                continue

            similarity = np.dot(query_embedding, entry["embedding"]) / (
                np.linalg.norm(query_embedding) *
                np.linalg.norm(entry["embedding"])
            )

            if similarity > EMBEDDING_SIM_THRESHOLD:
                entry["last_used"] = time.time()
                self.cache.move_to_end(key)
                return entry

        return None

    def add(self, query, answer):
        normalized = self.normalize(query)
        key = self.get_hash(normalized)

        if len(self.cache) >= CACHE_MAX_SIZE:
            self.cache.popitem(last=False)

        embedding = np.array(get_embedding(query))

        self.cache[key] = {
            "query": query,
            "answer": answer,
            "embedding": embedding,
            "created_at": time.time(),
            "last_used": time.time()
        }

    def size(self):
        return len(self.cache)
