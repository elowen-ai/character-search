import os
import time
import faiss
import hashlib
import numpy as np
from dotenv import load_dotenv
from typing import List, Tuple, Dict
from functools import lru_cache
from rapidfuzz import fuzz, process
from cassandra.cluster import Cluster
from datetime import datetime, timedelta

load_dotenv()

class CharacterSearch:
    def __init__(self) -> None:
        self.host: str = os.getenv("CASSANDRA_HOST")
        self.keyspace = os.getenv("CASSANDRA_KEYSPACE")
        self.cluster = None
        self.session = None
        self.index = None
        self.character_names = []
        self.name_lookup = {}
        self.min_faiss_distance = float("inf")
        self.max_faiss_distance = 0
        self.search_cache = {}
        self.cache_ttl = 300

    def connect_db(self):
        try:
            self.cluster = Cluster([self.host])
            self.session = self.cluster.connect(self.keyspace)
        except Exception as e:
            raise ConnectionError(f"Database connection failed: {str(e)}")

    def disconnect_db(self):
        if self.cluster:
            self.cluster.shutdown()

    @staticmethod
    def normalize_name(name: str) -> List[str]:
        words = name.lower().split()
        sorted_version = " ".join(sorted(words))
        original_version = " ".join(words)
        return [original_version, sorted_version]

    @staticmethod
    def text_to_vector(text: str) -> np.ndarray:
        vector = np.zeros(512, dtype=np.float32)
        text = text.lower()
        for char in text:
            vector[ord(char) % 256] += 1
        for i in range(len(text) - 1):
            bigram = text[i:i+2]
            hash_val = hash(bigram) % 256
            vector[256 + hash_val] += 0.5
        return vector / (len(text) + 1)

    def build_index(self):
        if not self.session: 
            self.connect_db()

        try:
            rows = self.session.execute("SELECT name FROM characters;")
            self.character_names = [row.name for row in rows]

            if not self.character_names:
                raise ValueError("No character names found in the database")

            self.name_lookup = {}
            for name in self.character_names:
                for variant in self.normalize_name(name):
                    self.name_lookup[variant] = name

            unique_names = list(set(self.name_lookup.values()))
            vectors = np.array([self.text_to_vector(name) for name in unique_names])
            self.index = faiss.IndexFlatL2(512)
            self.index.add(vectors)

            distances, _ = self.index.search(vectors, 1)
            self.min_faiss_distance = np.min(distances)
            self.max_faiss_distance = np.max(distances)

        except Exception as e:
            raise RuntimeError(f"Index building failed: {str(e)}")
        finally:
            self.disconnect_db()

    def normalize_faiss_distance(self, distance: float) -> float:
        if self.max_faiss_distance == self.min_faiss_distance:
            return 0
        return max(0, 100 - ((distance - self.min_faiss_distance) / 
                             (self.max_faiss_distance - self.min_faiss_distance) * 100))

    def _generate_cache_key(self, query: str, alpha: float) -> str:
        return hashlib.md5(f"{query.lower()}:{alpha}".encode()).hexdigest()

    def _clean_expired_cache(self):
        current_time = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self.search_cache.items()
            if (current_time - timestamp).total_seconds() > self.cache_ttl
        ]
        for key in expired_keys:
            del self.search_cache[key]

    def _get_all_results(self, query: str, alpha: float = 0.7) -> List[Tuple[str, float]]:
        if not self.index or not self.character_names:
            raise RuntimeError("Search index not built. Call build_index() first.")

        query = query.strip()
        results = []
        query_variants = self.normalize_name(query)

        for variant in query_variants:
            if variant in self.name_lookup:
                return [(self.name_lookup[variant], 100.0)]

        fuzzy_matches = process.extract(
            query, self.character_names, scorer=fuzz.ratio, limit=50
        )
        fuzzy_results = {match[0]: match[1] for match in fuzzy_matches}

        query_vector = np.array([self.text_to_vector(query_variants[0])])
        distances, indices = self.index.search(query_vector, 50)

        faiss_results = {}
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.character_names):
                name = self.character_names[idx]
                faiss_score = self.normalize_faiss_distance(dist)
                faiss_results[name] = faiss_score

        substring_results = {
            name: 80 for name in self.character_names if query.lower() in name.lower()
        }

        all_names = set(fuzzy_results.keys()).union(
            set(faiss_results.keys()), set(substring_results.keys())
        )
        for name in all_names:
            fuzzy_score = fuzzy_results.get(name, 0)
            faiss_score = faiss_results.get(name, 0)
            substring_score = substring_results.get(name, 0)

            final_score = (
                max(fuzzy_score, substring_score) * alpha + faiss_score * (1 - alpha)
            )
            results.append((name, final_score))

        results.sort(key=lambda x: x[1], reverse=True)
        results = [r for r in results if r[1] >= 32]
        
        return results

    def search_paginated(
        self, 
        query: str, 
        page: int = 1, 
        page_size: int = 10, 
        alpha: float = 0.7
    ) -> Dict:
        if len(self.search_cache) > 100:
            self._clean_expired_cache()
        
        cache_key = self._generate_cache_key(query, alpha)
        
        if cache_key in self.search_cache:
            all_results, _ = self.search_cache[cache_key]
        else:
            all_results = self._get_all_results(query, alpha)
            self.search_cache[cache_key] = (all_results, datetime.now())
        
        total_results = len(all_results)
        total_pages = (total_results + page_size - 1) // page_size
        
        if page < 1:
            page = 1
        if page > total_pages and total_pages > 0:
            page = total_pages
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_results = all_results[start_idx:end_idx]
        
        return {
            "results": page_results,
            "pagination": {
                "current_page": page,
                "page_size": page_size,
                "total_results": total_results,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_previous": page > 1
            },
            "query": query
        }

    @lru_cache(maxsize=1000)
    def search(self, query: str, top_k: int = 3, alpha: float = 0.7) -> List[Tuple[str, float]]:
        all_results = self._get_all_results(query, alpha)
        return all_results[:top_k]

def main():
    searcher = CharacterSearch()
    print("Building search index...")
    start_time = time.time()
    searcher.build_index()
    print(f"Index built in {time.time() - start_time:.2f} seconds")

    while True:
        query = input("\nEnter character name (or 'q' to quit): ")
        if query.lower() == 'q':
            break
            
        page_input = input("Enter page number (default 1): ")
        page = int(page_input) if page_input.strip() else 1
        
        page_size_input = input("Enter page size (default 10): ")
        page_size = int(page_size_input) if page_size_input.strip() else 10
        
        start_time = time.time()
        result = searcher.search_paginated(query, page=page, page_size=page_size)
        search_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"Results for '{result['query']}' (found in {search_time:.3f} seconds):")
        print(f"Page {result['pagination']['current_page']} of {result['pagination']['total_pages']}")
        print(f"Total results: {result['pagination']['total_results']}")
        print(f"{'='*60}\n")
        
        for name, confidence in result['results']:
            print(f"- {name} (confidence: {confidence:.1f}%)")
        
        print(f"\n{'='*60}")
        if result['pagination']['has_previous']:
            print(f"Previous page available")
        if result['pagination']['has_next']:
            print(f"Next page available")


if __name__ == "__main__":
    main()