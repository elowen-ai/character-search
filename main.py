import os
import time
import faiss
import numpy as np
from dotenv import load_dotenv
from typing import List, Tuple
from functools import lru_cache
from rapidfuzz import fuzz, process
from cassandra.cluster import Cluster

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
    def normalize_name(name: str) -> str:
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
        if not self.session: self.connect_db()

        try:
            rows = self.session.execute("SELECT name FROM characters;")
            self.character_names = [row.name for row in rows]

            if not self.character_names:
                raise ValueError("No character names found in the database")

            # Normalize names and create lookup dictionary
            self.name_lookup = {}
            for name in self.character_names:
                for variant in self.normalize_name(name):
                    self.name_lookup[variant] = name

            # Build FAISS Index
            unique_names = list(set(self.name_lookup.values()))  # Remove duplicates
            vectors = np.array([self.text_to_vector(name) for name in unique_names])
            self.index = faiss.IndexFlatL2(512)
            self.index.add(vectors)

            # Precompute min and max FAISS distances for normalization
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
        return max(0, 100 - ((distance - self.min_faiss_distance) / (self.max_faiss_distance - self.min_faiss_distance) * 100))

    @lru_cache(maxsize=1000)
    def search(self, query: str, top_k: int = 3, alpha: float = 0.7) -> List[Tuple[str, float]]:
        """
        Searches for the closest character names using:
        - Exact name match (priority)
        - Fuzzy matching
        - FAISS vector search
        - Substring matching (lower priority)
        """
        if not self.index or not self.character_names:
            raise RuntimeError("Search index not built. Call build_index() first.")

        query = query.strip()
        results = []

        # Generate normalized versions of the query
        query_variants = self.normalize_name(query)

        # 1. **Exact Match Check** (Highest Priority)
        for variant in query_variants:
            if variant in self.name_lookup:
                return [(self.name_lookup[variant], 100.0)]

        # 2. **Fuzzy Matching**
        fuzzy_matches = process.extract(
            query, self.character_names, scorer=fuzz.ratio, limit=top_k
        )
        fuzzy_results = {match[0]: match[1] for match in fuzzy_matches}

        # 3. **FAISS Search**
        query_vector = np.array([self.text_to_vector(query_variants[0])])
        distances, indices = self.index.search(query_vector, top_k)

        faiss_results = {}
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.character_names):
                name = self.character_names[idx]
                faiss_score = self.normalize_faiss_distance(dist)
                faiss_results[name] = faiss_score

        # 4. **Substring Match (Lower Priority)**
        substring_results = {
            name: 80 for name in self.character_names if query.lower() in name.lower()
        }

        # Merge Results
        all_names = set(fuzzy_results.keys()).union(
            set(faiss_results.keys()), set(substring_results.keys())
        )
        for name in all_names:
            fuzzy_score = fuzzy_results.get(name, 0)
            faiss_score = faiss_results.get(name, 0)
            substring_score = substring_results.get(name, 0)

            # Combine scores with priority: Exact > Fuzzy > FAISS > Substring
            final_score = (
                max(fuzzy_score, substring_score) * alpha + faiss_score * (1 - alpha)
            )
            results.append((name, final_score))

        # Sort by confidence
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

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
        start_time = time.time()
        matches = searcher.search(query)
        search_time = time.time() - start_time
        print(f"\nResults (found in {search_time:.3f} seconds):")
        for name, confidence in matches:
            print(f"- {name} (confidence: {confidence:.1f}%)")

if __name__ == "__main__":
    main()
