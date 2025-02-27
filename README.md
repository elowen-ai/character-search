# Elowen.ai Character Search Engine

This is a character search engine that uses FAISS, fuzzy matching, and substring matching to find character names from Cassandra database efficiently.

This engine is used at **elowen.ai**, **Telegram bot** and **X bot**.

## Features
- Supports **exact**, **fuzzy**, and **vector-based** search.
- Handles **name reordering** (e.g., "Tanjiro Kamado" vs. "Kamado Tanjiro").
- **Optimized** with FAISS for fast retrieval.
- Uses **Cassandra DB** as the backend storage.

## Usage
1. Build the search index:
   ```python
   searcher = CharacterSearch()
   searcher.build_index()
   ```
2. Search for a character:
   ```python
   print(searcher.search("Eren"))
   print(searcher.search("Yeager"))
   ```
3. Get the matches list:
    ```python
    [('Eren Yeager', 56.0), ('Frieren', 56.0), ('Serena Williams', 56.0)]
    [('Eren Yeager', 56.0), ('Heath Ledger', 38.888888888888886), ('Saber', 38.18181818181817)]
    ```