"""
Simple RAG utilities backed by Chroma for SportsMind.

Provides collection management per matchup, simple text chunking, ingestion
(avoiding duplicates), and retrieval of nearest-document contexts.

Public API:
- get_collection(home_team: str = "", away_team: str = "") -> chromadb.Collection
- chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]
- ingest(text: str, source: str, home_team: str = "", away_team: str = "") -> None
- retrieve(query: str, n_results: int = 4, home_team: str = "", away_team: str = "") -> str

Returns:
- ingest: None (side-effect: writes to Chroma persistent store)
- retrieve: concatenated context string ("" if none available)

Raises / Errors:
- chromadb client/collection operations may raise on I/O or configuration errors.
- Caller should ensure DB_DIR is writable and embedding model is available.

Notes:
- Each matchup uses a hashed collection name to avoid context bleed.
- Chunking is word-based and returns overlapping segments for better recall.

Example:
>>> ingest("sample text", source="news", home_team="LAL", away_team="GSW")
>>> ctx = retrieve("Lakers playoff history", home_team="LAL", away_team="GSW")
"""

import hashlib
import json
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DB_DIR = ".cache/chroma"

embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path=DB_DIR)

def get_collection(home_team: str = "", away_team: str = ""):
    """Each matchup gets its own collection to prevent context bleed."""
    if home_team and away_team:
        key = f"{home_team}_{away_team}".lower().replace(" ", "_")
        name = f"sm_{hashlib.md5(key.encode()).hexdigest()[:12]}"
    else:
        name = "sportsmind_historical"
    return client.get_or_create_collection(
        name=name,
        embedding_function=embedding_fn
    )

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def ingest(text: str, source: str, home_team: str = "", away_team: str = "") -> None:
    collection = get_collection(home_team, away_team)
    chunks = chunk_text(text)
    ids, docs, metas = [], [], []
    for i, chunk in enumerate(chunks):
        doc_id = hashlib.md5(f"{source}_{i}_{chunk[:50]}".encode()).hexdigest()
        existing = collection.get(ids=[doc_id])
        if existing["ids"]:
            continue
        ids.append(doc_id)
        docs.append(chunk)
        metas.append({"source": source})
    if ids:
        collection.add(ids=ids, documents=docs, metadatas=metas)

def retrieve(query: str, n_results: int = 4, home_team: str = "", away_team: str = "") -> str:
    collection = get_collection(home_team, away_team)
    count = collection.count()
    if count == 0:
        return ""
    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, count)
    )
    docs = results["documents"][0] if results["documents"] else []
    return "\n\n".join(docs) if docs else ""