"""
Build the search index from data/chunks.jsonl.

Embeds every chunk with a local model and stores it in a local Chroma DB.
Rebuilds from scratch on every run, so the index always matches chunks.jsonl.

Run from the server/ folder:
    uv run python -m ingest.build_index
"""

import json

import chromadb
from sentence_transformers import SentenceTransformer

import config


def main():
    with open(config.CHUNKS_FILE, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    # normalize_embeddings=True makes Chroma's default distance rank results by cosine similarity
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    try:
        client.delete_collection(config.COLLECTION)
    except Exception:
        pass  # first run: nothing to delete yet
    collection = client.create_collection(config.COLLECTION)

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=texts,
        embeddings=vectors.tolist(),
        metadatas=[c["metadata"] for c in chunks],
    )
    print(f"Indexed {collection.count()} chunks into {config.CHROMA_PATH}")


if __name__ == "__main__":
    main()