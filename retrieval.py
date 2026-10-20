"""
search(query): find the HR procedure chunks that best answer a question.

This is the only thing the bot calls. Ingestion builds the index; this file
only reads it. Moving to Azure AI Search later means rewriting this file only.
"""

from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

import config


# Loading the model and opening the DB takes seconds, so do it once and reuse.
# This matters for a voice bot: every question would otherwise pay that delay.
@lru_cache(maxsize=1)
def _model():
    return SentenceTransformer(config.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    return client.get_collection(config.COLLECTION)


def search(query, k=3):
    """Return the k closest chunks, best first.

    Each result: {"text", "metadata", "distance"}. Lower distance = closer meaning.
    """
    vector = _model().encode([query], normalize_embeddings=True)
    res = _collection().query(query_embeddings=vector.tolist(), n_results=k)
    return [
        {"text": text, "metadata": meta, "distance": round(dist, 3)}
        for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]