"""Settings shared by ingestion and retrieval.

Change them here only, so the index and the bot always use
the same embedding model and paths.
"""

CHUNKS_FILE = "data/chunks.jsonl"
CHROMA_PATH = "data/chroma"
COLLECTION = "hr_procedures"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"