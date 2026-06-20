import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_db"

# Create directories if they do not exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# File Settings
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt"}
MAX_FILE_COUNT = 10

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Vector DB Settings
CHROMA_COLLECTION_NAME = "rag_documents"

# Models
GEMINI_MODEL_NAME = "gemini-3.5-flash"
EMBEDDING_MODEL_NAME = "models/gemini-embedding-2"
FALLBACK_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval Settings
DEFAULT_TOP_K = 5
MAX_TOP_K = 15

# Query Expansion Setting
MULTI_QUERY_COUNT = 3

# Logging
LOG_FILE_PATH = BASE_DIR / "app.log"
