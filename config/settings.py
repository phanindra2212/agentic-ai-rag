import os
from pathlib import Path
from dotenv import load_dotenv

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)
DATA_DIR = BASE_DIR / "data"
# Base Temp Directory for user sessions
TEMP_DIR = BASE_DIR / "temp"

# Default paths for backward compatibility and tests
UPLOAD_DIR = TEMP_DIR / "default" / "uploads"
CHROMA_DIR = TEMP_DIR / "default" / "chroma"

# Create default directories if they do not exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

def get_session_id() -> str:
    """Gets the active session ID from Streamlit session state, or returns 'default'."""
    try:
        import streamlit as st
        if st.runtime.exists() and "session_id" in st.session_state:
            return st.session_state.session_id
    except Exception:
        pass
    return "default"

def get_upload_dir(session_id: str = None) -> Path:
    """Returns the isolated upload directory for a session."""
    if session_id is None:
        session_id = get_session_id()
    upload_dir = TEMP_DIR / session_id / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir

def get_chroma_dir(session_id: str = None) -> Path:
    """Returns the isolated Chroma database directory for a session."""
    if session_id is None:
        session_id = get_session_id()
    chroma_dir = TEMP_DIR / session_id / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chroma_dir

def cleanup_orphaned_sessions():
    """Scans base temp directory and deletes folders of inactive session IDs."""
    import time
    # Run at most once every 5 minutes per process to avoid excessive disk I/O
    now = time.time()
    last_cleanup = getattr(cleanup_orphaned_sessions, "_last_run", 0.0)
    if now - last_cleanup < 300:
        return
    cleanup_orphaned_sessions._last_run = now
    
    try:
        from streamlit.runtime import get_instance
        runtime = get_instance()
        if not runtime:
            return
            
        active_ids = {s.id for s in runtime._session_info_by_id.values()}
        active_ids.add("default")
        
        if TEMP_DIR.exists():
            import shutil
            for folder in TEMP_DIR.iterdir():
                if folder.is_dir() and folder.name not in active_ids:
                    try:
                        from utils.logger import logger
                        logger.info(f"Cleaning up orphaned session directory: {folder}")
                    except Exception:
                        pass
                    shutil.rmtree(folder, ignore_errors=True)
    except Exception as e:
        try:
            from utils.logger import logger
            logger.warning(f"Error in cleanup_orphaned_sessions: {e}")
        except Exception:
            pass

def clear_current_session(session_id: str):
    """Deletes uploaded files and Chroma database for the specified session ID."""
    import shutil
    try:
        from utils.logger import logger
        logger.info(f"Clearing session directories for session: {session_id}")
    except Exception:
        pass
    
    upload_dir = get_upload_dir(session_id)
    if upload_dir.exists():
        shutil.rmtree(upload_dir, ignore_errors=True)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
    chroma_dir = get_chroma_dir(session_id)
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir, ignore_errors=True)
        chroma_dir.mkdir(parents=True, exist_ok=True)

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
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
FALLBACK_EMBEDDING_MODEL_NAME = "models/gemini-embedding-2"

# Retrieval Settings
DEFAULT_TOP_K = 5
MAX_TOP_K = 15

# Query Expansion Setting
MULTI_QUERY_COUNT = 3

# Logging
LOG_FILE_PATH = BASE_DIR / "app.log"

def get_gemini_api_key() -> str:
    """Returns the custom API key from Streamlit session state if available,
    otherwise falls back to the system environment variable.
    """
    try:
        import streamlit as st
        # Only return custom key if it's set and not empty
        if st.runtime.exists() and st.session_state.get("custom_gemini_api_key"):
            return st.session_state["custom_gemini_api_key"].strip()
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "")
