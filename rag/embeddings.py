import os
from typing import Any
from langchain_core.embeddings import Embeddings
from config.settings import EMBEDDING_MODEL_NAME, FALLBACK_EMBEDDING_MODEL_NAME
from utils.logger import logger

def get_embeddings_model() -> Embeddings:
    """Returns the primary embeddings model (Google Generative AI Embeddings)
    or falls back to HuggingFaceEmbeddings if the API key is not configured.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    
    if api_key:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        # Try primary model: text-embedding-004
        try:
            logger.info(f"Initializing Google Embeddings (Model: {EMBEDDING_MODEL_NAME})...")
            embeddings = GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL_NAME,
                google_api_key=api_key
            )
            # Test embedding a tiny text to verify key works
            embeddings.embed_query("test")
            logger.info("Google Embeddings (text-embedding-004) initialized successfully.")
            return embeddings
        except Exception as e:
            logger.warning(
                f"Google Embeddings initialization failed with {EMBEDDING_MODEL_NAME}: {e}. "
                "Attempting fallback to models/embedding-001..."
            )
            
        # Try secondary model: gemini-embedding-001
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=api_key
            )
            # Test embedding
            embeddings.embed_query("test")
            logger.info("Google Embeddings (gemini-embedding-001) initialized successfully.")
            return embeddings
        except Exception as e:
            logger.warning(
                f"Google Embeddings initialization failed with models/gemini-embedding-001: {e}. "
                "Attempting fallback to Sentence Transformers..."
            )
            
    # Fallback embeddings
    try:
        logger.info(f"Initializing Fallback Embeddings: {FALLBACK_EMBEDDING_MODEL_NAME}...")
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name=FALLBACK_EMBEDDING_MODEL_NAME)
        logger.info("Fallback Embeddings (Sentence Transformers) initialized successfully.")
        return embeddings
    except Exception as e:
        logger.critical(f"Failed to initialize any embeddings model: {e}", exc_info=True)
        raise RuntimeError("No embeddings model could be initialized.")
