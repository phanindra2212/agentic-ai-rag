import os
from langchain_core.embeddings import Embeddings
from config.settings import EMBEDDING_MODEL_NAME, FALLBACK_EMBEDDING_MODEL_NAME
from utils.logger import logger

def get_embeddings_model() -> Embeddings:
    """Returns the primary embeddings model (Hugging Face Inference API using BAAI/bge-m3)
    and handles fallbacks to local Sentence Transformers or Google Embeddings.
    """
    # 1. Try Hugging Face Inference API (unlimited cloud calculation, zero CPU/GPU load)
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if token:
        try:
            logger.info(f"Initializing Hugging Face Inference API (Model: {EMBEDDING_MODEL_NAME})...")
            from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
            embeddings = HuggingFaceInferenceAPIEmbeddings(
                api_key=token,
                model_name=EMBEDDING_MODEL_NAME
            )
            # Test embedding connectivity
            try:
                embeddings.embed_query("test")
                logger.info("Hugging Face Inference API Embeddings initialized successfully.")
                return embeddings
            except Exception as internet_err:
                logger.warning(f"Hugging Face Inference API connection test failed: {internet_err}. Triggering fallbacks...")
                raise internet_err
        except Exception as e:
            logger.warning(f"Hugging Face Inference API failed to initialize: {e}. Trying fallbacks...")
            
    # 2. Try local HuggingFace Embeddings second (offline fallback)
    try:
        logger.info("Initializing local Hugging Face Embeddings (Model: sentence-transformers/all-MiniLM-L6-v2)...")
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        embeddings.embed_query("test")
        logger.info("Local Hugging Face Embeddings initialized successfully.")
        return embeddings
    except Exception as e:
        logger.warning(f"Local Hugging Face Embeddings failed to initialize: {e}. Trying Google fallback...")
        
    # 3. Fallback to Google Generative AI Embeddings
    from config.settings import get_gemini_api_key
    api_key = get_gemini_api_key()
    if api_key:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        try:
            logger.info(f"Initializing Google Embeddings (Model: {FALLBACK_EMBEDDING_MODEL_NAME})...")
            embeddings = GoogleGenerativeAIEmbeddings(
                model=FALLBACK_EMBEDDING_MODEL_NAME,
                google_api_key=api_key
            )
            embeddings.embed_query("test")
            logger.info("Google Embeddings fallback initialized successfully.")
            return embeddings
        except Exception as e:
            logger.warning(f"Google Embeddings fallback failed: {e}")
            
    # 4. Final Fail-Safe: Fake/Mock Embeddings if nothing else is available
    try:
        logger.warning("No real embeddings model could be initialized. Setting up FakeEmbeddings as a last-resort fail-safe.")
        from langchain_core.embeddings import FakeEmbeddings
        # Size 1024 matches BAAI/bge-m3 vector dimension
        return FakeEmbeddings(size=1024)
    except Exception as e:
        logger.critical(f"Failed to initialize any embeddings model: {e}", exc_info=True)
        raise RuntimeError("No embeddings model could be initialized.")
