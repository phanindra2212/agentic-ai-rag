from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from rag.ingest import get_vector_store
from utils.logger import logger
from config.settings import validate_session_isolation

def build_chroma_filter(file_names: List[str] = None, file_types: List[str] = None) -> Dict[str, Any]:
    """Constructs a Chroma metadata filter based on file names and file types."""
    filters = []
    
    if file_names:
        # Standardize filenames
        cleaned_names = [f for f in file_names if f]
        if len(cleaned_names) == 1:
            filters.append({"file_name": cleaned_names[0]})
        elif len(cleaned_names) > 1:
            filters.append({"$or": [{"file_name": name} for name in cleaned_names]})
            
    if file_types:
        cleaned_types = [t.upper() for t in file_types if t]
        if len(cleaned_types) == 1:
            filters.append({"file_type": cleaned_types[0]})
        elif len(cleaned_types) > 1:
            filters.append({"$or": [{"file_type": t} for t in cleaned_types]})
            
    if not filters:
        return {}
    elif len(filters) == 1:
        return filters[0]
    else:
        return {"$and": filters}

def search_documents(
    query: str,
    top_k: int = 5,
    file_names: List[str] = None,
    file_types: List[str] = None
) -> List[Tuple[Document, float]]:
    """Performs similarity search on Chroma vector store with scores.
    
    Converts Chroma distance (L2) to a normalized similarity score: 1 / (1 + distance).
    
    Returns:
        List of (Document, similarity_score)
    """
    if not validate_session_isolation():
        logger.critical("Security Violation Mismatch: Session mismatch detected during document search! Blocking retrieval.")
        return []
        
    logger.info(f"Retrieving documents for query: '{query}' with top_k={top_k}")
    
    try:
        db = get_vector_store()
        chroma_filter = build_chroma_filter(file_names, file_types)
        
        # If filter is empty, pass None to similarity_search_with_score
        filter_arg = chroma_filter if chroma_filter else None
        
        # Get raw results with distances
        results = db.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=filter_arg
        )
        
        # Convert distances to similarity scores (0.0 to 1.0)
        scored_docs = []
        for doc, distance in results:
            # L2 distance conversion: 1 / (1 + distance)
            # Ensure distance is positive to avoid division by zero or negative score
            safe_dist = max(0.0, float(distance))
            similarity = 1.0 / (1.0 + safe_dist)
            scored_docs.append((doc, similarity))
            
        logger.info(f"Retrieved {len(scored_docs)} documents.")
        return scored_docs
        
    except Exception as e:
        logger.error(f"Error during document search: {e}", exc_info=True)
        return []
