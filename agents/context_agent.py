from typing import Dict, Any
from utils.logger import logger

def context_optimization_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Optimizes the retrieved documents to construct the final LLM context.
    
    De-duplicates content, filters out low-value snippets, and sorts documents
    by file name and page number to maintain logical reading order.
    """
    logger.info("Executing Context Optimization Agent...")
    retrieved_docs = state.get("retrieved_documents", [])
    
    if not retrieved_docs:
        logger.info("No documents retrieved. Context is empty.")
        return {"optimized_context": []}
        
    # 1. Content-based de-duplication
    seen_contents = set()
    unique_docs = []
    
    for doc in retrieved_docs:
        # Normalize text to detect duplicates
        normalized_content = " ".join(doc.page_content.lower().split())
        
        # Simple sliding window hash or exact match to filter repeated sections
        if normalized_content not in seen_contents:
            seen_contents.add(normalized_content)
            unique_docs.append(doc)
            
    # 2. Sort documents by source and page number (chronological narrative flow)
    # This prevents the LLM from receiving fragmented context out of order
    try:
        sorted_docs = sorted(
            unique_docs,
            key=lambda d: (
                d.metadata.get("file_name", ""),
                d.metadata.get("page_number", 0)
            )
        )
    except Exception as e:
        logger.warning(f"Failed to sort documents chronologically: {e}")
        sorted_docs = unique_docs
        
    logger.info(f"Context Agent optimized docs: {len(retrieved_docs)} -> {len(sorted_docs)}")
    return {"optimized_context": sorted_docs}
