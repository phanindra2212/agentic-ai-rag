from typing import Dict, Any
from rag.retriever import search_documents
from utils.logger import logger

def retrieval_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieves documents from Chroma DB using the generated expanded queries.
    
    Merges results, de-duplicates chunks, and calculates retrieval confidence.
    """
    logger.info("Executing Retrieval Agent...")
    generated_queries = state.get("generated_queries", [])
    filters = state.get("filters", {})
    
    # Extract filter criteria
    file_names = filters.get("file_names", [])
    file_types = filters.get("file_types", [])
    top_k = filters.get("top_k", 5)
    
    if not generated_queries:
        # Fallback to current query if generated queries is empty
        current_q = state.get("current_query", state.get("question", ""))
        generated_queries = [current_q]
        
    all_results = {}
    
    # 1. Search for each query expansion
    for q in generated_queries:
        scored_docs = search_documents(
            query=q,
            top_k=top_k,
            file_names=file_names,
            file_types=file_types
        )
        
        # Merge and keep the highest similarity score for duplicates
        for doc, similarity in scored_docs:
            chunk_id = doc.metadata.get("chunk_id")
            if not chunk_id:
                chunk_id = doc.page_content # Fallback if chunk_id is missing
                
            if chunk_id not in all_results or similarity > all_results[chunk_id][1]:
                all_results[chunk_id] = (doc, similarity)
                
    # 2. Sort merged results by similarity score descending
    sorted_results = sorted(all_results.values(), key=lambda x: x[1], reverse=True)
    
    # 3. Take Top-K of the merged results
    final_results = sorted_results[:top_k]
    
    # Extract documents and scores
    retrieved_docs = [item[0] for item in final_results]
    scores = [item[1] for item in final_results]
    
    # 4. Calculate confidence score
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    if avg_score >= 0.70:
        confidence = "High Confidence"
    elif avg_score >= 0.45:
        confidence = "Medium Confidence"
    else:
        confidence = "Low Confidence"
        
    logger.info(
        f"Retrieval Agent complete. Merged {len(sorted_results)} chunks to Top-{len(retrieved_docs)}. "
        f"Average similarity score: {avg_score:.4f} ({confidence})"
    )
    
    return {
        "retrieved_documents": retrieved_docs,
        "confidence_score": confidence,
        "confidence_val": avg_score
    }
