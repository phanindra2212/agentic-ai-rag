import time
from datetime import datetime
from typing import Dict, Any, List
from utils.logger import logger

# Global fallback for non-streamlit contexts (like unit testing)
_fallback_metrics = None

def _get_session_metrics_store() -> Dict[str, Any]:
    """Gets the session metrics store from Streamlit session state, or a fallback for unit tests."""
    try:
        import streamlit as st
        if st.runtime.exists():
            if "session_metrics" not in st.session_state:
                st.session_state.session_metrics = {
                    "total_documents": 0,
                    "total_chunks": 0,
                    "total_queries": 0,
                    "retrieval_times": [],
                    "response_times": [],
                    "search_history": [],
                    "token_usage": {"input_tokens": 0, "output_tokens": 0},
                    "estimated_cost": 0.0,
                    "confidence_scores": [],
                    "citation_coverage": []
                }
            return st.session_state.session_metrics
    except Exception:
        pass
        
    # Fallback for testing environments
    global _fallback_metrics
    if _fallback_metrics is None:
        _fallback_metrics = {
            "total_documents": 0,
            "total_chunks": 0,
            "total_queries": 0,
            "retrieval_times": [],
            "response_times": [],
            "search_history": [],
            "token_usage": {"input_tokens": 0, "output_tokens": 0},
            "estimated_cost": 0.0,
            "confidence_scores": [],
            "citation_coverage": []
        }
    return _fallback_metrics

def update_document_stats(doc_count: int, chunk_count: int) -> None:
    """Updates the count of documents and chunks for the current session."""
    store = _get_session_metrics_store()
    store["total_documents"] = doc_count
    store["total_chunks"] = chunk_count
    logger.info(f"Updated session document stats: docs={doc_count}, chunks={chunk_count}")

def record_query(
    question: str,
    answer: str,
    retrieval_time: float,
    response_time: float,
    evaluation: Dict[str, float] = None,
    confidence_val: float = 0.0,
    citations: List[Dict[str, Any]] = None,
    context_chunks: List[str] = None
) -> None:
    """Records a query transaction for the active session, calculating cost and coverage."""
    store = _get_session_metrics_store()
    store["total_queries"] += 1
    store["retrieval_times"].append(retrieval_time)
    store["response_times"].append(response_time)
    
    # Cap time arrays to prevent unbounded growth
    if len(store["retrieval_times"]) > 100:
        store["retrieval_times"] = store["retrieval_times"][-100:]
    if len(store["response_times"]) > 100:
        store["response_times"] = store["response_times"][-100:]
        
    # 1. Estimate Token Usage
    # Inputs: Question, Context text chunks, chat history
    ctx_words = sum(len(c.split()) for c in (context_chunks or []))
    ans_words = len(answer.split())
    q_words = len(question.split())
    
    # Estimate context history word count from past session searches
    hist_words = 0
    for entry in store["search_history"][-5:]:
        hist_words += len(entry["question"].split()) + len(entry["answer"].split())
        
    # Base multiplier for multiple LLM runs: 
    # Context is processed in: Response Agent, Validation Agent, and 3 Evaluation prompts (Precision, Recall, Faithfulness)
    # Hence context is passed ~5 times. Prompts have overhead instructions (~1200 words).
    est_input_tokens = int((q_words * 7 + ctx_words * 5 + hist_words * 2 + ans_words * 3) * 1.33) + 1200
    est_output_tokens = int((ans_words + 400) * 1.33)  # Query agent options + Generator answer + Validation reasons + Evaluator JSONs
    
    store["token_usage"]["input_tokens"] += est_input_tokens
    store["token_usage"]["output_tokens"] += est_output_tokens
    
    # 2. Estimate Gemini Cost (gemini-2.5-flash: $0.075/1M input, $0.30/1M output)
    query_cost = (est_input_tokens * 0.075 / 1000000.0) + (est_output_tokens * 0.30 / 1000000.0)
    store["estimated_cost"] += query_cost
    
    # 3. Citation Coverage
    # Coverage is: unique sources cited / unique sources retrieved
    citation_coverage = 0.0
    if context_chunks and citations:
        citation_coverage = min(1.0, len(citations) / len(context_chunks))
    store["citation_coverage"].append(citation_coverage)
    if len(store["citation_coverage"]) > 100:
        store["citation_coverage"] = store["citation_coverage"][-100:]
        
    # 4. Confidence Score list
    store["confidence_scores"].append(confidence_val)
    if len(store["confidence_scores"]) > 100:
        store["confidence_scores"] = store["confidence_scores"][-100:]
        
    # 5. History entry
    history_entry = {
        "question": question,
        "answer": answer,
        "timestamp": datetime.now().isoformat(),
        "retrieval_time": retrieval_time,
        "response_time": response_time,
        "evaluation": evaluation or {},
        "confidence_val": confidence_val,
        "citation_coverage": citation_coverage,
        "input_tokens": est_input_tokens,
        "output_tokens": est_output_tokens,
        "cost": query_cost
    }
    store["search_history"].append(history_entry)
    logger.info(f"Recorded session query. Cost: ${query_cost:.6f} | Coverage: {citation_coverage*100:.1f}%")

def get_analytics() -> Dict[str, Any]:
    """Computes and returns aggregate analytics metrics for the current session."""
    store = _get_session_metrics_store()
    ret_times = store.get("retrieval_times", [])
    resp_times = store.get("response_times", [])
    history = store.get("search_history", [])
    
    avg_ret_time = sum(ret_times) / len(ret_times) if ret_times else 0.0
    avg_resp_time = sum(resp_times) / len(resp_times) if resp_times else 0.0
    
    # Calculate average evaluations
    eval_precision = []
    eval_recall = []
    eval_faithfulness = []
    eval_relevance = []
    
    for entry in history:
        evals = entry.get("evaluation", {})
        if "context_precision" in evals:
            eval_precision.append(evals["context_precision"])
        if "context_recall" in evals:
            eval_recall.append(evals["context_recall"])
        if "faithfulness" in evals:
            eval_faithfulness.append(evals["faithfulness"])
        if "answer_relevance" in evals:
            eval_relevance.append(evals["answer_relevance"])
            
    avg_precision = sum(eval_precision) / len(eval_precision) if eval_precision else 0.0
    avg_recall = sum(eval_recall) / len(eval_recall) if eval_recall else 0.0
    avg_faithfulness = sum(eval_faithfulness) / len(eval_faithfulness) if eval_faithfulness else 0.0
    avg_relevance = sum(eval_relevance) / len(eval_relevance) if eval_relevance else 0.0
    
    # Calculate average citation coverage
    citation_coverages = store.get("citation_coverage", [])
    avg_citation_coverage = sum(citation_coverages) / len(citation_coverages) if citation_coverages else 0.0
    
    # Calculate session duration
    duration = 0.0
    try:
        import streamlit as st
        if st.runtime.exists() and "session_start_time" in st.session_state:
            duration = time.time() - st.session_state.session_start_time
    except Exception:
        pass
        
    # Get collection stats (documents and chunks in this session's database)
    doc_count = 0
    chunk_count = 0
    try:
        from rag.ingest import get_collection_statistics
        stats = get_collection_statistics()
        doc_count = stats.get("total_documents", 0)
        chunk_count = stats.get("total_chunks", 0)
    except Exception:
        doc_count = store.get("total_documents", 0)
        chunk_count = store.get("total_chunks", 0)
        
    return {
        "total_documents": doc_count,
        "total_chunks": chunk_count,
        "total_queries": store.get("total_queries", 0),
        "average_retrieval_time": avg_ret_time,
        "average_response_time": avg_resp_time,
        "average_context_precision": avg_precision,
        "average_context_recall": avg_recall,
        "average_faithfulness": avg_faithfulness,
        "average_answer_relevance": avg_relevance,
        "search_history": history,
        "session_duration": duration,
        "token_usage": store.get("token_usage", {"input_tokens": 0, "output_tokens": 0}),
        "estimated_cost": store.get("estimated_cost", 0.0),
        "confidence_scores": store.get("confidence_scores", []),
        "average_citation_coverage": avg_citation_coverage,
        "citation_coverages": citation_coverages
    }

def clear_all_metrics() -> None:
    """Resets the metrics storage for the current session."""
    store = _get_session_metrics_store()
    store["total_documents"] = 0
    store["total_chunks"] = 0
    store["total_queries"] = 0
    store["retrieval_times"] = []
    store["response_times"] = []
    store["search_history"] = []
    store["token_usage"] = {"input_tokens": 0, "output_tokens": 0}
    store["estimated_cost"] = 0.0
    store["confidence_scores"] = []
    store["citation_coverage"] = []
    logger.info("Cleared all session metrics and query history.")
