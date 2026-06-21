import json
from datetime import datetime
from typing import Dict, Any
from config.settings import DATA_DIR
from utils.logger import logger

METRICS_FILE = DATA_DIR / "metrics.json"

def _load_metrics() -> Dict[str, Any]:
    """Loads metrics from the persistent JSON file."""
    if not METRICS_FILE.exists():
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "total_queries": 0,
            "retrieval_times": [],
            "response_times": [],
            "search_history": []
        }
    try:
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading metrics file: {e}")
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "total_queries": 0,
            "retrieval_times": [],
            "response_times": [],
            "search_history": []
        }

def _save_metrics(data: Dict[str, Any]) -> None:
    """Saves metrics to the persistent JSON file."""
    try:
        with open(METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving metrics file: {e}")

def update_document_stats(doc_count: int, chunk_count: int) -> None:
    """Updates the count of documents and chunks."""
    data = _load_metrics()
    data["total_documents"] = doc_count
    data["total_chunks"] = chunk_count
    _save_metrics(data)
    logger.info(f"Updated document stats: docs={doc_count}, chunks={chunk_count}")

def record_query(
    question: str,
    answer: str,
    retrieval_time: float,
    response_time: float,
    evaluation: Dict[str, float] = None
) -> None:
    """Records a single query transaction with execution times and evaluations."""
    data = _load_metrics()
    data["total_queries"] += 1
    data["retrieval_times"].append(retrieval_time)
    data["response_times"].append(response_time)
    
    # Cap list sizes to prevent infinite growth
    if len(data["retrieval_times"]) > 100:
        data["retrieval_times"] = data["retrieval_times"][-100:]
    if len(data["response_times"]) > 100:
        data["response_times"] = data["response_times"][-100:]
        
    history_entry = {
        "question": question,
        "answer": answer,
        "timestamp": datetime.now().isoformat(),
        "retrieval_time": retrieval_time,
        "response_time": response_time,
        "evaluation": evaluation or {}
    }
    data["search_history"].append(history_entry)
    _save_metrics(data)
    logger.info(f"Recorded query transaction for: '{question}'")

def get_analytics() -> Dict[str, Any]:
    """Computes and returns aggregate analytics metrics."""
    data = _load_metrics()
    ret_times = data.get("retrieval_times", [])
    resp_times = data.get("response_times", [])
    
    avg_ret_time = sum(ret_times) / len(ret_times) if ret_times else 0.0
    avg_resp_time = sum(resp_times) / len(resp_times) if resp_times else 0.0
    
    # Calculate average evaluations if present
    eval_precision = []
    eval_recall = []
    eval_faithfulness = []
    eval_relevance = []
    
    for entry in data.get("search_history", []):
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
    
    return {
        "total_documents": data.get("total_documents", 0),
        "total_chunks": data.get("total_chunks", 0),
        "total_queries": data.get("total_queries", 0),
        "average_retrieval_time": avg_ret_time,
        "average_response_time": avg_resp_time,
        "average_context_precision": avg_precision,
        "average_context_recall": avg_recall,
        "average_faithfulness": avg_faithfulness,
        "average_answer_relevance": avg_relevance,
        "search_history": data.get("search_history", [])
    }

def clear_all_metrics() -> None:
    """Resets the metrics storage file."""
    data = {
        "total_documents": 0,
        "total_chunks": 0,
        "total_queries": 0,
        "retrieval_times": [],
        "response_times": [],
        "search_history": []
    }
    _save_metrics(data)
    logger.info("Cleared all metrics and query history.")
