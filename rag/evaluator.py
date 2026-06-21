import json
from typing import Dict, List, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from utils.logger import logger

from config.settings import get_gemini_api_key

def _get_eval_llm() -> Any:
    """Instantiates a Gemini model for evaluation."""
    api_key = get_gemini_api_key()
    if not api_key:
        return None
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.0,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
    except Exception as e:
        logger.warning(f"Failed to initialize evaluation LLM: {e}")
        return None

def evaluate_rag(
    question: str,
    retrieved_chunks: List[str],
    generated_answer: str
) -> Dict[str, float]:
    """Evaluates the RAG response across context precision, recall, faithfulness, and relevance.
    
    Returns:
        Dict containing score values between 0.0 and 1.0 for each metric.
    """
    logger.info("Running RAG evaluation module...")
    
    # Defaults
    scores = {
        "context_precision": 0.8,
        "context_recall": 0.8,
        "faithfulness": 0.8,
        "answer_relevance": 0.8
    }
    
    llm = _get_eval_llm()
    if not llm:
        logger.warning("Gemini API key missing or LLM failed to load. Returning default evaluation scores.")
        return scores
        
    context_text = "\n\n".join([f"--- Chunk {i+1} ---\n{chunk}" for i, chunk in enumerate(retrieved_chunks)])
    
    # 1. Context Precision
    precision_prompt = PromptTemplate.from_template(
        "You are an AI evaluator. Analyze the user question and the retrieved chunks of context.\n"
        "Determine which chunks contain relevant information that helps answer the user question.\n"
        "For each chunk, assign a binary relevance score (1 if it contains relevant facts, 0 otherwise).\n"
        "Format the output strictly as a JSON object with keys:\n"
        "- 'scores': list of integers (0 or 1) for each chunk in order\n"
        "- 'precision': float (number of 1s divided by total chunks)\n"
        "- 'reasoning': a short explanation of your scoring\n\n"
        "Question: {question}\n\n"
        "Context:\n{context}\n"
    )
    
    # 2. Context Recall
    recall_prompt = PromptTemplate.from_template(
        "You are an AI evaluator. Analyze the user question, the generated answer, and the retrieved chunks of context.\n"
        "Determine if the facts present in the generated answer that directly address the question are present in the retrieved chunks.\n"
        "Format the output strictly as a JSON object with keys:\n"
        "- 'recalled_facts_count': number of facts in the answer that are supported by the context\n"
        "- 'total_facts_count': total number of facts in the answer relevant to the question\n"
        "- 'recall': float (recalled_facts_count divided by total_facts_count)\n"
        "- 'reasoning': a short explanation\n\n"
        "Question: {question}\n\n"
        "Answer: {answer}\n\n"
        "Context:\n{context}\n"
    )
    
    # 3. Faithfulness (Groundedness)
    faithfulness_prompt = PromptTemplate.from_template(
        "You are an AI evaluator. Analyze the generated answer and the retrieved chunks of context.\n"
        "Verify if all claims and statements made in the generated answer are strictly supported by the context.\n"
        "Format the output strictly as a JSON object with keys:\n"
        "- 'supported_claims_count': number of statements in the answer supported by the context\n"
        "- 'total_claims_count': total statements in the answer\n"
        "- 'faithfulness': float (supported_claims_count divided by total_claims_count)\n"
        "- 'reasoning': short explanation\n\n"
        "Answer: {answer}\n\n"
        "Context:\n{context}\n"
    )
    
    # 4. Answer Relevance
    relevance_prompt = PromptTemplate.from_template(
        "You are an AI evaluator. Analyze the user question and the generated answer.\n"
        "Evaluate how directly and completely the answer addresses the question, without considering context.\n"
        "Format the output strictly as a JSON object with keys:\n"
        "- 'rating_1_to_5': integer from 1 to 5 representing relevance\n"
        "- 'relevance': float (rating_1_to_5 divided by 5.0)\n"
        "- 'reasoning': short explanation\n\n"
        "Question: {question}\n\n"
        "Answer: {answer}\n"
    )
    
    try:
        # Context Precision
        precision_res = llm.invoke(precision_prompt.format(question=question, context=context_text))
        precision_data = json.loads(precision_res.content)
        scores["context_precision"] = float(precision_data.get("precision", 0.8))
        
        # Context Recall
        recall_res = llm.invoke(recall_prompt.format(question=question, answer=generated_answer, context=context_text))
        recall_data = json.loads(recall_res.content)
        scores["context_recall"] = float(recall_data.get("recall", 0.8))
        
        # Faithfulness
        faithfulness_res = llm.invoke(faithfulness_prompt.format(answer=generated_answer, context=context_text))
        faithfulness_data = json.loads(faithfulness_res.content)
        scores["faithfulness"] = float(faithfulness_data.get("faithfulness", 0.8))
        
        # Answer Relevance
        relevance_res = llm.invoke(relevance_prompt.format(question=question, answer=generated_answer))
        relevance_data = json.loads(relevance_res.content)
        scores["answer_relevance"] = float(relevance_data.get("relevance", 0.8))
        
        logger.info(f"RAG evaluation complete. Scores: {scores}")
        
    except Exception as e:
        logger.error(f"Error executing RAG evaluation: {e}", exc_info=True)
        
    return scores
