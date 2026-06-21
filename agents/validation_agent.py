import json
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from utils.logger import logger

from config.settings import get_gemini_api_key

def _get_validation_llm() -> Any:
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
        logger.warning(f"Validation Agent failed to init LLM: {e}")
        return None

def validation_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validates the generated response against the optimized context.
    
    Checks for hallucination or ungrounded claims. If errors are found, flags
    for self-correction.
    """
    logger.info("Executing Validation Agent...")
    response = state.get("response", "")
    optimized_context = state.get("optimized_context", [])
    retries = state.get("retries", 0)
    
    # 1. Bypass validation if fallback message was outputted
    fallback_msgs = [
        "the uploaded documents do not contain enough information",
        "error: gemini api key is missing"
    ]
    if any(msg in response.lower() for msg in fallback_msgs):
        logger.info("Response is fallback or error. Skipping validation.")
        return {
            "is_hallucinated": False,
            "validation_feedback": "",
            "retries": retries
        }
        
    # 2. Format context for validation
    context_str = "\n\n".join([doc.page_content for doc in optimized_context])
    
    llm = _get_validation_llm()
    if not llm:
        logger.warning("No API key or LLM init failed. Skipping validation checks.")
        return {
            "is_hallucinated": False,
            "validation_feedback": "",
            "retries": retries
        }
        
    prompt = PromptTemplate.from_template(
        "You are a Validation Agent for a RAG system.\n"
        "Your task is to verify if the generated answer is strictly grounded in the retrieved context.\n"
        "Inspect every claim and fact in the answer. If a claim is not directly stated or logically implied by the context, flag it as hallucinated.\n\n"
        "Format the output strictly as a JSON object with keys:\n"
        "- 'is_hallucinated': boolean (true if any claim is ungrounded/hallucinated, false if all is grounded)\n"
        "- 'feedback': string (detailed explanation of which specific claims are ungrounded, or empty string if none)\n\n"
        "Retrieved Context:\n{context}\n\n"
        "Generated Answer:\n{answer}\n"
    )
    
    try:
        res = llm.invoke(prompt.format(
            context=context_str,
            answer=response
        ))
        
        data = json.loads(res.content)
        is_hallucinated = data.get("is_hallucinated", False)
        feedback = data.get("feedback", "")
        
        # Limit retries to prevent infinite loops
        if is_hallucinated:
            if retries >= 2:
                logger.warning(f"Validation Agent: Hallucination detected, but retry limit reached ({retries}). Accepting current answer.")
                return {
                    "is_hallucinated": False,
                    "validation_feedback": "",
                    "retries": retries
                }
            logger.warning(f"Validation Agent: Hallucination detected. Feedback: '{feedback}'. Incrementing retries.")
            return {
                "is_hallucinated": True,
                "validation_feedback": feedback,
                "retries": retries + 1
            }
        else:
            logger.info("Validation Agent: Answer is fully grounded and validated.")
            return {
                "is_hallucinated": False,
                "validation_feedback": "",
                "retries": retries
            }
    except Exception as e:
        logger.error(f"Error in Validation Agent node: {e}", exc_info=True)
        return {
            "is_hallucinated": False,
            "validation_feedback": "",
            "retries": retries
        }
