import json
from typing import Dict, List, Any, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from utils.logger import logger
from config.settings import get_gemini_api_key

class AgentState(TypedDict):
    question: str
    current_query: str
    complexity: str
    generated_queries: List[str]
    retrieved_documents: List[Any]
    optimized_context: List[Any]
    response: str
    citations: List[Dict[str, Any]]
    confidence_score: str
    confidence_val: float
    is_hallucinated: bool
    validation_feedback: str
    chat_history: List[Dict[str, str]]
    filters: Dict[str, Any]
    retries: int

def _get_llm() -> Any:
    api_key = get_gemini_api_key()
    if not api_key:
        return None
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.2,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
    except Exception as e:
        logger.warning(f"Query Agent failed to init LLM: {e}")
        return None

def query_understanding_node(state: AgentState) -> Dict[str, Any]:
    """Analyzes the user's intent, rewrites/expands the query, and classifies complexity."""
    logger.info("Executing Query Understanding Agent...")
    question = state.get("question", "")
    chat_history = state.get("chat_history", [])
    
    # 1. Format chat history for context
    history_str = ""
    if chat_history:
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-3:]])
        
    llm = _get_llm()
    if not llm:
        logger.warning("No API key or LLM initialization failed. Using fallbacks in Query Agent.")
        return {
            "current_query": question,
            "generated_queries": [question],
            "complexity": "simple"
        }
        
    prompt = PromptTemplate.from_template(
        "You are a Query Understanding Agent.\n"
        "Given a user's question and optionally the recent chat history, perform the following tasks:\n"
        "1. Classify the question complexity: 'simple' (fact retrieval, single concept) or 'complex' (synthesizing, comparison, multi-step).\n"
        "2. Formulate a search-optimized query that resolves any pronoun references from the chat history.\n"
        "3. Generate exactly {num_queries} distinct search queries to capture different phrasing and synonyms for vector DB retrieval.\n\n"
        "Format the output strictly as a JSON object with keys:\n"
        "- 'complexity': 'simple' or 'complex'\n"
        "- 'optimized_query': the rewritten search-optimized query\n"
        "- 'expanded_queries': a list of {num_queries} strings representing search variations\n\n"
        "Chat History:\n{history}\n\n"
        "User Question: {question}\n"
    )
    
    try:
        res = llm.invoke(prompt.format(
            history=history_str if history_str else "No prior history.",
            question=question,
            num_queries=3
        ))
        
        data = json.loads(res.content)
        complexity = data.get("complexity", "simple")
        optimized_query = data.get("optimized_query", question)
        expanded_queries = data.get("expanded_queries", [question])
        
        # Ensure the original question/optimized query is in the list
        if optimized_query not in expanded_queries:
            expanded_queries.insert(0, optimized_query)
            
        logger.info(f"Query Agent output: complexity={complexity}, optimized='{optimized_query}', expanded={expanded_queries}")
        return {
            "current_query": optimized_query,
            "generated_queries": expanded_queries,
            "complexity": complexity
        }
    except Exception as e:
        logger.error(f"Error in Query Understanding node: {e}", exc_info=True)
        return {
            "current_query": question,
            "generated_queries": [question],
            "complexity": "simple"
        }
