import os
from typing import Dict, List, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from rag.memory import format_chat_history_for_prompt
from utils.logger import logger
from utils.helpers import format_citations

def _get_response_llm() -> Any:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.3
        )
    except Exception as e:
        logger.warning(f"Response Agent failed to init LLM: {e}")
        return None

def response_generation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the grounded response using Gemini 2.5 Flash based on context and history."""
    logger.info("Executing Response Generation Agent...")
    question = state.get("question", "")
    optimized_context = state.get("optimized_context", [])
    chat_history = state.get("chat_history", [])
    validation_feedback = state.get("validation_feedback", "")
    
    # 1. Format context for prompt
    if not optimized_context:
        context_str = "No document context is available. The database is empty or no relevant chunks were retrieved."
    else:
        context_blocks = []
        for i, doc in enumerate(optimized_context):
            meta = doc.metadata
            fname = meta.get("file_name", "Unknown")
            page = meta.get("page_number", 1)
            context_blocks.append(f"[{i+1}] Source Document: {fname} (Page {page})\nContent:\n{doc.page_content}")
        context_str = "\n\n".join(context_blocks)
        
    # 2. Format memory
    history_str = format_chat_history_for_prompt(chat_history)
    
    llm = _get_response_llm()
    if not llm:
        logger.warning("Gemini API key missing or LLM failed to load. Returning fallback answer.")
        return {
            "response": "Error: Gemini API key is missing. Please set your GEMINI_API_KEY environment variable.",
            "citations": []
        }
        
    # Build System Prompt with strict hallucination reduction rules
    system_prompt = (
        "You are an expert Response Generation Agent for a RAG system.\n"
        "Your task is to answer the user's question using ONLY the provided Document Context below.\n\n"
        "STRICT GROUNDING RULES:\n"
        "- Answer the question using ONLY facts directly mentioned in the Document Context.\n"
        "- Do NOT assume, extrapolate, or bring in outside knowledge. Everything must be grounded.\n"
        "- If the Document Context does not contain enough information to answer the question, you must respond EXACTLY with:\n"
        "  'The uploaded documents do not contain enough information.'\n"
        "- When you reference facts from a numbered document block (e.g. [1], [2]), please include inline citations like [Document: Name, Page: Num] where appropriate in your sentences.\n"
        "- Do NOT invent citations.\n"
    )
    
    # Adjust prompt if there is validation feedback from the Validation Agent
    feedback_section = ""
    if validation_feedback:
        feedback_section = (
            f"\n\nWARNING: Your previous answer was rejected by the Validation Agent for the following reason:\n"
            f"'{validation_feedback}'\n"
            f"Please revise your answer to address this issue. Ensure strict adherence to the Document Context."
        )
        
    user_prompt = (
        "Document Context:\n"
        "----------------------\n"
        "{context}\n"
        "----------------------\n\n"
        "Chat History:\n"
        "{history}\n\n"
        "User Question: {question}"
        "{feedback_section}\n\n"
        "Grounded Answer:"
    )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt)
    ])
    
    try:
        formatted_prompt = prompt_template.format_messages(
            context=context_str,
            history=history_str,
            question=question,
            feedback_section=feedback_section
        )
        
        response = llm.invoke(formatted_prompt)
        answer_text = response.content.strip()
        
        # 3. Extract citations
        citations = format_citations(optimized_context)
        
        logger.info("Successfully generated response.")
        return {
            "response": answer_text,
            "citations": citations
        }
    except Exception as e:
        logger.error(f"Error in response generation: {e}", exc_info=True)
        return {
            "response": f"An error occurred while generating response: {e}",
            "citations": []
        }
