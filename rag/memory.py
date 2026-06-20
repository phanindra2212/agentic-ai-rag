from typing import List, Dict, Any

def format_chat_history_for_prompt(chat_history: List[Dict[str, str]], limit: int = 5) -> str:
    """Formats list of message dicts into a plain text history block for the LLM.
    
    Format:
        User: ...
        Assistant: ...
    """
    if not chat_history:
        return "No prior conversation history."
        
    # Take the last 'limit' turns
    recent_history = chat_history[-limit:]
    formatted = []
    
    for message in recent_history:
        role = "User" if message.get("role") == "user" else "Assistant"
        content = message.get("content", "")
        formatted.append(f"{role}: {content}")
        
    return "\n".join(formatted)

def clear_chat_history(session_state_chat: List[Any]) -> None:
    """Clears the session state chat history."""
    session_state_chat.clear()
