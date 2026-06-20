import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """Cleans text content by normalizing whitespaces and line endings."""
    if not text:
        return ""
    # Split lines, clean spaces on each line, and remove extra tabs/spaces
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(lines)
    # Replace multiple empty lines with double newlines
    text = re.sub(r"\n\n+", "\n\n", text)
    return text.strip()

def get_file_extension(file_path: str) -> str:
    """Returns lowercase file extension of a file path."""
    return Path(file_path).suffix.lower()

def get_current_timestamp() -> str:
    """Returns the current timestamp in ISO format."""
    return datetime.now().isoformat()

def format_citations(documents: List[Any]) -> List[Dict[str, Any]]:
    """Formats retrieved LangChain Documents into a clean list of sources with metadata."""
    citations = []
    seen = set()
    for doc in documents:
        meta = doc.metadata
        file_name = meta.get("file_name", "Unknown Source")
        page = meta.get("page_number", 1)
        chunk_id = meta.get("chunk_id", "")
        
        # Unique identifier to avoid duplicate citation displays
        citation_key = f"{file_name}_p{page}"
        if citation_key not in seen:
            seen.add(citation_key)
            citations.append({
                "file_name": file_name,
                "page_number": page,
                "file_type": meta.get("file_type", ""),
                "source": meta.get("source", ""),
                "chunk_id": chunk_id,
                "content_snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
    return citations
