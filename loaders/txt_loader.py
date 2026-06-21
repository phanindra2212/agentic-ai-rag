from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from utils.logger import logger

def load_txt(file_path: str) -> List[Document]:
    """Loads a TXT file and extracts its content as LangChain Documents.
    
    Injects metadata: file_name, file_type, page_number, source.
    """
    logger.info(f"Starting TXT loading for file: {file_path}")
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"TXT file not found: {file_path}")
        
    try:
        # Load using utf-8 encoding to support diverse text characters
        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
        
        for doc in docs:
            doc.metadata["file_name"] = path.name
            doc.metadata["file_type"] = "TXT"
            doc.metadata["page_number"] = 1
            doc.metadata["source"] = str(path)
            
        logger.info("Successfully loaded TXT file.")
        return docs
    except Exception as e:
        logger.error(f"Error loading TXT file {file_path}: {e}", exc_info=True)
        raise e
