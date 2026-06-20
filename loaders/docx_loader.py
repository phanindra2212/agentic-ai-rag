from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader
from utils.logger import logger

def load_docx(file_path: str) -> List[Document]:
    """Loads a DOCX document and extracts its pages/text as LangChain Documents.
    
    Injects metadata: file_name, file_type, page_number, source.
    """
    logger.info(f"Starting DOCX loading for file: {file_path}")
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"DOCX file not found: {file_path}")
        
    try:
        loader = Docx2txtLoader(str(path))
        docs = loader.load()
        
        # Word documents are typically loaded as one single text document block in Docx2txtLoader
        for doc in docs:
            doc.metadata["file_name"] = path.name
            doc.metadata["file_type"] = "DOCX"
            doc.metadata["page_number"] = 1  # Standard page number fallback for DOCX
            doc.metadata["source"] = str(path)
            
        logger.info(f"Successfully loaded DOCX. Documents: {len(docs)}")
        return docs
    except Exception as e:
        logger.error(f"Error loading DOCX file {file_path}: {e}", exc_info=True)
        raise e
