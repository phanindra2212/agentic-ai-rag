from pathlib import Path
from typing import List
from langchain_core.documents import Document

from langchain_community.document_loaders import PyPDFLoader
from utils.logger import logger

def load_pdf(file_path: str) -> List[Document]:
    """Loads a PDF document and extracts its pages as LangChain Documents.
    
    Injects metadata: file_name, file_type, page_number, source.
    """
    logger.info(f"Starting PDF loading for file: {file_path}")
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
        
    try:
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        
        # Standardize metadata
        for i, doc in enumerate(docs):
            doc.metadata["file_name"] = path.name
            doc.metadata["file_type"] = "PDF"
            doc.metadata["page_number"] = i + 1  # 1-indexed
            doc.metadata["source"] = str(path)
            
        logger.info(f"Successfully loaded PDF. Pages: {len(docs)}")
        return docs
    except Exception as e:
        logger.error(f"Error loading PDF file {file_path}: {e}", exc_info=True)
        raise e
