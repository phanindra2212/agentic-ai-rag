from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config.settings import get_chroma_dir, CHROMA_COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP
from loaders.pdf_loader import load_pdf
from loaders.docx_loader import load_docx
from loaders.pptx_loader import load_pptx
from loaders.txt_loader import load_txt
from rag.embeddings import get_embeddings_model
from utils.logger import logger
from utils.helpers import clean_text, get_file_extension
from utils.metrics import update_document_stats

def get_vector_store() -> Chroma:
    """Initializes and returns the persistent Chroma vector store.
    Uses model-specific collections to avoid dimension mismatch errors when switching models.
    """
    embeddings = get_embeddings_model()
    # Ensure directory exists
    chroma_dir = get_chroma_dir()
    chroma_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate model-specific collection name
    model_identifier = "default"
    if hasattr(embeddings, "model_name"):
        model_identifier = embeddings.model_name
    elif hasattr(embeddings, "model"):
        model_identifier = embeddings.model
    elif hasattr(embeddings, "size"):
        model_identifier = f"fake_{embeddings.size}"
        
    # Clean the identifier for Chroma collection naming rules (alphanumeric, _ or - only)
    clean_id = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in model_identifier)
    clean_id = clean_id.strip("_")[:40]
    collection_name = f"{CHROMA_COLLECTION_NAME}_{clean_id}"
    
    logger.info(f"Using Chroma collection: {collection_name}")
    
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(chroma_dir)
    )

def extract_documents(file_path: str) -> List[Document]:
    """Extracts text from a file based on its extension."""
    ext = get_file_extension(file_path)
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    elif ext == ".pptx":
        return load_pptx(file_path)
    elif ext == ".txt":
        return load_txt(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def process_and_index_file(file_path: str) -> Tuple[int, int]:
    """Loads, chunks, extracts metadata, and indexes a file into Chroma DB.
    
    Returns:
        Tuple of (number of chunks, total pages)
    """
    logger.info(f"Indexing file: {file_path}")
    try:
        path = Path(file_path)
        file_name = path.name
        ext = get_file_extension(str(path))
        
        # 1. Load document
        docs = extract_documents(str(path))
        total_pages = len(docs)
        
        # 2. Clean content
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)
            
        # 3. Chunk documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        chunks = text_splitter.split_documents(docs)
        
        # 4. Add unique IDs and rich metadata to each chunk
        upload_time = datetime.now().isoformat()
        processed_chunks = []
        
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{file_name}_chunk_{idx}"
            chunk.metadata.update({
                "file_name": file_name,
                "file_type": ext[1:].upper() if ext.startswith(".") else ext.upper(),
                "chunk_id": chunk_id,
                "source": str(path),
                "upload_time": upload_time
            })
            processed_chunks.append(chunk)
            
        # 5. Connect to Chroma and handle duplicate prevention
        db = get_vector_store()
        
        # Check if files were previously indexed and delete them
        try:
            # Check if collection exists and delete old vectors for the same file
            existing = db.get(where={"file_name": file_name})
            if existing and existing.get("ids"):
                logger.info(f"Removing {len(existing['ids'])} existing chunks for {file_name} from database.")
                db.delete(ids=existing["ids"])
        except Exception as e:
            logger.warning(f"Could not check/delete existing entries for {file_name}: {e}", exc_info=True)
            
        # 6. Add new chunks to database with batching and exponential backoff retries
        if processed_chunks:
            import time
            chunk_ids = [chunk.metadata["chunk_id"] for chunk in processed_chunks]
            
            # Batch size of 40 to avoid hitting free-tier 100 requests-per-minute limits
            batch_size = 40
            total_batches = (len(processed_chunks) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_i = batch_idx * batch_size
                end_i = min(start_i + batch_size, len(processed_chunks))
                batch_docs = processed_chunks[start_i:end_i]
                batch_ids = chunk_ids[start_i:end_i]
                
                max_retries = 6
                base_delay = 5.0  # start sleeping at 5s, then 10s, 20s, 40s...
                
                for attempt in range(max_retries):
                    try:
                        db.add_documents(batch_docs, ids=batch_ids)
                        logger.info(f"Successfully indexed batch {batch_idx + 1}/{total_batches} for {file_name}.")
                        break
                    except Exception as e:
                        # Catch rate limit 429/RESOURCE_EXHAUSTED exceptions
                        err_msg = str(e).upper()
                        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "QUOTA" in err_msg:
                            delay = base_delay * (2 ** attempt)
                            logger.warning(
                                f"Rate limit hit during batch {batch_idx + 1} indexing. "
                                f"Retrying in {delay:.1f}s... (Attempt {attempt + 1}/{max_retries})"
                            )
                            time.sleep(delay)
                        else:
                            logger.error(f"Failed to add document batch {batch_idx + 1}: {e}", exc_info=True)
                            raise e
                else:
                    logger.critical(f"Failed to index batch {batch_idx + 1} after {max_retries} retries due to rate limits.")
                    raise RuntimeError("Document ingestion aborted: rate limits exceeded repeatedly.")
            
        # 7. Update analytics/metrics
        # Count unique documents in DB
        try:
            db_data = db.get()
            if db_data and db_data.get("metadatas"):
                unique_files = len(set(meta["file_name"] for meta in db_data["metadatas"]))
                total_db_chunks = len(db_data["ids"])
                update_document_stats(unique_files, total_db_chunks)
            else:
                update_document_stats(0, 0)
        except Exception as e:
            logger.error(f"Error calculating collection stats: {e}", exc_info=True)
            
        return len(processed_chunks), total_pages

    except Exception as e:
        logger.error(f"Critical error during document indexing for {file_path}: {e}", exc_info=True)
        raise e

def get_collection_statistics() -> Dict[str, Any]:
    """Returns analytics metadata about the Chroma database collection."""
    try:
        db = get_vector_store()
        data = db.get()
        if not data or not data.get("ids"):
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "document_list": []
            }
            
        metadatas = data["metadatas"]
        document_details = {}
        
        for meta in metadatas:
            fname = meta.get("file_name", "Unknown")
            if fname not in document_details:
                document_details[fname] = {
                    "file_name": fname,
                    "file_type": meta.get("file_type", "Unknown"),
                    "chunks": 0,
                    "pages": set(),
                    "upload_time": meta.get("upload_time", "Unknown")
                }
            document_details[fname]["chunks"] += 1
            if "page_number" in meta:
                document_details[fname]["pages"].add(meta["page_number"])
                
        doc_list = []
        for fname, info in document_details.items():
            doc_list.append({
                "file_name": fname,
                "file_type": info["file_type"],
                "chunks": info["chunks"],
                "pages": len(info["pages"]),
                "upload_time": info["upload_time"]
            })
            
        return {
            "total_documents": len(doc_list),
            "total_chunks": len(data["ids"]),
            "document_list": doc_list
        }
    except Exception as e:
        logger.error(f"Error retrieving collection stats: {e}")
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "document_list": []
        }

def remove_document_from_db(file_name: str) -> None:
    """Removes a file and all its chunks from Chroma DB."""
    logger.info(f"Request to delete file: {file_name} from database.")
    db = get_vector_store()
    try:
        existing = db.get(where={"file_name": file_name})
        if existing and existing.get("ids"):
            db.delete(ids=existing["ids"])
            logger.info(f"Deleted {len(existing['ids'])} chunks for {file_name}.")
            
            # Recalculate stats
            new_data = db.get()
            if new_data and new_data.get("metadatas"):
                unique_files = len(set(meta["file_name"] for meta in new_data["metadatas"]))
                total_db_chunks = len(new_data["ids"])
                update_document_stats(unique_files, total_db_chunks)
            else:
                update_document_stats(0, 0)
    except Exception as e:
        logger.error(f"Error deleting file {file_name} from database: {e}")
        raise e
