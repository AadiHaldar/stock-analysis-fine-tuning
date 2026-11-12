import re
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def split_text_into_chunks(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[str]:
    """
    Splits long document text into overlapping chunks using regex delimiters.
    """
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split text by sentences to prevent sentence fracturing
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_len = len(sentence)
        if current_length + sentence_len > chunk_size and current_chunk:
            # Join and add current chunk
            chunks.append(" ".join(current_chunk))
            
            # Start a new chunk, keeping the last few sentences as overlap
            overlap_chunk = []
            overlap_len = 0
            for prev_sent in reversed(current_chunk):
                if overlap_len + len(prev_sent) < chunk_overlap:
                    overlap_chunk.insert(0, prev_sent)
                    overlap_len += len(prev_sent)
                else:
                    break
            current_chunk = overlap_chunk
            current_length = overlap_len
            
        current_chunk.append(sentence)
        current_length += sentence_len
        
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def load_financial_documents(
    directory_path: Path,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[Dict[str, Any]]:
    """
    Loads all text files (.txt, .md) in a directory, chunks them,
    and formats them with metadata.
    """
    documents = []
    
    if not directory_path.exists():
        logger.warning(f"Directory {directory_path} does not exist. No files loaded.")
        return []
        
    # Search for files
    file_patterns = ["*.txt", "*.md"]
    for pattern in file_patterns:
        for file_path in directory_path.glob(pattern):
            logger.info(f"Loading document: {file_path}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                # Split content into chunks
                chunks = split_text_into_chunks(content, chunk_size, chunk_overlap)
                
                # Deduce ticker from filename if named like TSLA_sec.txt
                filename = file_path.stem
                ticker = "GLOBAL"
                for part in filename.split("_"):
                    if part.isupper() and len(part) in [3, 4]:
                        ticker = part
                        break
                        
                for i, chunk in enumerate(chunks):
                    documents.append({
                        "content": chunk,
                        "metadata": {
                            "source": file_path.name,
                            "chunk_id": i,
                            "ticker": ticker
                        }
                    })
                    
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                
    logger.info(f"Loaded {len(documents)} document chunks in total.")
    return documents
