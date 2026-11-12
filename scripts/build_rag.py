import sys
from pathlib import Path
import logging

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings
from src.rag.embeddings import FinancialEmbedder
from src.rag.vector_store import LocalFAISSStore
from src.rag.document_loader import load_financial_documents

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def create_sample_documents(doc_dir: Path):
    """
    Creates sample financial documents in data/raw/documents/ for demonstrating RAG capabilities.
    """
    doc_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample 1: Tesla SEC Filing Summary
    tsla_doc = doc_dir / "TSLA_sec_summary.txt"
    if not tsla_doc.exists():
        with open(tsla_doc, "w", encoding="utf-8") as f:
            f.write(
                "Tesla Inc. SEC Form 10-K Summary (FY2025)\n"
                "Tesla reported total automotive revenues of $82.4 billion, representing a 12% increase year-over-year. "
                "The gross margin for automotive operations fell slightly to 18.2% due to price adjustments in key markets like China and Europe. "
                "Operating income reached $9.1 billion, driven by reductions in cost of goods sold (COGS) and manufacturing efficiencies. "
                "Capital expenditures for gigafactories in Texas and Berlin reached $7.8 billion. "
                "Tesla plans to begin production of its next-generation platform in late 2026, aiming to lower vehicle costs below $25,000. "
                "Furthermore, the company is increasing research spending in FSD Supercomputing clusters by $1.2 billion."
            )
            
    # Sample 2: Apple Earnings Call Transcript Segment
    aapl_doc = doc_dir / "AAPL_earnings_call.txt"
    if not aapl_doc.exists():
        with open(aapl_doc, "w", encoding="utf-8") as f:
            f.write(
                "Apple Inc. Q1 2026 Earnings Call Highlights\n"
                "Tim Cook announced a record revenue of $124.3 billion, up 6% from Q1 2025. "
                "iPhone revenue accounted for $69.7 billion, showing strong demand for the iPhone 17 series with edge-AI chips. "
                "Services revenue reached an all-time high of $25.1 billion, growing 14% year-over-year. "
                "Luca Maestri noted that the gross margin was 45.9%, at the high end of guidance. "
                "Apple returned $27 billion to shareholders through dividends and share repurchases. "
                "Regarding inventory levels, Apple expects supply chain bottlenecks to subside by late Q3. "
                "R&D expenditure rose to $8.2 billion, reflecting intensive investments in Apple Intelligence and spatial computing headsets."
            )
            
    logger.info("Sample financial documents created successfully.")

def main():
    logger.info("Building RAG Vector Index...")
    
    doc_dir = settings.RAW_DATA_DIR / "documents"
    create_sample_documents(doc_dir)
    
    # 1. Load documents
    chunks = load_financial_documents(doc_dir)
    if not chunks:
        logger.error("No document chunks loaded. Exiting build script.")
        return
        
    # 2. Extract contents and metadata
    texts = [c["content"] for c in chunks]
    metas = [c["metadata"] for c in chunks]
    
    # 3. Load embedder
    embedder = FinancialEmbedder()
    
    # 4. Generate embeddings
    logger.info("Generating dense vector embeddings for document chunks...")
    embeddings = embedder.embed_documents(texts)
    
    # 5. Create store and add vectors
    store = LocalFAISSStore(dimension=embeddings.shape[1])
    store.add_vectors(embeddings, texts, metas)
    
    # 6. Save store
    store.save()
    logger.info("RAG Vector Index built and saved successfully!")

if __name__ == "__main__":
    main()
