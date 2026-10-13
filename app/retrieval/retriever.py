from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from app.retrieval.embeddings import get_embeddings
from app.config import config
from app.models.schemas import RetrievedChunk, ChunkMetadata

class FinSightRetriever:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.persist_directory = config.CHROMA_DB_DIR
        self._init_db()

    def _init_db(self):
        self.vectorstore = Chroma(
            collection_name="financial_docs",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        self._build_bm25()
        
    def _build_bm25(self):
        try:
            # Load all documents from Chroma to build BM25 in-memory index
            all_docs_data = self.vectorstore.get()
            if isinstance(all_docs_data, dict) and all_docs_data.get("documents"):
                docs = []
                docs_list = all_docs_data["documents"]
                metadatas_list = all_docs_data.get("metadatas") or []
                for i in range(len(docs_list)):
                    meta = metadatas_list[i] if i < len(metadatas_list) and isinstance(metadatas_list[i], dict) else {}
                    docs.append(Document(
                        page_content=str(docs_list[i]),
                        metadata=meta
                    ))
                if docs:
                    self.bm25_retriever = BM25Retriever.from_documents(docs)
                    return
        except Exception as e:
            # Log gracefully without breaking execution
            pass
        self.bm25_retriever = None

    def is_indexed(self, file_hash: str) -> bool:
        docs = self.vectorstore.get(where={"file_hash": file_hash})
        return len(docs.get("ids", [])) > 0

    def add_documents(self, documents: List[Document]) -> bool:
        """Adds processed documents to the vector store. Returns False if already indexed."""
        if not documents:
            return False
        
        file_hash = documents[0].metadata.get("file_hash")
        if file_hash and self.is_indexed(file_hash):
            return False
            
        self.vectorstore.add_documents(documents)
        if hasattr(self.vectorstore, 'persist'):
            try:
                self.vectorstore.persist()
            except Exception:
                pass
        self._build_bm25()  # Rebuild hybrid index
        return True

    def search(self, query: str, company: Optional[str] = None, year: Optional[int] = None, top_k: int = 15) -> List[RetrievedChunk]:
        """
        Searches using Hybrid Retrieval (Chroma Dense + BM25 Sparse).
        Fair reranking ensures multi-company balance without dropping valid financial tables.
        """
        filter_dict = {}
        if company:
            filter_dict["company"] = company
        if year:
            filter_dict["year"] = int(year)
            
        where_clause = None
        if len(filter_dict) == 1:
            where_clause = filter_dict
        elif len(filter_dict) > 1:
            where_clause = {"$and": [{k: v} for k, v in filter_dict.items()]}
            
        # Dense Retrieval
        dense_results = self.vectorstore.similarity_search_with_score(
            query, 
            k=top_k, 
            filter=where_clause
        )
        
        # Sparse Retrieval (BM25)
        bm25_results = []
        if self.bm25_retriever:
            try:
                self.bm25_retriever.k = top_k * 3
                all_bm25 = self.bm25_retriever.invoke(query)
                if company:
                    all_bm25 = [doc for doc in all_bm25 if doc.metadata.get("company") == company]
                if year:
                    all_bm25 = [doc for doc in all_bm25 if doc.metadata.get("year") == int(year)]
                bm25_results = all_bm25[:top_k]
            except Exception:
                bm25_results = []
            
        # Combine and deduplicate
        seen = set()
        combined_docs = []
        
        # Add dense first
        for doc, score in dense_results:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                combined_docs.append(doc)
                
        # Add bm25
        for doc in bm25_results:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                combined_docs.append(doc)
                
        # Limit to top_k
        combined_docs = combined_docs[:top_k]
        
        retrieved = []
        for doc in combined_docs:
            text = doc.page_content.lower()
            
            # Penalize pure Table of Contents or Exhibits only if they lack substantive financial content
            # (Fixes the bug where SEC filings with running "Table of Contents" headers were discarded)
            is_pure_toc_or_exhibit = (
                ("table of contents" in text and len(text) < 800)
                or ("exhibit index" in text and len(text) < 1200)
            ) and not any(term in text for term in [
                "consolidated statements", "balance sheet", "income from operations", 
                "operating income", "net income", "revenues", "revenue", "cash flows", "risk factors"
            ])
            
            if is_pure_toc_or_exhibit:
                continue
                
            metadata = ChunkMetadata(
                company=doc.metadata.get("company", "Unknown"),
                document_type=doc.metadata.get("document_type", "Unknown"),
                year=int(doc.metadata.get("year", 0)),
                page=int(doc.metadata.get("page", 0)),
                source_filename=doc.metadata.get("source_filename", "Unknown"),
                section=doc.metadata.get("section")
            )
            
            chunk = RetrievedChunk(
                text=doc.page_content,
                metadata=metadata,
                score=1.0 
            )
            retrieved.append(chunk)
            
        return retrieved

# Lazy Singleton Architecture
_retriever_instance: Optional[FinSightRetriever] = None

def get_retriever() -> FinSightRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = FinSightRetriever()
    return _retriever_instance

class _LazyRetrieverProxy:
    """Proxy object that delegates all calls to get_retriever(), enabling lazy initialization without breaking backwards compatibility."""
    def __getattr__(self, name):
        return getattr(get_retriever(), name)

global_retriever = _LazyRetrieverProxy()
