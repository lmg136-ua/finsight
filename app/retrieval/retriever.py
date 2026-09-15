from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma
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

    def add_documents(self, documents: List[Document]):
        """Adds processed documents to the vector store."""
        self.vectorstore.add_documents(documents)
        # Chroma modern API auto-persists in the background or when deleted, 
        # but in older versions it required vectorstore.persist(). 
        # Using it here to ensure compatibility depending on Chroma version installed.
        if hasattr(self.vectorstore, 'persist'):
            self.vectorstore.persist()

    def search(self, query: str, company: Optional[str] = None, top_k: int = 5) -> List[RetrievedChunk]:
        """
        Searches the vector store.
        Optionally filters by company.
        """
        filter_dict = {}
        if company:
            filter_dict["company"] = company
            
        # Using similarity_search_with_score
        results = self.vectorstore.similarity_search_with_score(
            query, 
            k=top_k, 
            filter=filter_dict if filter_dict else None
        )
        
        retrieved = []
        for doc, score in results:
            metadata = ChunkMetadata(
                company=doc.metadata.get("company", "Unknown"),
                document_type=doc.metadata.get("document_type", "Unknown"),
                year=doc.metadata.get("year", 0),
                page=doc.metadata.get("page", 0),
                source_filename=doc.metadata.get("source_filename", "Unknown"),
                section=doc.metadata.get("section")
            )
            
            chunk = RetrievedChunk(
                text=doc.page_content,
                metadata=metadata,
                score=float(score)
            )
            retrieved.append(chunk)
            
        return retrieved
