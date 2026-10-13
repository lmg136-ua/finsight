import pytest
from unittest.mock import patch, MagicMock
from app.retrieval.retriever import FinSightRetriever
from langchain_core.documents import Document

@patch('app.retrieval.retriever.Chroma')
@patch('app.retrieval.retriever.get_embeddings')
def test_retriever_search(mock_get_embeddings, mock_chroma):
    mock_vectorstore = MagicMock()
    mock_vectorstore.get.return_value = {"documents": [], "metadatas": []}
    mock_chroma.return_value = mock_vectorstore
    
    # Setup mock return for similarity search
    mock_doc = Document(page_content="Consolidated Revenue grew 10%", metadata={"company": "Microsoft", "year": 2024, "page": 42, "source_filename": "msft.pdf"})
    mock_vectorstore.similarity_search_with_score.return_value = [(mock_doc, 0.8)]
    
    retriever = FinSightRetriever()
    results = retriever.search("growth", company="Microsoft", top_k=5)
    
    assert len(results) == 1
    assert results[0].text == "Consolidated Revenue grew 10%"
    assert results[0].metadata.company == "Microsoft"
    assert results[0].metadata.page == 42
    mock_vectorstore.similarity_search_with_score.assert_called_with("growth", k=5, filter={"company": "Microsoft"})

@patch('app.retrieval.retriever.Chroma')
@patch('app.retrieval.retriever.get_embeddings')
def test_retriever_toc_safe_filter(mock_get_embeddings, mock_chroma):
    mock_vectorstore = MagicMock()
    mock_vectorstore.get.return_value = {"documents": [], "metadatas": []}
    mock_chroma.return_value = mock_vectorstore
    
    # Filing page with 'Table of Contents' header but containing actual financial revenue statements
    doc_with_header = Document(
        page_content="Table of Contents\nAlphabet Inc. Consolidated Statements of Income\nRevenues: $350,000 million",
        metadata={"company": "Alphabet", "year": 2024, "page": 53, "source_filename": "goog.pdf"}
    )
    mock_vectorstore.similarity_search_with_score.return_value = [(doc_with_header, 0.9)]
    
    retriever = FinSightRetriever()
    results = retriever.search("Alphabet revenues", company="Alphabet", top_k=5)
    
    assert len(results) == 1
    assert "Consolidated Statements of Income" in results[0].text
    assert results[0].metadata.company == "Alphabet"
