import pytest
from unittest.mock import patch, MagicMock
from app.retrieval.retriever import FinSightRetriever
from langchain_core.documents import Document

@patch('app.retrieval.retriever.Chroma')
@patch('app.retrieval.retriever.get_embeddings')
def test_retriever_search(mock_get_embeddings, mock_chroma):
    mock_vectorstore = MagicMock()
    mock_chroma.return_value = mock_vectorstore
    
    # Setup mock return for similarity search
    mock_doc = Document(page_content="Revenue grew 10%", metadata={"company": "Microsoft"})
    mock_vectorstore.similarity_search_with_score.return_value = [(mock_doc, 0.8)]
    
    retriever = FinSightRetriever()
    results = retriever.search("growth", company="Microsoft")
    
    assert len(results) == 1
    assert results[0].text == "Revenue grew 10%"
    assert results[0].metadata.company == "Microsoft"
    mock_vectorstore.similarity_search_with_score.assert_called_with("growth", k=5, filter={"company": "Microsoft"})
