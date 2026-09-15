import pytest
from unittest.mock import patch, MagicMock
from app.retrieval.ingestion import process_document

@patch('app.retrieval.ingestion.fitz.open')
def test_process_document(mock_fitz_open):
    # Mocking a PDF document with one page
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.get_text.return_value = "This is a test financial report for Microsoft."
    mock_doc.__len__.return_value = 1
    mock_doc.load_page.return_value = mock_page
    mock_fitz_open.return_value = mock_doc

    docs = process_document("dummy.pdf", "Microsoft", "10-K", 2023)
    
    assert len(docs) > 0
    assert docs[0].metadata["company"] == "Microsoft"
    assert docs[0].metadata["year"] == 2023
    assert "test financial report" in docs[0].page_content
