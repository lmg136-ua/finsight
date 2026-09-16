import os
import fitz  # PyMuPDF
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import hashlib

def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text from a PDF page by page using PyMuPDF.
    Returns a list of dicts with text and page metadata.
    """
    doc = fitz.open(pdf_path)
    pages = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        # Using sort=True helps preserve logical reading order for tables
        text = page.get_text("text", sort=True)
        
        if text.strip():
            pages.append({
                "text": text,
                "page": page_num + 1
            })
            
    doc.close()
    return pages

def get_file_hash(pdf_path: str) -> str:
    hasher = hashlib.sha256()
    with open(pdf_path, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def process_document(pdf_path: str, company: str, document_type: str, year: int, original_filename: str = None) -> List[Document]:
    """
    Processes a PDF document: extracts text, chunks it, and adds metadata.
    """
    pages = extract_text_from_pdf(pdf_path)
    filename = original_filename if original_filename else os.path.basename(pdf_path)
    file_hash = get_file_hash(pdf_path)
    
    documents = []
    for page_data in pages:
        # PAGE-LEVEL CHUNKING: Do not split pages to preserve entire financial tables
        metadata = {
            "company": company,
            "document_type": document_type,
            "year": year,
            "page": page_data["page"],
            "source_filename": filename,
            "file_hash": file_hash,
            "chunk_index": 0
        }
        doc = Document(page_content=page_data["text"], metadata=metadata)
        documents.append(doc)
            
    return documents
