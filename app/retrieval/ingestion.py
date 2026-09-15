import os
import fitz  # PyMuPDF
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text from a PDF page by page using PyMuPDF.
    Returns a list of dicts with text and page metadata.
    """
    doc = fitz.open(pdf_path)
    pages = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        
        if text.strip():
            pages.append({
                "text": text,
                "page": page_num + 1
            })
            
    doc.close()
    return pages

def process_document(pdf_path: str, company: str, document_type: str, year: int) -> List[Document]:
    """
    Processes a PDF document: extracts text, chunks it, and adds metadata.
    """
    pages = extract_text_from_pdf(pdf_path)
    filename = os.path.basename(pdf_path)
    
    # We use RecursiveCharacterTextSplitter to create reasonable sized chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    documents = []
    for page_data in pages:
        chunks = text_splitter.split_text(page_data["text"])
        
        for i, chunk in enumerate(chunks):
            metadata = {
                "company": company,
                "document_type": document_type,
                "year": year,
                "page": page_data["page"],
                "source_filename": filename,
                "chunk_index": i
            }
            doc = Document(page_content=chunk, metadata=metadata)
            documents.append(doc)
            
    return documents
