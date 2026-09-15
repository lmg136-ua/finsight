from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import config

def get_embeddings():
    """
    Returns the configured embeddings model.
    Using HuggingFace sentence-transformers for local, free embeddings.
    """
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
