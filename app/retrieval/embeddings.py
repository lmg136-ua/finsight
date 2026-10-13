import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import config

_EMBEDDINGS_INSTANCE = None

def get_embeddings():
    """
    Returns the configured embeddings model (cached singleton).
    Using HuggingFace sentence-transformers for local, free embeddings.
    """
    global _EMBEDDINGS_INSTANCE
    if _EMBEDDINGS_INSTANCE is None:
        # Avoid redundant downloads or network hangs
        _EMBEDDINGS_INSTANCE = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _EMBEDDINGS_INSTANCE
