import os
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from app.config import config

def get_llm(temperature: float = 0.0, provider: Optional[str] = None, model: Optional[str] = None) -> BaseChatModel:
    """
    Returns the configured LLM instance with robust fallback support.
    Supported providers: groq, google, openai.
    """
    target_provider = (provider or config.DEFAULT_LLM_PROVIDER).lower()
    
    if target_provider == "groq":
        from langchain_groq import ChatGroq
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is not set in environment variables.")
        
        # Support user-specified model or reliable production default
        target_model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        return ChatGroq(
            model=target_model, 
            temperature=temperature, 
            api_key=groq_api_key, 
            max_retries=3
        )
        
    elif target_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeai, ChatGoogleGenerativeAI
        google_api_key = config.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY is not set.")
        
        # gemini-2.0-flash is ultra-fast, robust, and supports structured output
        target_model = model or os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
        return ChatGoogleGenerativeAI(
            model=target_model, 
            temperature=temperature, 
            google_api_key=google_api_key,
            max_retries=3
        )
        
    elif target_provider == "openai":
        from langchain_openai import ChatOpenAI
        openai_api_key = config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
            
        target_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(
            model=target_model, 
            temperature=temperature, 
            api_key=openai_api_key,
            max_retries=3
        )
        
    else:
        raise ValueError(f"Unsupported LLM provider: {target_provider}. Choose 'groq', 'google', or 'openai'.")
