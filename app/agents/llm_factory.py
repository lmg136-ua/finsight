from langchain_core.language_models.chat_models import BaseChatModel
from app.config import config

def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    Returns the configured LLM instance based on the environment variable.
    """
    provider = config.DEFAULT_LLM_PROVIDER.lower()
    
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")
        return ChatOpenAI(model="gpt-4o", temperature=temperature, max_retries=3)
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        if not config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set.")
        return ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=temperature, max_retries=3)
    elif provider == "groq":
        from langchain_groq import ChatGroq
        import os
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is not set in environment variables.")
        # We use a model well-suited for tool calling and reasoning
        return ChatGroq(model="openai/gpt-oss-120b", temperature=temperature, api_key=groq_api_key, max_retries=5)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
