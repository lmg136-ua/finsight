from langchain_core.language_models.chat_models import BaseChatModel
from app.config import config

def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    Returns the configured LLM instance based on the environment variable.
    Default is Google Gemini, fallback to OpenAI.
    """
    provider = config.DEFAULT_LLM_PROVIDER.lower()
    
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")
        return ChatOpenAI(model="gpt-4o", temperature=temperature)
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        if not config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set.")
        return ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=temperature)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
