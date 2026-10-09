"""
Central configuration, loaded from environment variables / .env.
Every other module should import `settings` from here rather than
reading os.environ directly, so all tunable values live in one place.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config =    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    # Which provider to use for generation + validation. "gemini" is free;
    # "anthropic" is available if you later want to switch to a paid key.
    llm_provider: str = "gemini"

    # Gemini (free tier) -- get a key at https://aistudio.google.com/apikey
    gemini_api_key: str = ""
    gemini_model_name: str = "gemini-2.5-flash"

    # Anthropic (paid) -- kept for optional later use
    anthropic_api_key: str = ""
    model_name: str = "claude-sonnet-4-5"

    chroma_dir: str = "./chroma_db"
    chroma_collection: str = "medquad"

    retrieval_confidence_threshold: float = 0.35
    retrieval_top_k: int = 3


settings = Settings()
