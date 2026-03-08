from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "docsense_connectors"

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"

    # Claude
    anthropic_api_key: str
    claude_model: str = "claude-sonnet-4-20250514"
    claude_max_tokens: int = 1024

    # Retrieval
    default_top_k: int = 5
    chunk_size_tokens: int = 400
    chunk_overlap_tokens: int = 50
    chunking_strategy: str = "heading"

    model_config = {"env_file": ".env"}


settings = Settings()
