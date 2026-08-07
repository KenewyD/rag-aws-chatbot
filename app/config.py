import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    AWS_REGION = os.getenv("AWS_REGION", "eu-west-3")
    AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

    # OpenSearch (local via Docker par défaut)
    OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "opensearch")
    OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
    OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "rag-documents")
    OPENSEARCH_USE_SSL = os.getenv("OPENSEARCH_USE_SSL", "false").lower() == "true"

    # Bedrock
    EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
    LLM_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"

    # RAG
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    TOP_K = 5


settings = Settings()
