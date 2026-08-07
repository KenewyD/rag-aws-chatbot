import json

import boto3
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from opensearchpy import OpenSearch

from app.config import settings


class IngestionPipeline:
    def __init__(self):
        self.bedrock = boto3.client("bedrock-runtime", region_name=settings.AWS_REGION)
        self.client = OpenSearch(
            hosts=[{"host": settings.OPENSEARCH_HOST, "port": settings.OPENSEARCH_PORT}],
            use_ssl=settings.OPENSEARCH_USE_SSL,
            verify_certs=False,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        self._ensure_index()

    def _ensure_index(self):
        if not self.client.indices.exists(settings.OPENSEARCH_INDEX):
            self.client.indices.create(
                index=settings.OPENSEARCH_INDEX,
                body={
                    "settings": {"index": {"knn": True}},
                    "mappings": {
                        "properties": {
                            "text": {"type": "text"},
                            "embedding": {
                                "type": "knn_vector",
                                "dimension": 1024,
                                "method": {
                                    "name": "hnsw",
                                    "space_type": "cosinesimil",
                                    "engine": "nmslib",
                                },
                            },
                            "metadata": {"type": "object"},
                        }
                    },
                },
            )

    def _get_embedding(self, text: str):
        response = self.bedrock.invoke_model(
            modelId=settings.EMBEDDING_MODEL,
            body=json.dumps({"inputText": text}),
        )
        return json.loads(response["body"].read())["embedding"]

    def ingest_file(self, file_path: str):
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)

        documents = loader.load()
        chunks = self.splitter.split_documents(documents)

        for i, chunk in enumerate(chunks):
            embedding = self._get_embedding(chunk.page_content)
            self.client.index(
                index=settings.OPENSEARCH_INDEX,
                body={
                    "text": chunk.page_content,
                    "embedding": embedding,
                    "metadata": {
                        "source": file_path,
                        "page": chunk.metadata.get("page", 0),
                        "chunk_id": i,
                    },
                },
            )
        return len(chunks)
