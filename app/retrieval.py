import json

import boto3
from opensearchpy import OpenSearch

from app.config import settings


class RAGChain:
    def __init__(self):
        self.bedrock = boto3.client("bedrock-runtime", region_name=settings.AWS_REGION)
        self.client = OpenSearch(
            hosts=[{"host": settings.OPENSEARCH_HOST, "port": settings.OPENSEARCH_PORT}],
            use_ssl=settings.OPENSEARCH_USE_SSL,
            verify_certs=False,
        )

    def _get_embedding(self, text: str):
        response = self.bedrock.invoke_model(
            modelId=settings.EMBEDDING_MODEL,
            body=json.dumps({"inputText": text}),
        )
        return json.loads(response["body"].read())["embedding"]

    def _retrieve(self, query: str, top_k: int = 5):
        embedding = self._get_embedding(query)
        search_body = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": embedding,
                        "k": top_k,
                    }
                }
            },
        }
        response = self.client.search(index=settings.OPENSEARCH_INDEX, body=search_body)
        hits = response["hits"]["hits"]
        return [
            {
                "text": hit["_source"]["text"],
                "score": hit["_score"],
                "metadata": hit["_source"]["metadata"],
            }
            for hit in hits
        ]

    def _generate(self, query: str, context: str) -> str:
        prompt = f"""Tu es un assistant expert. Réponds à la question en te basant UNIQUEMENT sur le contexte fourni.
Si tu ne trouves pas la réponse dans le contexte, dis-le clairement.

Contexte :
{context}

Question : {query}

Réponse :"""

        response = self.bedrock.invoke_model(
            modelId=settings.LLM_MODEL,
            body=json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "messages": [{"role": "user", "content": prompt}],
                }
            ),
        )
        return json.loads(response["body"].read())["content"][0]["text"]

    def ask(self, query: str) -> dict:
        docs = self._retrieve(query, settings.TOP_K)
        context = "\n\n---\n\n".join([d["text"] for d in docs])
        answer = self._generate(query, context)

        return {
            "query": query,
            "answer": answer,
            "sources": docs,
            "retrieval_count": len(docs),
        }
