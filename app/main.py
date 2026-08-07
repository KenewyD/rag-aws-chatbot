import os
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.ingestion import IngestionPipeline
from app.retrieval import RAGChain

app = FastAPI(title="RAG Enterprise API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGChain()


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "service": "rag-aws-api"}


@app.post("/ask")
def ask(req: AskRequest):
    try:
        result = rag.ask(req.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
def ingest(file: UploadFile = File(...)):
    try:
        os.makedirs("data/documents", exist_ok=True)
        file_path = f"data/documents/{file.filename}"
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        pipeline = IngestionPipeline()
        chunks = pipeline.ingest_file(file_path)
        return {"message": "Ingested", "chunks": chunks, "file": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
