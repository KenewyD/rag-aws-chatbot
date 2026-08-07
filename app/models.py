from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    text: str
    score: float
    metadata: dict


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[Source]
    retrieval_count: int
