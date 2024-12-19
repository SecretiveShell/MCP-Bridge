from typing import Optional, List
from pydantic import BaseModel


class Delta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class Choice(BaseModel):
    index: int
    delta: Delta
    logprobs: Optional[dict] = None
    finish_reason: Optional[str] = None


class SSEData(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
