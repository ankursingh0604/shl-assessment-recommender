"""
Request/response schemas for the SHL Assessment Recommender API.

These map 1:1 onto the spec in the assignment doc. Kept deliberately strict
(extra="forbid" on the response side) because the brief is explicit that
schema deviation breaks the automated evaluator — better to fail loudly in
our own tests than silently ship an extra/missing field.
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]

    @field_validator("messages")
    @classmethod
    def non_empty(cls, v):
        if not v:
            raise ValueError("messages must contain at least one entry")
        return v


class Recommendation(BaseModel):
    model_config = {"extra": "forbid"}
    name: str
    url: str
    test_type: str  # comma-joined letter codes, e.g. "P" or "K,S" — matches spec example


class ChatResponse(BaseModel):
    model_config = {"extra": "forbid"}
    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False

    @field_validator("recommendations")
    @classmethod
    def bounded_shortlist(cls, v):
        if len(v) > 10:
            raise ValueError("recommendations must contain at most 10 items")
        return v


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
