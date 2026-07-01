"""
Thin LLM wrapper.

Defaults to Groq (OpenAI-compatible API, generous free tier, fast — same
provider used in the Voice Intent Router project) but works with any
OpenAI-compatible endpoint by changing LLM_BASE_URL / LLM_MODEL env vars.
Swap to Anthropic/OpenAI directly by changing get_client() if preferred.

Two responsibilities live here, not just "call the API":
1. Force JSON output and parse it.
2. If parsing/validation fails, do ONE repair pass where we show the model
   its own broken output plus the validation error and ask it to fix just
   that, rather than silently retrying blind or crashing the request. This
   is the difference between "happy path only" and handling the realistic
   failure mode of LLMs occasionally emitting malformed JSON under load.
"""
import json
import os
from typing import Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        )
    return _client


def _model() -> str:
    return os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")


def call_structured(system: str, user: str, schema: Type[T], temperature: float = 0.0) -> T:
    """Call the LLM, force JSON output, parse+validate into `schema`.
    One repair attempt on failure; raises on a second failure so the caller's
    own fallback logic (always defined at the graph-node level) takes over
    rather than this module silently returning something unvalidated."""
    client = get_client()
    messages = [
        {"role": "system", "content": system + "\n\nRespond with ONLY a single JSON object. No prose, no markdown fences."},
        {"role": "user", "content": user},
    ]
    resp = client.chat.completions.create(
        model=_model(), messages=messages, temperature=temperature,
        response_format={"type": "json_object"}, max_tokens=1500,
    )
    raw = resp.choices[0].message.content
    try:
        return schema.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as e:
        repair_messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": f"That output failed validation: {e}\n"
                                         f"Return a corrected JSON object only, matching the required schema."},
        ]
        resp2 = client.chat.completions.create(
            model=_model(), messages=repair_messages, temperature=0.0,
            response_format={"type": "json_object"}, max_tokens=1500,
        )
        raw2 = resp2.choices[0].message.content
        return schema.model_validate(json.loads(raw2))  # let this raise if still broken


def call_text(system: str, user: str, temperature: float = 0.3) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model=_model(),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature, max_tokens=600,
    )
    return resp.choices[0].message.content.strip()
