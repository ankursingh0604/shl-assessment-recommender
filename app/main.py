"""
FastAPI service: GET /health, POST /chat.

Catalog + compiled graph are built once at import time (module-level), not
per-request — the catalog load + BM25 index build is the only "slow" setup
work, and re-running it per request would blow the 30s-per-call budget for
no reason. The service itself is stateless per the spec: nothing here is
written to disk or memory keyed by conversation.
"""
import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .catalog import Catalog
from .graph import build_graph, run_turn, MAX_TURNS
from .schemas import ChatRequest, ChatResponse, HealthResponse, Recommendation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shl_agent")

DATA_PATH = os.environ.get("CATALOG_PATH", str(Path(__file__).parent.parent / "data" / "catalog.json"))

app = FastAPI(title="SHL Assessment Recommender")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_catalog: Catalog | None = None
_graph = None


def get_catalog() -> Catalog:
    global _catalog, _graph
    if _catalog is None:
        logger.info(f"Loading catalog from {DATA_PATH}")
        _catalog = Catalog(DATA_PATH)
        _graph = build_graph(_catalog)
        logger.info(f"Catalog loaded: {len(_catalog)} items. Graph compiled.")
    return _catalog


@app.on_event("startup")
def _warm_start():
    # Load catalog + compile graph at startup so the FIRST /health hit after
    # a cold start (which the spec gives up to 2 minutes for) doesn't also
    # have to pay for index build on top of container boot.
    get_catalog()


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    get_catalog()  # ensures warm even if startup hook didn't fire (e.g. tests)
    if len(req.messages) > MAX_TURNS:
        # Hard protocol violation from the caller's side — still respond with
        # valid schema rather than raising, since "never break schema" beats
        # "technically correct error code" for an evaluator replaying turns.
        truncated = req.messages[-MAX_TURNS:]
    else:
        truncated = req.messages

    messages = [{"role": m.role, "content": m.content} for m in truncated]

    t0 = time.monotonic()
    try:
        result = run_turn(_catalog, _graph, messages)
    except Exception:
        logger.exception("chat turn failed")
        return ChatResponse(
            reply="Sorry, something went wrong on my end putting that together. Could you try rephrasing?",
            recommendations=[], end_of_conversation=False,
        )
    elapsed = time.monotonic() - t0
    if elapsed > 25:
        logger.warning(f"/chat turn took {elapsed:.1f}s — close to the 30s budget")

    return ChatResponse(
        reply=result["reply"],
        recommendations=[Recommendation(**r) for r in result["recommendations"]],
        end_of_conversation=result["end_of_conversation"],
    )
