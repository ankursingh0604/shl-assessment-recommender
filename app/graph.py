"""
The agent graph.

parse_node -> (conditional route) -> retrieve_node (for recommend) -------\
                                   -> compare_lookup_node (for compare)    >-- compose_node -> grounding_filter
                                   -> noop (for clarify / refuse) --------/

Why a graph instead of one big prompt: the four conversational behaviors the
brief asks for (clarify / recommend / refine / compare / refuse) are
fundamentally different *operations* — one does no retrieval and asks a
question, one does keyword retrieval and writes a shortlist, one looks up two
specific items and diffs them, one declines. Routing them explicitly means
each path gets exactly the context it needs, and a reviewer can see the
decision point in code rather than trusting it happened inside a prompt.

Grounding is enforced AFTER the LLM responds, not just requested in the
prompt: `selected_urls` gets intersected with the actual candidate set before
becoming the response's `recommendations`. An LLM that hallucinates a URL
gets silently corrected, not trusted.
"""
import os
from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

from .catalog import Catalog, CatalogEntry
from .llm import call_structured
from .prompts import PARSE_SYSTEM, COMPOSE_SYSTEM

MAX_TURNS = 8


class ParseResult(BaseModel):
    action: Literal["clarify", "recommend", "compare", "refuse"]
    facts_summary: str = ""
    search_queries: list[str] = Field(default_factory=list)
    compare_targets: list[str] = Field(default_factory=list)
    clarifying_question: str = ""
    refusal_reason: str = ""


class ComposeResult(BaseModel):
    reply: str
    selected_urls: list[str] = Field(default_factory=list)
    end_of_conversation: bool = False


class GraphState(TypedDict, total=False):
    messages: list[dict]
    parse: ParseResult
    candidates: list[CatalogEntry]
    compose: ComposeResult
    forced_end: bool


def _history_text(messages: list[dict]) -> str:
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


def build_graph(catalog: Catalog):

    def parse_node(state: GraphState) -> GraphState:
        history = _history_text(state["messages"])
        try:
            result = call_structured(
                system=PARSE_SYSTEM,
                user=f"Conversation so far:\n{history}\n\nProduce the structured analysis.",
                schema=ParseResult,
            )
        except Exception:
            # If the parse stage itself fails (provider hiccup etc.), fail
            # safe into a clarifying question rather than crashing the request.
            result = ParseResult(action="clarify",
                                  clarifying_question="Could you tell me a bit more about the role and what you need the assessment for?")
        return {"parse": result}

    def retrieve_node(state: GraphState) -> GraphState:
        queries = state["parse"].search_queries or [state["parse"].facts_summary]
        candidates = catalog.multi_search(queries, top_k_each=8, top_k_total=20)

        # Safety net: also keep retrievable anything the agent already named
        # in an earlier turn (see find_names_mentioned docstring — this
        # covers the case where search_queries this turn focus only on
        # alternatives to an existing item, not the item itself, which would
        # otherwise silently starve it out of the candidate pool even if the
        # compose stage wants to keep it).
        history_text = _history_text(state["messages"])
        mentioned = catalog.find_names_mentioned(history_text)
        seen_urls = {c.url for c in candidates}
        for m in mentioned:
            if m.url not in seen_urls:
                candidates.append(m)
                seen_urls.add(m.url)

        return {"candidates": candidates}

    def compare_lookup_node(state: GraphState) -> GraphState:
        found = []
        for target in state["parse"].compare_targets:
            hit = catalog.fuzzy_find(target)
            if hit:
                found.append(hit)
        return {"candidates": found}

    def noop_node(state: GraphState) -> GraphState:
        return {"candidates": []}

    def compose_node(state: GraphState) -> GraphState:
        parse = state["parse"]
        candidates = state.get("candidates", [])
        history = _history_text(state["messages"])

        if parse.action == "clarify":
            user_block = (f"Action: clarify\nClarifying question to ask: {parse.clarifying_question}\n"
                          f"Conversation:\n{history}")
        elif parse.action == "refuse":
            user_block = (f"Action: refuse\nReason: {parse.refusal_reason}\n"
                          f"Conversation:\n{history}")
        elif parse.action == "compare":
            cand_block = "\n".join(f"- {c.name} | {c.test_type_str} | {c.description} | {c.url}" for c in candidates)
            user_block = (f"Action: compare\nTargets requested: {parse.compare_targets}\n"
                          f"CANDIDATES:\n{cand_block or '(none found)'}\nConversation:\n{history}")
        else:  # recommend
            cand_block = "\n".join(f"- {c.name} | {c.test_type_str} | {c.description} | {c.url}" for c in candidates)
            user_block = (f"Action: recommend\nFacts: {parse.facts_summary}\n"
                          f"CANDIDATES:\n{cand_block or '(none found)'}\nConversation:\n{history}")

        try:
            result = call_structured(system=COMPOSE_SYSTEM, user=user_block, schema=ComposeResult)
        except Exception:
            result = ComposeResult(
                reply="Sorry, I hit an issue putting that together. Could you rephrase or give me a bit more detail?",
                selected_urls=[], end_of_conversation=False,
            )
        return {"compose": result}

    graph = StateGraph(GraphState)
    graph.add_node("parse", parse_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("compare_lookup", compare_lookup_node)
    graph.add_node("noop", noop_node)
    graph.add_node("compose", compose_node)

    def route(state: GraphState) -> str:
        action = state["parse"].action
        if action == "recommend":
            return "retrieve"
        if action == "compare":
            return "compare_lookup"
        return "noop"

    graph.set_entry_point("parse")
    graph.add_conditional_edges("parse", route, {
        "retrieve": "retrieve", "compare_lookup": "compare_lookup", "noop": "noop",
    })
    graph.add_edge("retrieve", "compose")
    graph.add_edge("compare_lookup", "compose")
    graph.add_edge("noop", "compose")
    graph.add_edge("compose", END)

    return graph.compile()


# Imported here (after class defs) to avoid a circular-looking top-of-file
# block; kept as plain module-level constants for prompt clarity in tests.
from .prompts import PARSE_SYSTEM as PARSE_SYSTEM, COMPOSE_SYSTEM as COMPOSE_SYSTEM


def run_turn(catalog: Catalog, compiled_graph, messages: list[dict]) -> dict:
    """Runs one /chat turn. Returns a dict matching ChatResponse fields.
    Turn-cap enforcement lives here, outside the graph, because it's a hard
    protocol rule (max 8 turns) rather than a conversational decision — it
    should not be something the LLM could reason its way around."""
    turn_count = len(messages)
    at_cap = turn_count >= MAX_TURNS

    state = compiled_graph.invoke({"messages": messages})
    parse: ParseResult = state["parse"]
    compose: ComposeResult = state["compose"]
    candidates: list[CatalogEntry] = state.get("candidates", [])

    candidate_urls = {c.url for c in candidates}
    grounded = [c for c in candidates if c.url in candidate_urls and c.url in set(compose.selected_urls)]
    recommendations = [c.to_recommendation_dict() for c in grounded][:10]

    end_of_conversation = compose.end_of_conversation
    reply = compose.reply

    if at_cap and not end_of_conversation:
        # Force a resolution rather than silently exceeding the harness's
        # turn budget. If we still have nothing grounded to show, say so
        # honestly instead of fabricating a shortlist.
        end_of_conversation = True
        if not recommendations:
            reply = reply + "\n\n(We're at the end of this conversation's turn limit — based on what's been shared, I don't have enough to ground a confident shortlist yet.)"

    return {
        "reply": reply,
        "recommendations": recommendations,
        "end_of_conversation": end_of_conversation,
    }