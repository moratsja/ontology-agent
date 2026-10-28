import sys

import src.graph.graph as graph_mod
from src.config import MAX_REVISIONS
from src.graph.graph import build_baseline_graph, build_graph, initial_state
from src.graph.routing import route_after_evaluate, route_after_extract
from src.nodes.claims import extract_claims
from src.nodes.evaluate import evaluate
from src.nodes.rewrite import _user_message, rewrite
from src.nodes.summarize import summarize


def _docs():
    return [{"text": "hello", "group": "White", "topic": "T"}]

def _claim(text):
    return {"subject": "s", "predicate": "p", "object": "o", "text": text}

def _grouped(status, message):
    return [{
        "claim_index": 0,
        "template": "type",
        "status": status,
        "message": message,
        "results": [],
    }]

def test_default_allows_three_rewrites():
    assert MAX_REVISIONS == 3
    state = initial_state(_docs(), "prompt", "T")
    assert state["max_revisions"] == 3
    assert state["summaries"] == []
    assert state["evaluations"] == []
    assert state["stop_reason"] == ""
    assert state["revision"] == 0

def test_summarize_appends_one_summary(monkeypatch):
    monkeypatch.setattr(sys.modules["src.nodes.summarize"], "complete", lambda system, user: "First summary.")
    result = summarize({
        "topic": "T",
        "user_prompt": "prompt",
        "k": 1,
        "documents": _docs(),
    })
    assert result["summary"] == "First summary."
    assert result["summaries"] == ["First summary."]

def test_not_verifiable_does_not_rewrite():
    state = initial_state(_docs(), "prompt", "T")
    state["summary"] = "A claim the ontology cannot check."
    state["claims"] = [_claim(state["summary"])]
    state["query_results_by_claim"] = _grouped("NOT_VERIFIABLE", "The ontology does not entail it.")
    result = evaluate(state)
    assert result["issues"] == []
    assert result["passed"] is False
    assert result["stop_reason"] == "no_contradiction"
    assert result["evaluations"][0]["claims"][0]["status"] == "NOT_VERIFIABLE"
    merged = {**state, "issues": result["issues"]}
    assert route_after_evaluate(merged) == "end"

def test_contradiction_rewrites_until_the_limit():
    state = initial_state(_docs(), "prompt", "T", max_revisions=3)
    state["summary"] = "Wrong."
    state["claims"] = [_claim("Wrong.")]
    state["query_results_by_claim"] = _grouped("CONTRADICTED", "Adding it makes the ontology inconsistent.")
    state["revision"] = 1
    result = evaluate(state)
    assert result["issues"][0]["kind"] == "inconsistency"
    assert result["stop_reason"] == ""
    assert route_after_evaluate({**state, "issues": result["issues"]}) == "rewrite"
    state["revision"] = 3
    result = evaluate(state)
    assert result["stop_reason"] == "rewrite_limit"
    assert route_after_evaluate({**state, "issues": result["issues"]}) == "end"

def test_supported_claims_pass_and_stop():
    state = initial_state(_docs(), "prompt", "T")
    state["summary"] = "Fine."
    state["claims"] = [_claim("Fine.")]
    state["query_results_by_claim"] = _grouped("SUPPORTED", "The reasoner finds that it follows.")
    result = evaluate(state)
    assert result["passed"] is True
    assert result["issues"] == []
    assert result["stop_reason"] == "passed"
    assert result["evaluations"][0]["revision"] == 0
    assert route_after_evaluate({**state, "issues": result["issues"]}) == "end"

def test_no_claims_records_evaluation_and_ends(monkeypatch):
    monkeypatch.setattr("src.llm.model.complete", lambda system, user: '{"claims": []}')
    state = initial_state(_docs(), "prompt", "T")
    state["summary"] = "Nothing to check."
    result = extract_claims(state)
    assert result["claims"] == []
    assert result["passed"] is False
    assert result["stop_reason"] == "no_claims"
    assert result["evaluations"] == [{
        "revision": 0,
        "passed": False,
        "issues": [],
        "claims": [],
    }]
    assert route_after_extract({"claims": []}) == "end"

def test_rewrite_uses_current_problems_and_keeps_old_ones_apart(monkeypatch):
    monkeypatch.setattr(sys.modules["src.nodes.rewrite"], "complete", lambda system, user: "Revised summary.")
    state = initial_state(_docs(), "prompt", "T", k=5)
    state["summary"] = "Current summary."
    state["revision"] = 1
    state["claims"] = [_claim("Current summary.")]
    state["claim_evaluations"] = [{
        "status": "CONTRADICTED",
        "query_id": "type",
        "message": "CURRENT",
        "claim": state["claims"][0],
    }]
    state["issues"] = [{"kind": "inconsistency", "query_id": "type", "message": "CURRENT"}]
    state["query_results"] = []
    state["evaluations"] = [
        {
            "revision": 0,
            "passed": False,
            "issues": [{"kind": "inconsistency", "query_id": "type", "message": "OLD"}],
            "claims": [{"status": "CONTRADICTED", "query_id": "type", "message": "OLD", "claim": {}}],
        },
        {
            "revision": 1,
            "passed": False,
            "issues": state["issues"],
            "claims": state["claim_evaluations"],
        },
    ]
    prompt = _user_message(state)
    current, _, earlier = prompt.partition("Earlier revisions")
    assert "Current summary." in current
    assert "CURRENT" in current
    assert "OLD" not in current
    assert "OLD" in earlier
    assert "NOT_VERIFIABLE means the ontology cannot check the claim" in prompt
    assert "exactly 5 sentences" in prompt
    result = rewrite(state)
    assert result["summary"] == "Revised summary."
    assert result["summaries"] == ["Revised summary."]
    assert result["revision"] == 2
    assert "evaluations" not in result

def test_history_follows_each_summary_and_stops_when_it_passes(monkeypatch):
    result = _run(monkeypatch, ["CONTRADICTED", "CONTRADICTED", "SUPPORTED"], ["v0", "v1", "v2"])
    assert result["summaries"] == ["v0", "v1", "v2"]
    assert result["summary"] == "v2"
    assert result["revision"] == 2
    assert result["trace"].count("rewrite") == 2
    assert [item["revision"] for item in result["evaluations"]] == [0, 1, 2]
    for item in result["evaluations"]:
        assert item["claims"][0]["claim"]["text"] == result["summaries"][item["revision"]]
    assert result["evaluations"][0]["issues"][0]["kind"] == "inconsistency"
    assert result["evaluations"][-1]["passed"] is True
    assert result["issues"] == []
    assert result["claims"] == [_claim("v2")]
    assert result["stop_reason"] == "passed"

def test_rewrite_limit_stops_with_the_contradiction(monkeypatch):
    result = _run(
        monkeypatch,
        ["CONTRADICTED", "CONTRADICTED", "CONTRADICTED", "CONTRADICTED"],
        ["v0", "v1", "v2", "v3"],
        max_revisions=3,
    )
    assert result["summaries"] == ["v0", "v1", "v2", "v3"]
    assert result["revision"] == 3
    assert result["trace"].count("rewrite") == 3
    assert result["passed"] is False
    assert result["issues"][0]["kind"] == "inconsistency"
    assert result["stop_reason"] == "rewrite_limit"
    assert len(result["evaluations"]) == 4

def test_not_verifiable_loop_does_not_rewrite(monkeypatch):
    result = _run(monkeypatch, ["NOT_VERIFIABLE"], ["v0"])
    assert result["trace"] == ["summarize", "extract_claims", "ontology_mapping", "select_query", "evaluate"]
    assert result["revision"] == 0
    assert result["summaries"] == ["v0"]
    assert result["issues"] == []
    assert result["passed"] is False
    assert result["stop_reason"] == "no_contradiction"

def test_baseline_only_summarizes(monkeypatch):
    monkeypatch.setattr(
        graph_mod,
        "summarize",
        lambda state: {"summary": "once", "summaries": ["once"], "trace": ["summarize"]},
    )
    graph = build_baseline_graph()
    result = graph.invoke(initial_state(_docs(), "prompt", "T", mode="baseline"))
    assert result["trace"] == ["summarize"]
    assert result["summary"] == "once"
    assert result["summaries"] == ["once"]
    assert result["revision"] == 0
    assert result["evaluations"] == []
    assert result["stop_reason"] == ""

def _run(monkeypatch, statuses, texts, max_revisions=3):
    pending = list(texts)
    step = {"n": 0}

    def fake_summarize(state):
        text = pending.pop(0)
        return {"summary": text, "summaries": [text], "trace": ["summarize"]}

    def fake_rewrite(state):
        text = pending.pop(0)
        return {
            "summary": text,
            "summaries": [text],
            "revision": state["revision"] + 1,
            "trace": ["rewrite"],
        }

    def fake_extract(state):
        return {
            "claims": [_claim(state["summary"])],
            "summary_sentences": [state["summary"]],
            "trace": ["extract_claims"],
        }

    def fake_mapping(state):
        return {"mapped_claims": [{}], "trace": ["ontology_mapping"]}

    def fake_select(state):
        status = statuses[step["n"]]
        step["n"] += 1
        return {
            "query_results": [{
                "query_id": "type",
                "manchester": "",
                "status": "ok",
                "detail": status,
                "rows": [],
            }],
            "query_results_by_claim": _grouped(status, status),
            "trace": ["select_query"],
        }

    monkeypatch.setattr(graph_mod, "summarize", fake_summarize)
    monkeypatch.setattr(graph_mod, "rewrite", fake_rewrite)
    monkeypatch.setattr(graph_mod, "extract_claims", fake_extract)
    monkeypatch.setattr(graph_mod, "ontology_mapping", fake_mapping)
    monkeypatch.setattr(graph_mod, "select_query", fake_select)
    graph = build_graph()
    return graph.invoke(initial_state(_docs(), "prompt", "T", max_revisions=max_revisions))
