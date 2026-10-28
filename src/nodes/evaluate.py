def evaluate(state):
    claims = state.get("claims") or []
    grouped = state.get("query_results_by_claim") or []
    verdicts = []
    for index, claim in enumerate(claims):
        item = _item_for(index, grouped)
        status = item.get("status") or "NOT_VERIFIABLE"
        if status not in {"SUPPORTED", "CONTRADICTED", "NOT_VERIFIABLE"}:
            status = "NOT_VERIFIABLE"
        message = item.get("message") or "No ontology check was stored for this claim."
        verdicts.append({
            "status": status,
            "query_id": item.get("template") or "",
            "message": message,
            "claim": claim,
        })
    issues = [
        {
            "kind": "inconsistency",
            "query_id": verdict["query_id"],
            "message": verdict["message"],
        }
        for verdict in verdicts
        if verdict["status"] == "CONTRADICTED"
    ]
    passed = bool(verdicts) and all(_is_supported(verdict) for verdict in verdicts)
    return {
        "claim_evaluations": verdicts,
        "issues": issues,
        "passed": passed,
        "stop_reason": loop_stop_reason(state, issues, passed),
        "evaluations": [revision_snapshot(state["revision"], verdicts, issues, passed)],
        "trace": ["evaluate"],
    }

def revision_snapshot(revision, claims, issues, passed):
    return {
        "revision": revision,
        "passed": passed,
        "issues": issues,
        "claims": claims,
    }

def loop_stop_reason(state, issues, passed, no_claims=False):
    if no_claims:
        return "no_claims"
    if passed:
        return "passed"
    contradicted = any(issue.get("kind") == "inconsistency" for issue in issues)
    if not contradicted:
        return "no_contradiction"
    if state["revision"] >= state["max_revisions"]:
        return "rewrite_limit"
    return ""

def _is_supported(verdict):
    if verdict["status"] != "SUPPORTED":
        return False
    message = verdict["message"]
    if message.startswith("Reasoner error") or "inconsistent before" in message:
        return False
    return True

def _item_for(index, grouped):
    for item in grouped:
        if item.get("claim_index") == index:
            return item
    return {}
