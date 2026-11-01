def route_after_bias_review(state):
    issues = state["bias_review"].get("issues") or []
    if issues and state["revision"] < state["max_revisions"]:
        return "rewrite"
    return "evaluate"
