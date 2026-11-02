def route_after_bias_review(state):
    # Every rewrite returns here
    issues = state["bias_review"].get("issues") or []
    if issues and state["revision"] < state["max_revisions"]:
        return "rewrite"
    return "evaluate"
