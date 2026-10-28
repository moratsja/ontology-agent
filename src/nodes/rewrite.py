import json

from src.llm.model import complete
from src.nodes.extract import format_documents

def rewrite(state):
    revised = complete("", _user_message(state))
    return {
        "summary": revised,
        "summaries": [revised],
        "revision": state["revision"] + 1,
        "trace": ["rewrite"],
    }

def _user_message(state):
    return "\n".join(
        [
            f"Topic: {state['topic']}",
            f"User request: {state['user_prompt']}",
            f"Write exactly {state['k']} sentences.",
            "",
            "Current summary:",
            state["summary"],
            "",
            "Extracted claims:",
            json.dumps(state.get("claims") or [], ensure_ascii=False, indent=2),
            "",
            "Claim checks for this summary (SUPPORTED, CONTRADICTED, or NOT_VERIFIABLE):",
            json.dumps(state.get("claim_evaluations") or [], ensure_ascii=False, indent=2),
            "",
            "Problems to fix in this summary:",
            json.dumps(state["issues"], ensure_ascii=False, indent=2),
            "",
            "Earlier revisions. Use these only to avoid repeating a contradicted claim. They are not problems in the current summary:",
            _earlier(state),
            "",
            "Ontology query results:",
            json.dumps(state["query_results"], ensure_ascii=False, indent=2),
            "",
            "Posts:",
            format_documents(state["documents"]),
            "",
            f"Rewrite the summary in exactly {state['k']} sentences.",
            "Fix claims marked CONTRADICTED. The ontology reason is under Problems to fix.",
            "Keep claims marked SUPPORTED.",
            "NOT_VERIFIABLE means the ontology cannot check the claim. Do not treat it as false.",
            "Leave a NOT_VERIFIABLE claim unchanged unless the posts themselves do not support it.",
            "Do not add facts that are not in the posts or in a SUPPORTED check.",
            "Return only the revised summary.",
        ]
    )

def _earlier(state):
    history = state.get("evaluations") or []
    earlier = history[:-1]
    if not earlier:
        return "None."
    lines = []
    for item in earlier:
        contradicted = [
            verdict["message"]
            for verdict in item.get("claims") or []
            if verdict.get("status") == "CONTRADICTED"
        ]
        detail = "; ".join(contradicted) if contradicted else "none"
        lines.append(f"revision {item['revision']}: contradicted: {detail}")
    return "\n".join(lines)
