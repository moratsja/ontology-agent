from src.llm.model import complete
from src.graph.tweets import format_documents
from src.graph.nodes.bias_review import bias_feedback
from src.ontology.reasoner import format_grounding

REWRITE_SYSTEM_PROMPT = """Rewrite the summary in exactly 5 sentences. Only change the summary when the bias review suggests it.
Fix problems in the bias review.
Use earlier revisions only to avoid repeating a contradicted claim. They are not problems in the current summary.
The bias review is not an ontology verdict.
Return only the revised summary.
"""

def rewrite(state):
    revised = complete(REWRITE_SYSTEM_PROMPT, _user_message(state))
    return {
        "summary": revised,
        "summaries": [revised],
        "revision": state["revision"] + 1,
    }

def _user_message(state):
    return "\n".join(
        [
            f"User request: {state['user_prompt']}",
            "",
            "Summary:",
            state["summary"],
            "",
            "Ontology Grounding:",
            format_grounding(state.get("ontology_grounding")),
            "",
            "Earlier revisions:",
            _earlier(state),
            "",
            "Bias review:",
            bias_feedback(state.get("bias_review")),
            "",
            "Posts:",
            format_documents(state["documents"]),
        ]
    )

# formats the earlier revisions 
def _earlier(state):
    earlier = (state.get("summaries") or [])[:-1]
    if not earlier:
        return "None."
    blocks = [f"Revision {number}:\n{summary}" for number, summary in enumerate(earlier, start=1)]
    return "\n\n".join(blocks)
    