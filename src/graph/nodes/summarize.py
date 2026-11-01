from src.llm.model import complete
from src.graph.tweets import format_documents
from src.ontology.reasoner import format_grounding

def summarize(state):
    message = _user_message(state)
    print(message)
    summary = complete("", message)
    return {"summary": summary, "summaries": [summary]}

def _user_message(state):
    tweets = format_documents(state["documents"])
    return "\n".join(
        [
            f"User request: {state['user_prompt']}",
            "",
            "Ontology grounding:",
            format_grounding(state.get("ontology_grounding")),
            "",
            "Posts:",
            tweets,
        ]
    )