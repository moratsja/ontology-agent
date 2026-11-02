import json

from src.llm.model import complete
from src.ontology.queries import QUERIES
from src.ontology.reasoner import ground

_SYSTEM = """Choose which ontology queries to run. Return only JSON: {"query_ids": ["id", "id"]}
Each query line starts with an id and a topic tag. Return every id whose topic tag is the same as the topic in the user message. Include all of them. Leave out every id tagged with a different topic.
"""


def ontology_grounding(state):
    topic = state.get("topic") or ""
    chosen = choose_queries(topic, state.get("user_prompt") or "")
    grounding = ground(topic, chosen)
    grounding["selected_query_ids"] = chosen
    return {"ontology_grounding": grounding}


def choose_queries(topic, user_prompt):
    available = list(QUERIES)
    if not available:
        return []
    valid_ids = {query.id for query in available}
    catalogue = "\n".join(
        f"{query.id} | topic: {', '.join(query.topics)} | {query.title} | {query.manchester}" for query in available
    )
    message = "\n".join(
        [
            f"Topic: {topic}",
            "",
            f"User request: {user_prompt}",
            "",
            "Queries:",
            catalogue,
        ]
    )
    chosen = _parse(complete(_SYSTEM, message), valid_ids)
    if chosen is None:
        return [query.id for query in available]
    return chosen


def _parse(raw, valid_ids):
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("query_ids"), list):
        return None
    chosen = []
    seen = set()
    for query_id in data["query_ids"]:
        text = str(query_id)
        if text not in valid_ids or text in seen:
            continue
        seen.add(text)
        chosen.append(text)
    return chosen
