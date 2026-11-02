from typing import Annotated, TypedDict
import operator

class AgentState(TypedDict):
    topic: str
    documents: list[dict[str, str]]
    user_prompt: str
    prompt_id: str
    max_revisions: int
    summary: str
    summaries: Annotated[list[str], operator.add]
    ontology_grounding: dict
    bias_review: dict
    evaluation: dict
    started: float
    revision: int