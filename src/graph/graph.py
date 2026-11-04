import time

from langgraph.graph import START, END, StateGraph

from src.config import MAX_REVISIONS
from src.graph.state import AgentState
from src.graph.routing import route_after_bias_review
from src.graph.nodes.ontology_grounding import ontology_grounding
from src.graph.nodes.summarize import summarize
from src.graph.nodes.bias_review import bias_review
from src.graph.nodes.rewrite import rewrite
from src.graph.nodes.evaluate import evaluate

# build agent graph
def agent_graph():
    builder = StateGraph(AgentState)

    builder.add_node("ontology_grounding", ontology_grounding)
    builder.add_node("summarize", summarize)
    builder.add_node("bias_review", bias_review)
    builder.add_node("rewrite", rewrite)
    builder.add_node("evaluate", evaluate)

    builder.add_edge(START, "ontology_grounding")
    builder.add_edge("ontology_grounding", "summarize")
    builder.add_edge("summarize", "bias_review")
    builder.add_edge("rewrite", "bias_review")
    builder.add_conditional_edges(
        "bias_review", # the node we just left
        route_after_bias_review, # the function to decide where to go next
        {"rewrite": "rewrite", "evaluate": "evaluate"}, # the possible next nodes
    )
    builder.add_edge("evaluate", END)
    return builder.compile()

# baseline for the comparison run
def baseline_graph():
    builder = StateGraph(AgentState)
    builder.add_node("summarize", summarize)
    builder.add_node("bias_review", bias_review)
    builder.add_node("rewrite", rewrite)
    builder.add_node("evaluate", evaluate)
    
    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", "bias_review")
    builder.add_edge("bias_review", "rewrite")
    builder.add_edge("rewrite", "evaluate")
    builder.add_conditional_edges(
        "bias_review",
        route_after_bias_review,
        {"rewrite": "rewrite", "evaluate": "evaluate"},
    )
    builder.add_edge("evaluate", END)
    return builder.compile()

def initial_state(
    documents,
    user_prompt,
    topic,
    prompt_id = "",
    max_revisions = MAX_REVISIONS, # from config
):
    return {
        "topic": topic,
        "documents": documents,
        "user_prompt": user_prompt,
        "prompt_id": prompt_id,
        "max_revisions": max_revisions,
        "summary": "",
        "summaries": [],
        "ontology_grounding": {},
        "bias_review": {"issues": []},
        "evaluation": {},
        "started": time.time(),
        "revision": 0,
    }