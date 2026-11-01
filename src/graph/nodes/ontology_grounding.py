from src.ontology.reasoner import ground

def ontology_grounding(state):
    return {"ontology_grounding": ground(state.get("topic") or "")}
