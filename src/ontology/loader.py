from owlready2 import get_ontology

from src.config import ONTOLOGY_PATH

# loads the ontology into the world
def load_ontology(world=None):
    if world is None:
        return get_ontology(str(ONTOLOGY_PATH)).load()
    return world.get_ontology(str(ONTOLOGY_PATH)).load()
