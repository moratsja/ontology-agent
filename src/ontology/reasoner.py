# we had to use some AI coding assistance to get this working.
import shutil
from pathlib import Path

import owlready2
from owlready2 import World, sync_reasoner

from src.ontology.loader import load_ontology
from src.ontology.queries import QUERIES, queries_for

def _java_command():
    found = shutil.which("java")
    if found:
        return found
    root = Path(r"C:\Program Files\Eclipse Adoptium")
    matches = sorted(root.glob("*/bin/java.exe")) if root.is_dir() else []
    if matches:
        return str(matches[-1])
    return "java"

owlready2.JAVA_EXE = _java_command()

_CACHE = None

# grounds the topic into a dictionary of queries and facts
def ground(topic, query_ids=None):
    reasoned = _reasoned()
    onto = reasoned["onto"]
    if query_ids is None:
        selected = queries_for(topic)
    else:
        wanted = {str(query_id) for query_id in query_ids}
        selected = [query for query in QUERIES if query.id in wanted]
    queries = []
    names = _topic_names(onto, topic)
    for query in selected:
        row = reasoned["results"][query.id]
        queries.append({
            "id": query.id,
            "title": query.title,
            "results": row["results"],
            "error": row["error"],
        })
        names.extend(row["results"])
    return {
        "topic": topic,
        "queries": queries,
        "facts": _facts(onto, names),
    }

# formats the grounding into a string
def format_grounding(grounding):
    if not grounding:
        return ""
    if isinstance(grounding, str):
        return grounding
    lines = []
    for query in grounding.get("queries") or []:
        if query.get("error"):
            lines.append(f"{query['id']} {query['title']}: {query['error']}")
            continue
        found = ", ".join(query.get("results") or []) or "none"
        lines.append(f"{query['id']} {query['title']}: {found}")
    facts = grounding.get("facts") or []
    if facts:
        lines.append("Facts:")
        lines.extend(facts)
    return "\n".join(lines)

# returns a dictionary of query ids and their results
def query_results():
    reasoned = _reasoned()
    return {query_id: set(row["results"]) for query_id, row in reasoned["results"].items()}

def _reasoned():
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    world = World()
    onto = load_ontology(world)
    prepared = _prepare(onto, QUERIES)
    sync_reasoner(world, infer_property_values=True, debug=0)
    results = {}
    for query, cls, problem in prepared:
        if problem:
            results[query.id] = {"results": [], "error": problem}
            continue
        results[query.id] = {
            "results": [individual.name for individual in cls.instances()],
            "error": None,
        }
    _CACHE = {"onto": onto, "results": results}
    return _CACHE

def _prepare(onto, queries):
    prepared = []
    for query in queries:
        try:
            expression = query.build(onto)
        except Exception as exc:
            prepared.append((query, None, str(exc)))
            continue
        with onto:
            cls = type(query.id, (owlready2.Thing,), {"equivalent_to": [expression]})
        prepared.append((query, cls, None))
    return prepared

def _topic_names(onto, topic):
    prefix = topic.lower()
    if not prefix:
        return []
    return [individual.name for individual in onto.individuals() if individual.name.lower().startswith(prefix)]

def _facts(onto, names):
    skip = {query.id for query in QUERIES}
    lines = []
    seen = set()
    for name in names:
        individual = getattr(onto, name, None)
        if individual is None:
            continue
        classes = [
            cls.name
            for cls in individual.is_a
            if getattr(cls, "name", None) and cls.name not in skip and cls.name != "Thing"
        ]
        if classes:
            _add(lines, seen, f"{name} type {', '.join(classes)}")
        for prop in onto.object_properties():
            for value in prop[individual]:
                value_name = getattr(value, "name", None)
                if value_name:
                    _add(lines, seen, f"{name} {prop.name} {value_name}")
    return lines

def _add(lines, seen, line):
    if line not in seen:
        seen.add(line)
        lines.append(line)
