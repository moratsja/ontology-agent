from dataclasses import dataclass
from typing import Any, Callable

@dataclass(frozen=True)
class OntologyQuery:
    id: str
    title: str
    manchester: str
    topics: tuple[str, ...]
    build: Callable[[Any], Any]

def _events_hosted_in_europe(onto):
    # Event and (isHostedBy some (isLocatedIn value europe))
    return onto.Event & onto.isHostedBy.some(onto.isLocatedIn.value(onto.europe))

def _events_with_traditions(onto):
    # Event and (hasTradition some Tradition)
    return onto.Event & onto.hasTradition.some(onto.Tradition)

def _religious_events_hosted_in_ethiopia(onto):
    # ReligiousEvent and (isHostedBy value ethiopia)
    return onto.ReligiousEvent & onto.isHostedBy.value(onto.ethiopia)

def _netflix_content(onto):
    # MediaContent and (isOfferedBy value netflix)
    return onto.MediaContent & onto.isOfferedBy.value(onto.netflix)

def _netflix_romance_content(onto):
    # MediaContent and (isOfferedBy value netflix) and (hasGenre value romance)
    return onto.MediaContent & onto.isOfferedBy.value(onto.netflix) & onto.hasGenre.value(onto.romance)

def _capital_cities(onto):
    # CapitalCity and (isCapitalOf some Country)
    return onto.CapitalCity & onto.isCapitalOf.some(onto.Country)

QUERIES = (
    OntologyQuery(
        id="q1",
        title="Events hosted in Europe",
        manchester="Event and (isHostedBy some (isLocatedIn value europe))",
        topics=("Paris", "Christmas"),
        build=_events_hosted_in_europe,
    ),
    OntologyQuery(
        id="c1",
        title="Events with a tradition",
        manchester="Event and (hasTradition some Tradition)",
        topics=("Christmas",),
        build=_events_with_traditions,
    ),
    OntologyQuery(
        id="c2",
        title="Religious events hosted in Ethiopia",
        manchester="ReligiousEvent and (isHostedBy value ethiopia)",
        topics=("Christmas",),
        build=_religious_events_hosted_in_ethiopia,
    ),
    OntologyQuery(
        id="n1",
        title="Content offered by Netflix",
        manchester="MediaContent and (isOfferedBy value netflix)",
        topics=("Netflix",),
        build=_netflix_content,
    ),
    OntologyQuery(
        id="n2",
        title="Netflix content with genre romance",
        manchester="MediaContent and (isOfferedBy value netflix) and (hasGenre value romance)",
        topics=("Netflix",),
        build=_netflix_romance_content,
    ),
    OntologyQuery(
        id="p1",
        title="Capital cities",
        manchester="CapitalCity and (isCapitalOf some Country)",
        topics=("Paris",),
        build=_capital_cities,
    ),
)

def queries_for(topic):
    wanted = topic.lower()
    return [query for query in QUERIES if wanted in {name.lower() for name in query.topics}]
