from dataclasses import dataclass
from typing import Any, Callable
from owlready2 import Not

@dataclass(frozen=True)
class OntologyQuery:
    id: str
    title: str
    manchester: str
    topics: tuple[str, ...]
    build: Callable[[Any], Any]


def _cities_that_are_not_persons(onto):
    return onto.City & Not(onto.Person)

def _persons_that_are_not_cities(onto):
    return onto.Person & Not(onto.City)

def _cities_in_france(onto):
    return onto.City & onto.isLocatedIn.value(onto.france)

def _events_hosted_in_europe(onto):
    return onto.Event & onto.isHostedBy.some(onto.isLocatedIn.value(onto.europe))

def _events_with_traditions(onto):
    return onto.Event & onto.hasTradition.some(onto.Tradition)

def _religious_events_hosted_in_ethiopia(onto):
    return onto.ReligiousEvent & onto.isHostedBy.value(onto.ethiopia)

def _netflix_content(onto):
    return onto.MediaContent & onto.isOfferedBy.value(onto.netflix)

def _netflix_romance_content(onto):
    return onto.MediaContent & onto.isOfferedBy.value(onto.netflix) & onto.hasGenre.value(onto.romance)

def _capital_cities(onto):
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
        id="q2",
        title="Cities that are not persons",
        manchester="City and not Person",
        topics=("Paris"),
        build=_cities_that_are_not_persons,
    ),

    OntologyQuery(
        id="q3",
        title="Persons that are not cities",
        manchester="Person and not City",
        topics=("Paris"),
        build=_persons_that_are_not_cities,
    ),  

    OntologyQuery(
        id="q4",
        title="Cities in France",
        manchester="City and (isLocatedIn value france)",
        topics=("Paris"),
        build=_cities_in_france,
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
