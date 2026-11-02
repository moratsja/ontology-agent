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
    return (onto.City & Not(onto.Person))

def _persons_that_are_not_cities(onto):
    return (onto.Person & Not(onto.City))

def _capital_cities_in_europe(onto):
    return (onto.CapitalCity & onto.isLocatedIn.value(onto.europe))

def _events_with_traditions(onto):
    return (onto.Event & onto.hasTradition.some(onto.Tradition))

def _religious_events_hosted_in_ethiopia(onto):
    return (onto.ReligiousEvent & onto.isHostedBy.value(onto.ethiopia))

def _streaming_services_offering_series(onto):
    return (onto.StreamingService & onto.offersContent.some(onto.TVSeries))

def _netflix_content(onto):
    return (onto.MediaContent & onto.isOfferedBy.value(onto.netflix))

def _netflix_series_with_episode(onto):
    return (onto.TVSeries & onto.isOfferedBy.value(onto.netflix) & onto.seriesHasEpisode.some(onto.TVEpisode))

QUERIES = (
    OntologyQuery(
        id="q1",
        title="Cities that are not persons",
        manchester="City and not Person",
        topics=("Paris",),
        build=_cities_that_are_not_persons,
    ),

    OntologyQuery(
        id="q2",
        title="Persons that are not cities",
        manchester="Person and not City",
        topics=("Paris",),
        build=_persons_that_are_not_cities,
    ),  

    OntologyQuery(
        id="q3",
        title="Capital cities in Europe",
        manchester="CapitalCity and (isLocatedIn value europe)",
        topics=("Paris",),
        build=_capital_cities_in_europe,
    ),   

    OntologyQuery(
        id="q4",
        title="Streaming services that offer some TV series",
        manchester="StreamingService and (offersContent some TVSeries)",
        topics=("Netflix",),
        build=_streaming_services_offering_series,
    ),  

    OntologyQuery(
        id="q5",
        title="Content offered by Netflix",
        manchester="MediaContent and (isOfferedBy value netflix)",
        topics=("Netflix",),
        build=_netflix_content,
    ),

    OntologyQuery(
        id="q6",
        title="TV series offered by Netflix with some episode",
        manchester="TVSeries and (isOfferedBy value netflix) and (seriesHasEpisode some TVEpisode)",
        topics=("Netflix",),
        build=_netflix_series_with_episode,
    ),

    OntologyQuery(
        id="q7",
        title="Events with a tradition",
        manchester="Event and (hasTradition some Tradition)",
        topics=("Christmas",),
        build=_events_with_traditions,
    ),
    OntologyQuery(
        id="q8",
        title="Religious events hosted in Ethiopia",
        manchester="ReligiousEvent and (isHostedBy value ethiopia)",
        topics=("Christmas",),
        build=_religious_events_hosted_in_ethiopia,
    ),
    
)

def queries_for(topic):
    wanted = topic.lower()
    return [query for query in QUERIES if wanted in {name.lower() for name in query.topics}]
