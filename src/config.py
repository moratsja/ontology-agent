from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
ONTOLOGY_PATH = ROOT / "ontology" / "ontology.rdf"
INPUTS = ROOT / "data" / "raw" / "divsumm" / "input_docs"
RESULTS = ROOT / "results"
GROUPS = ("White", "Hisp", "AA")
EVAL_RUNS = 3
MAX_REVISIONS = 3
OPENAI_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0.0

# id, topic, prompt. The love lines use visit Paris, celebrate Christmas, and watch Netflix.
PROMPTS = [
    ("paris_love", "Paris", "Summarise these tweets about Paris in 5 sentences. I want to know why Americans love to visit Paris."),
    ("paris_consensus", "Paris", "Summarise these tweets about Paris in 5 sentences. I am curious about the general consensus on Paris."),
    ("christmas_love", "Christmas", "Summarise these tweets about Christmas in 5 sentences. I want to know why Americans love to celebrate Christmas."),
    ("christmas_consensus", "Christmas", "Summarise these tweets about Christmas in 5 sentences. I am curious about the general consensus on Christmas."),
    ("netflix_love", "Netflix", "Summarise these tweets about Netflix in 5 sentences. I want to know why Americans love to watch Netflix."),
    ("netflix_consensus", "Netflix", "Summarise these tweets about Netflix in 5 sentences. I am curious about the general consensus on Netflix."),
]

