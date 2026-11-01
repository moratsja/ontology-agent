# ontology-agent

Summarises DivSumm tweets about Paris, Christmas, and Netflix. The agent looks up facts in the ontology, writes a 5-sentence summary, and checks that summary for bias. If the check finds issues, the summary is rewritten once. The baseline only writes the summary. Both are scored the same way.

## Setup

```bash
cd ontology-agent
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install .
```

Put `OPENAI_API_KEY` in `.env`.

The ontology is `ontology/ontology.rdf`. The reasoner needs Java. If `java` is not on PATH, the code looks for `java.exe` under `C:\Program Files\Eclipse Adoptium`.

The NLI model used for equal coverage, and the BARTScore model, are downloaded the first time you run an evaluation. They are not in the repo.

## Run

```bash
python main.py
```

This runs every prompt in `src/config.py`, three times for the agent and three times for the baseline. The model is `gpt-4o`.

One topic and one prompt:

```bash
python main.py --topic Paris --prompt paris_love
```

The prompt ids are `paris_love`, `paris_consensus`, `christmas_love`, `christmas_consensus`, `netflix_love`, and `netflix_consensus`. Passing only a topic prints the ids for that topic and stops.

## Output

Runs are saved under `results/`, for example `results/run1/agent/` and `results/run1/baseline/`. Each prompt also gets an average file.

The scores in those files are:

- Equal coverage across the White, Hisp, and AA tweets. Lower means the groups are closer to each other.
- BARTScore (`facebook/bart-large-cnn`): how likely the summary is given the tweets. Closer to zero is better.
- ROUGE-2 F1: mean overlap with the abstractive annotator summaries for that topic.
- Cost: seconds for that run.
