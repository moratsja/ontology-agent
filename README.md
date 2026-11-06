# ontology-agent

Summarises DivSumm tweets about Paris, Christmas, and Netflix. The agent looks up facts in the ontology, writes a 5-sentence summary, and checks that summary for bias. If the check finds issues, the summary is rewritten and checked again, up to 3 rewrites. The last check is the one that goes on to scoring. The baseline writes the summary without grounding information from the ontology, but with the bias review. Both are scored the same way.

## Setup

```bash
cd ontology-agent
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install .
```

Put `OPENAI_API_KEY` in `.env`.

The ontology is `ontology/ontology.rdf`. The reasoner needs Java. If `java` is not on PATH, the code looks for `java.exe` under `C:\Program Files\Eclipse Adoptium`.

The NLI model used for equal coverage and the BARTScore model are downloaded the first time you run an evaluation. They are not in the repo.

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
- Cost (time): seconds the run took, from start to evaluation.
- Cost (tokens): total tokens summed over all LLM calls in the run, split into input and output tokens (prompt and completion). These are counted for the whole run, including the ontology lookup, every summary, every bias check and every rewrite. The agent therefore uses more tokens than the baseline.

## Model results (GPT-4o and Ollama) and report
The code was run using the GPT-4o model from OpenAI. `src/llm/model.py` uses `ChatOpenAI`, and `main.py` has no Ollama option or config.
This is because the Ollama results were produced seperately. Only the output files were added to the repo for comparison purposes against GPT-4o.
The local Ollama results are in `results/ollamarun/`:
- `ollama agent pipeline/`: agent runs
- `ollama baseline/`: baseline runs
  
Each folder contains 3 runs and an average file of those 3 runs, covering only the `paris_love` prompt. The same evaluation metrics are used as with the GPT-4o.
For the exact results of the GPT-4o and Ollama model, please check our final report. It lists the values of all evaluation metrics for every run and prompt, and explains the differences between the models and between the agent vs. baseline runs.
