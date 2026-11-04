import argparse
import csv
import html
import json
from langchain_community.callbacks import get_openai_callback

from src.config import EVAL_RUNS, GROUPS, INPUTS, ONTOLOGY_PATH, OPENAI_MODEL, PROMPTS, RESULTS
from src.graph.graph import agent_graph, baseline_graph, initial_state



def main():
    prompts = _selected_prompts()
    if not ONTOLOGY_PATH.is_file():
        print(f"Ontology file missing: {ONTOLOGY_PATH}")
    folder = _new_results_folder()
    print(f"saving this run in {folder}")
    for prompt_id, topic, _prompt in prompts:
        print(f"{prompt_id} ({topic})")
    _run_setting(agent_graph(), "agent", folder, prompts)
    _run_setting(baseline_graph(), "baseline", folder, prompts)


def _selected_prompts():
    parser = argparse.ArgumentParser(description="Run the summarisation evaluation.")
    parser.add_argument("--topic", help="Topic name, for example Paris.")
    parser.add_argument("--prompt", help="Prompt id, for example paris_love.")
    args = parser.parse_args()
    if args.prompt:
        chosen = [row for row in PROMPTS if row[0] == args.prompt]
        if not chosen:
            ids = ", ".join(row[0] for row in PROMPTS)
            raise SystemExit(f"Unknown prompt {args.prompt!r}. Choose one of: {ids}")
        prompt_id, topic, _prompt = chosen[0]
        if args.topic and args.topic.lower() != topic.lower():
            raise SystemExit(f"{prompt_id} belongs to {topic}, not {args.topic}.")
        return chosen
    if args.topic:
        chosen = [row for row in PROMPTS if row[1].lower() == args.topic.lower()]
        if not chosen:
            topics = ", ".join(sorted({row[1] for row in PROMPTS}))
            raise SystemExit(f"Unknown topic {args.topic!r}. Choose one of: {topics}")
        ids = ", ".join(row[0] for row in chosen)
        raise SystemExit(f"Pass --prompt as well. Prompts for {args.topic}: {ids}")
    return list(PROMPTS)


def _new_results_folder():
    RESULTS.mkdir(parents=True, exist_ok=True)
    number = 1
    while (RESULTS / f"run{number}").exists():
        number += 1
    folder = RESULTS / f"run{number}"
    folder.mkdir()
    (folder / "agent").mkdir()
    (folder / "baseline").mkdir()
    return folder


def _run_setting(graph, label, folder, prompts):
    for prompt_id, topic, prompt in prompts:
        documents = _load_documents(INPUTS / f"{topic}.txt")
        print(f"\n{prompt_id} ({label})")
        rows = []
        for run_index in range(1, EVAL_RUNS + 1):
            state = initial_state(
                documents=documents,
                user_prompt=prompt,
                topic=topic,
                prompt_id=prompt_id,
                mode=label,
            )
            
            # Track tokens across all LLM calls in this run
            with get_openai_callback() as cb:
                result = graph.invoke(state)

            scores = result.get("evaluation") or {}

            scores["tokens"] = {
                "total_tokens": cb.total_tokens,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
            }

            rows.append(scores)
            _print_scores(run_index, scores)
            _print_run(run_index, result)
            _save_run(folder, label, run_index, result, scores)
        _save_average(folder, prompt_id, label, _average(rows))


def _print_scores(run_index, scores):
    coverage = (scores.get("equal_coverage") or {}).get("ec_score")
    quality = (scores.get("quality") or {}).get("score")
    rouge = scores.get("rouge2")
    seconds = (scores.get("cost") or {}).get("seconds")
    tokens = scores.get("tokens") or {}
    
    print(f"\n[Run {run_index}]")
    print(f"  [Metric 1 - Equal Coverage] {_number(coverage)}")
    print(f"  [Metric 2 - BARTScore] {_number(quality)} (closer to zero is better)")
    if not rouge:
        print("  [Metric 3 - ROUGE-2 F1] none")
    else:
        print(
            f"  [Metric 3 - ROUGE-2 F1] {_number(rouge.get('f1'))} "
            f"(precision {_number(rouge.get('precision'))}, recall {_number(rouge.get('recall'))}, "
            f"{rouge.get('n_references')} references)"
        )
    print(f"  [Cost] {_number(seconds)} seconds")
    print(f"  [Tokens] Total: {tokens.get('total_tokens', 0)} (Prompt: {tokens.get('prompt_tokens', 0)}, Completion: {tokens.get('completion_tokens', 0)})")


def _print_run(run_index, result):
    print(f"\n=== run {run_index} ({result['mode']}) ===")
    print(result.get("summary") or "")
    if result["mode"] != "agent":
        return
    print(f"revisions: {result.get('revision')}")
    review = result.get("bias_review") or {}
    print(f"bias review: {review.get('status') or 'not run'}")
    for issue in review.get("issues") or []:
        print(f"- [{issue.get('issue_type')}] {issue.get('explanation')}")


def _save_run(folder, label, run_index, result, scores):
    destination = folder / label / f"{result['topic']}_{result['prompt_id']}_{label}_run{run_index}.json"
    payload = {
        "topic": result["topic"],
        "prompt_id": result["prompt_id"],
        "user_prompt": result["user_prompt"],
        "mode": result["mode"],
        "model": OPENAI_MODEL,
        "summary": result.get("summary") or "",
        "summaries": result.get("summaries") or [],
        "revision": result.get("revision"),
        "bias_review": result.get("bias_review") or {"issues": []},
        "ontology_grounding": result.get("ontology_grounding") or {},
        "evaluation": scores,
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {destination}")


def _save_average(folder, prompt_id, label, average):
    quality = average["quality"]["score"]
    rouge = average["rouge2"]["f1"]
    tokens_avg = average.get("tokens") or {}
    
    print(f"\n=== average over {average['runs']} runs ({prompt_id}, {label}) ===")
    print(f"  [Metric 1 - Equal Coverage] {_number(average['equal_coverage'])}")
    print(f"  [Metric 2 - BARTScore] {_number(quality)} (closer to zero is better)")
    print(f"  [Metric 3 - ROUGE-2 F1] {_number(rouge)}")
    print(f"  [Cost] {_number(average['cost']['seconds'])} seconds")
    print(f"  [Tokens] Total: {_number(tokens_avg.get('total_tokens'))} (Prompt: {_number(tokens_avg.get('prompt_tokens'))}, Completion: {_number(tokens_avg.get('completion_tokens'))})")
    
    destination = folder / label / f"{prompt_id}_{label}_average.json"
    destination.write_text(json.dumps(average, indent=2), encoding="utf-8")
    print(f"saved {destination}")


def _average(rows):
    def mean(values):
        kept = [value for value in values if value is not None]
        if not kept:
            return None
        return sum(kept) / len(kept)

    rouge_rows = [row.get("rouge2") for row in rows]
    return {
        "runs": len(rows),
        "equal_coverage": mean([(row.get("equal_coverage") or {}).get("ec_score") for row in rows]),
        "quality": {"score": mean([(row.get("quality") or {}).get("score") for row in rows])},
        "rouge2": {
            "f1": mean([None if row is None else row.get("f1") for row in rouge_rows]),
            "precision": mean([None if row is None else row.get("precision") for row in rouge_rows]),
            "recall": mean([None if row is None else row.get("recall") for row in rouge_rows]),
        },
        "cost": {"seconds": mean([(row.get("cost") or {}).get("seconds") for row in rows])},
        "tokens": {
            "total_tokens": mean([(row.get("tokens") or {}).get("total_tokens") for row in rows]),
            "prompt_tokens": mean([(row.get("tokens") or {}).get("prompt_tokens") for row in rows]),
            "completion_tokens": mean([(row.get("tokens") or {}).get("completion_tokens") for row in rows]),
        },
    }


def _number(value):
    if value is None:
        return "none"
    return f"{value:.4f}"


def _load_documents(path):
    documents = []
    topic = path.stem
    for row in csv.reader(_read_text(path).splitlines(), delimiter="\t", quotechar='"'):
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) != 2:
            raise ValueError(f"Expected group and tweet text in {path}, got {row!r}")
        group = row[0].strip()
        if group not in GROUPS:
            raise ValueError(f"Unknown group {group!r}; expected one of {GROUPS}")
        documents.append({
            "id": str(len(documents) + 1),
            "text": html.unescape(row[1]).strip(),
            "group": group,
            "topic": topic,
        })
    if not documents:
        raise ValueError(f"No tweets in {path}")
    return documents


def _read_text(path):
    return path.read_bytes().decode("cp1252")


if __name__ == "__main__":
    main()
