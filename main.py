import argparse
import json
import time
from pathlib import Path

from src.config import DATA_PROCESSED, ONTOLOGY_PATH, SUMMARY_SENTENCES, USER_PROMPTS, load_documents
from src.graph.graph import build_baseline_graph, build_graph, initial_state
from src.evaluation import AgentEvaluator
from src.llm.model import get_and_reset_session_tokens

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="DivSumm input_docs/*.txt file, or a JSON list of tweets",
    )
    parser.add_argument("--prompt", choices=sorted(USER_PROMPTS), required=True)
    parser.add_argument("--topic", help="Topic name. Defaults to the topic on the first tweet.")
    parser.add_argument("--k", type=int, default=SUMMARY_SENTENCES, help="Sentences in the summary")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--baseline", action="store_true", help="One-shot LLM, no ontology")
    parser.add_argument("--save", action="store_true", help="Write each run under data/processed")
    parser.add_argument("--evaluate", action="store_true", help="Run DeepEval and ROUGE metrics immediately after execution")
    return parser.parse_args()

def main():
    args = parse_args()
    documents = load_documents(args.data)
    topic = args.topic or documents[0]["topic"]
    if not topic:
        raise SystemExit("Pass --topic or include a topic field on the tweets.")
    prompt = USER_PROMPTS[args.prompt]
    mode = "baseline" if args.baseline else "agent"
    graph = build_baseline_graph() if args.baseline else build_graph()
    if mode == "agent" and not ONTOLOGY_PATH.is_file():
        print(f"Ontology file missing: {ONTOLOGY_PATH}")
    evaluator = AgentEvaluator() if args.evaluate else None
    for run_index in range(1, args.runs + 1):
        state = initial_state(
            documents=documents,
            user_prompt=prompt,
            topic=topic,
            prompt_id=args.prompt,
            k=args.k,
            mode=mode,
        )
        started = time.perf_counter()
        result = graph.invoke(state)
        elapsed = time.perf_counter() - started
        
        # Evaluate the run across 4 metrics if requested
        eval_results = {}
        if args.evaluate:
            print(f"\n[Evaluating run {run_index} across 4 metrics...] ")
            
            eval_results["equal_coverage"] = evaluator.evaluate_equal_coverage(documents, result["summary"])
            
            source_texts = [doc.get("text", "") for doc in documents if isinstance(doc, dict)]
            eval_results["quality"] = evaluator.evaluate_summary_quality(result["summary"], source_texts)
            
            eval_results["rouge2"] = evaluator.evaluate_rouge2(result["summary"], topic)
            
            session_tokens = get_and_reset_session_tokens()
            result["elapsed_seconds"] = elapsed
            eval_results["cost"] = evaluator.evaluate_cost(result, session_tokens)
            
            print(f"  [Metric 1 - Equal Coverage] Score: {eval_results['equal_coverage']:.4f}")
            print(f"  [Metric 2 - Summary Quality] Score: {eval_results['quality']['score']:.4f} (Alignment: {eval_results['quality']['alignment_score']}, Coverage: {eval_results['quality']['coverage_score']})")
            print(f"  [Metric 3 - ROUGE-2] Score: {eval_results['rouge2']:.4f}")
            print(f"  [Metric 4 - Cost] Time: {eval_results['cost']['elapsed_seconds']:.2f}s | Total Tokens: {eval_results['cost']['total_tokens']} (In: {eval_results['cost']['input_tokens']}, Out: {eval_results['cost']['output_tokens']})")
        _print_run(run_index, result, elapsed)
        if args.save:
            _save_run(args, run_index, result, elapsed, eval_results)

def _print_run(run_index, result, elapsed):
    print(f"\n=== run {run_index} ({result['mode']}) {elapsed:.2f}s ===")
    print(result["summary"])
    if result["mode"] == "agent":
        print(f"passed: {result['passed']}  revisions: {result['revision']}  stop: {result.get('stop_reason', '')}")
        for verdict in result.get("claim_evaluations") or []:
            claim = verdict["claim"]
            print(
                f"claim {verdict['status']}: {claim.get('subject')} {claim.get('predicate')} "
                f"{claim.get('object')} [{claim.get('scope')}]"
            )
        if result["issues"]:
            print("issues:")
            for issue in result["issues"]:
                print(f"- [{issue['kind']}] {issue['message']}")
        if result.get("stop_reason") == "rewrite_limit":
            print("Stopped at the rewrite limit with a contradicted claim still in the summary.")
        errors = [item for item in result["query_results"] if item["status"] == "error"]
        for item in errors:
            print(f"query error {item['query_id']}: {item['detail']}")

def _save_run(args, run_index, result, elapsed, eval_results=None):
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    mode = "baseline" if args.baseline else "agent"
    destination = DATA_PROCESSED / f"{result['topic']}_{args.prompt}_{mode}_run{run_index}.json"
    payload = {
        "topic": result["topic"],
        "prompt_id": result["prompt_id"],
        "mode": result["mode"],
        "summary": result["summary"],
        "summaries": result.get("summaries") or [],
        "issues": result["issues"],
        "evaluations": result.get("evaluations") or [],
        "stop_reason": result.get("stop_reason") or "",
        "claims": result.get("claims") or [],
        "mapped_claims": result.get("mapped_claims") or [],
        "claim_evaluations": result.get("claim_evaluations") or [],
        "revision": result["revision"],
        "passed": result["passed"] if result["mode"] == "agent" else None,
        "trace": result["trace"],
        "elapsed_seconds": round(elapsed, 3),
        "query_results": result["query_results"],
    }
    if eval_results:
        payload["evaluation"] = eval_results
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {destination}")

if __name__ == "__main__":
    main()
