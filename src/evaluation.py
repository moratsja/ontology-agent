import json
from pathlib import Path
from rouge_score import rouge_scorer
from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase

class AgentEvaluator:
    def __init__(self, judge_model_name: str = "gpt-4o"):
        self.judge_model_name = judge_model_name
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)

    # Calculates the 1st metric group: Equal Coverage score across demographic groups
    def evaluate_equal_coverage(self, documents: list[dict], summary: str) -> float:
        if not documents:
            return 0.0

        groups = {"White", "Hisp", "AA"}
        group_docs = {g: [] for g in groups}
        
        for doc in documents:
            group = doc.get("group")
            if group in groups:
                group_docs[group].append(doc)

        total_docs = len(documents)
        if total_docs == 0:
            return 0.0

        summary_words = set(summary.lower().split())
        covered_counts = {g: 0 for g in groups}
        total_counts = {g: len(docs) for g, docs in group_docs.items()}
        global_covered = 0

        for g, docs in group_docs.items():
            for doc in docs:
                text = str(doc.get("text", doc)).lower()
                doc_words = set(text.split())
                if doc_words and len(summary_words.intersection(doc_words)) >= 2:
                    covered_counts[g] += 1
                    global_covered += 1

        p_overall = global_covered / total_docs if total_docs > 0 else 0.0
        
        ec_score = 0.0
        for g in groups:
            g_total = total_counts[g]
            if g_total > 0:
                p_group = covered_counts[g] / g_total
                ec_score += abs(p_overall - p_group)

        return float(ec_score)

    # Calculates the 2nd metric group: DeepEval summarization quality score
    def evaluate_summary_quality(self, generated_summary: str, source_documents: list[str]) -> dict:
        try:
            source_text = "\n".join(source_documents)
            test_case = LLMTestCase(input=source_text, actual_output=generated_summary)

            metric = SummarizationMetric(
                threshold=0.5,
                model=self.judge_model_name,
                include_reason=True
            )
            metric.measure(test_case)

            breakdown = getattr(metric, "score_breakdown", None) or {}
            alignment = breakdown.get("Alignment", 0.0) or 0.0
            coverage = breakdown.get("Coverage", 0.0) or 0.0
            final_score = getattr(metric, "score", None)
            if final_score is None:
                final_score = min(alignment, coverage)

            return {
                "score": final_score,
                "alignment_score": alignment,
                "coverage_score": coverage,
                "reason": getattr(metric, "reason", "")
            }
        except Exception as e:
            return {
                "score": 0.0,
                "alignment_score": 0.0,
                "coverage_score": 0.0,
                "reason": f"Evaluation error: {str(e)}"
            }

    # Calculates the 3rd metric group: ROUGE-2 score against reference summaries
    def evaluate_rouge2(self, generated_summary: str, topic: str, dataset_root: Path = Path("data/raw/divsumm")) -> float:
        references = []
        clean_topic = topic.strip()

        for subfolder in ["abstractive", "extractive"]:
            sub_path = dataset_root / subfolder
            if sub_path.is_dir():
                pattern = f"{clean_topic}_Annotator*.txt"
                for ref_file in sub_path.glob(pattern):
                    try:
                        content = ref_file.read_text(encoding="utf-8").strip()
                    except UnicodeDecodeError:
                        content = ref_file.read_text(encoding="latin-1", errors="replace").strip()
                    
                    if content:
                        references.append(content)

        if not references:
            return 0.0

        max_score = 0.0
        for ref in references:
            scores = self.rouge_scorer.score(ref, generated_summary)
            rouge2_recall = scores['rouge2'].recall
            if rouge2_recall > max_score:
                max_score = rouge2_recall
        return max_score

    # Calculates the 4th metric group: cost in terms of elapsed time and token usage
    def evaluate_cost(self, run_payload: dict, session_tokens: dict) -> dict:
        elapsed_seconds = run_payload.get("elapsed_seconds", 0.0)
        
        return {
            "elapsed_seconds": elapsed_seconds,
            "input_tokens": session_tokens.get("input_tokens", 0),
            "output_tokens": session_tokens.get("output_tokens", 0),
            "total_tokens": session_tokens.get("input_tokens", 0) + session_tokens.get("output_tokens", 0)
        }