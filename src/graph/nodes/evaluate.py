import time

from rouge_score import rouge_scorer

from src.config import ROOT
from src.evaluation.equal_coverage import equal_coverage

REFERENCES = ROOT / "data" / "raw" / "divsumm"

_rouge = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True)
_bart = None

def evaluate(state):
    documents = state.get("documents") or []
    summary = state.get("summary") or ""
    posts = [doc.get("text") or "" for doc in documents]
    return {
        "evaluation": {
            "equal_coverage": equal_coverage(documents, summary),
            "quality": _quality(summary, posts),
            "rouge2": _rouge2(summary, state.get("topic") or ""),
            "cost": {"seconds": _seconds(state)},
        }
    }

def _quality(summary, posts):
    global _bart
    if _bart is None:
        from src.evaluation.bart_score import BARTScorer

        print("loading BARTScore (facebook/bart-large-cnn)")
        _bart = BARTScorer()
    return {"score": _bart.score_summary(posts, summary)}

def _rouge2(summary, topic):
    if not summary.strip() or not topic:
        return None
    references = []
    for subfolder in ("abstractive",):
        folder = REFERENCES / subfolder
        if not folder.is_dir():
            continue
        for path in folder.glob(f"{topic.strip()}_Annotator*.txt"):
            text = _read(path)
            if text:
                references.append(text)
    if not references:
        return None
    scores = [
        _rouge.score(reference, summary)["rouge2"]
        for reference in references
    ]
    count = len(scores)
    return {
        "f1": sum(score.fmeasure for score in scores) / count,
        "precision": sum(score.precision for score in scores) / count,
        "recall": sum(score.recall for score in scores) / count,
        "n_references": count,
    }

def _read(path):
    try:
        return path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="replace").strip()

def _seconds(state):
    started = state.get("started")
    if not started:
        return 0.0
    return round(time.time() - started, 3)
