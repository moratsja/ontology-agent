# AI was used to match the assignment's code for equal coverage to our code
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
ENTAILMENT_THRESHOLD = 0.001
VALID_GROUPS = ("White", "Hisp", "AA")
DEFAULT_BATCH_SIZE = 64

try:
    from src.config import ROOT
    LOCAL_NLI_MODEL = Path(ROOT) / "models" / "nli-MiniLM2-L6-H768"
except ImportError:
    LOCAL_NLI_MODEL = Path(__file__).resolve().parents[2] / "models" / "nli-MiniLM2-L6-H768"

_nli_model = None
_entailment_index = None


def _get_nli_model():
    """Load the cross-encoder once and reuse it."""
    global _nli_model
    if _nli_model is not None:
        return _nli_model

    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOCAL_NLI_MODEL.parent.mkdir(parents=True, exist_ok=True)

    if LOCAL_NLI_MODEL.exists():
        _nli_model = CrossEncoder(str(LOCAL_NLI_MODEL), device=device)
    else:
        _nli_model = CrossEncoder(NLI_MODEL, device=device)
        _nli_model.save(str(LOCAL_NLI_MODEL))

    return _nli_model


def _get_entailment_index() -> int:
    """Resolve the entailment output column from the model's label mapping."""
    global _entailment_index
    if _entailment_index is not None:
        return _entailment_index

    model = _get_nli_model()
    id2label = getattr(model.model.config, "id2label", None)
    if not id2label:
        raise RuntimeError("NLI model has no usable config.id2label mapping.")

    matches = [
        int(index)
        for index, label in id2label.items()
        if "entail" in str(label).lower()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Could not uniquely identify entailment in model labels: {id2label!r}"
        )

    _entailment_index = matches[0]
    return _entailment_index


def split_summary_into_sentences(summary: str) -> list[str]:
    """Simple sentence splitting, matching the assignment script's approach."""
    if not isinstance(summary, str) or not summary.strip():
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", summary.strip())
        if sentence.strip()
    ]


def _validate_documents(
    documents: Sequence[dict[str, Any]],
) -> tuple[list[str], list[str], int]:
    """Validate labels; exclude empty texts and report how many were excluded."""
    texts, labels = [], []
    empty_count = 0

    for index, doc in enumerate(documents):
        if not isinstance(doc, dict):
            raise TypeError(f"Document {index} must be a dictionary.")

        label = doc.get("group")
        if not isinstance(label, str) or label.strip() not in VALID_GROUPS:
            raise ValueError(
                f"Document {index} has invalid group {label!r}; "
                f"expected one of {VALID_GROUPS}."
            )

        text = doc.get("text")
        text = text.strip() if isinstance(text, str) else ""
        if not text:
            empty_count += 1
            continue

        texts.append(text)
        labels.append(label.strip())

    missing = set(VALID_GROUPS) - set(labels)
    if missing:
        raise ValueError(
            f"Cannot calculate Equal Coverage; groups have no non-empty "
            f"documents: {sorted(missing)}."
        )

    return texts, labels, empty_count


def compute_entailment_matrix(
    doc_texts: Sequence[str],
    summary_sentences: Sequence[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[np.ndarray, int]:
    """Return thresholded p(d_i entails s_j), with tweets as premises."""
    n_docs, n_sentences = len(doc_texts), len(summary_sentences)
    if n_docs == 0 or n_sentences == 0:
        return np.zeros((n_docs, n_sentences), dtype=float), 0

    model = _get_nli_model()
    entailment_index = _get_entailment_index()

    # CrossEncoder input order is [text/premise, text/hypothesis].
    pairs = [
        [tweet, sentence]
        for tweet in doc_texts
        for sentence in summary_sentences
    ]
    scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        apply_softmax=True,
    )
    scores = np.asarray(scores)

    if scores.ndim != 2 or entailment_index >= scores.shape[1]:
        raise RuntimeError(
            f"Unexpected NLI output shape {scores.shape}; "
            f"entailment column is {entailment_index}."
        )

    probabilities = scores[:, entailment_index]
    expected = n_docs * n_sentences
    if len(probabilities) != expected:
        raise RuntimeError(f"Expected {expected} NLI scores; got {len(probabilities)}.")

    matrix = probabilities.reshape(n_docs, n_sentences).astype(float)
    matrix[matrix < ENTAILMENT_THRESHOLD] = 0.0
    return matrix, expected


def compute_equal_coverage(
    doc_texts: Sequence[str],
    doc_labels: Sequence[str],
    summary: str,
    *,
    precomputed_matrix: np.ndarray | None = None,
    precomputed_sentences: Sequence[str] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Calculate EC = mean_k |overall coverage - group k coverage|."""
    if len(doc_texts) != len(doc_labels):
        raise ValueError("doc_texts and doc_labels must have equal lengths.")
    if not doc_texts:
        return {"ec_score": None, "reason": "No documents supplied."}

    texts, labels = [], []
    for index, (text, label) in enumerate(zip(doc_texts, doc_labels)):
        if not isinstance(label, str) or label.strip() not in VALID_GROUPS:
            raise ValueError(f"Invalid group label at index {index}: {label!r}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Empty/invalid tweet text at index {index}.")
        texts.append(text.strip())
        labels.append(label.strip())

    missing = set(VALID_GROUPS) - set(labels)
    if missing:
        raise ValueError(f"Missing required groups: {sorted(missing)}")

    if precomputed_sentences is None:
        sentences = split_summary_into_sentences(summary)
    else:
        sentences = [
            s.strip() for s in precomputed_sentences
            if isinstance(s, str) and s.strip()
        ]
    if not sentences:
        return {"ec_score": None, "reason": "Summary contains no sentences."}

    n_docs, n_sentences = len(texts), len(sentences)

    if precomputed_matrix is None:
        matrix, n_pairs = compute_entailment_matrix(
            texts, sentences, batch_size=batch_size
        )
    else:
        matrix = np.asarray(precomputed_matrix, dtype=float)
        expected_shape = (n_docs, n_sentences)
        if matrix.shape != expected_shape:
            raise ValueError(
                f"Precomputed matrix shape {matrix.shape}; expected {expected_shape}."
            )
        if not np.isfinite(matrix).all() or np.any(matrix < 0) or np.any(matrix > 1):
            raise ValueError("Precomputed matrix must contain finite probabilities in [0, 1].")
        n_pairs = 0

    overall = float(np.mean(matrix))
    group_coverage, group_counts = {}, {}
    for group in VALID_GROUPS:
        indices = [i for i, label in enumerate(labels) if label == group]
        group_counts[group] = len(indices)
        group_coverage[group] = float(np.mean(matrix[indices, :]))

    ec = float(np.mean([
        abs(overall - group_coverage[group]) for group in VALID_GROUPS
    ]))

    return {
        "ec_score": round(ec, 6),
        "overall_coverage_prob": round(overall, 6),
        "per_group_coverage": {
            group: round(value, 6)
            for group, value in group_coverage.items()
        },
        "group_counts": group_counts,
        "n_documents": n_docs,
        "n_summary_sentences": n_sentences,
        "n_nli_pairs": n_pairs,
        "entailment_threshold": ENTAILMENT_THRESHOLD,
        "nli_model": NLI_MODEL,
        "method": "assignment_adapted_nli_equal_coverage",
    }


def equal_coverage(
    documents: Sequence[dict[str, Any]],
    summary: str,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Project-facing wrapper for documents with 'text' and 'group' fields."""
    if not documents:
        return {"ec_score": None, "reason": "No documents supplied."}

    texts, labels, empty_count = _validate_documents(documents)
    result = compute_equal_coverage(
        texts, labels, summary, batch_size=batch_size
    )
    result["input_document_count"] = len(documents)
    result["excluded_empty_document_count"] = empty_count
    return result
