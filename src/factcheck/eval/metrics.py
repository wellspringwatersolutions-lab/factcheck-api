"""Scores against outside labels. Model agreement is not the metric."""

from __future__ import annotations

LABELS = ["Supported", "Contradicted", "Unverified"]


def map_gold(label: str) -> str:
    key = (label or "").strip().upper().replace("_", " ").replace("-", " ")
    if key in {"SUPPORT", "SUPPORTS", "SUPPORTED"}:
        return "Supported"
    if key in {"CONTRADICT", "CONTRADICTS", "REFUTE", "REFUTES", "REFUTED"}:
        return "Contradicted"
    if key in {"NOT ENOUGH INFO", "NEI", "NOTENOUGHINFO", "UNVERIFIED"}:
        return "Unverified"
    raise ValueError(f"unknown label: {label}")


def map_verdict(verdict: str | None) -> str:
    """Abstentions are not counted as lucky guesses."""
    if verdict in {None, "Unsupported", "Hypothesis"}:
        return "Unverified"
    if verdict in {"Supported", "Contradicted", "Unverified", "Conflicting"}:
        return verdict
    return "Unverified"


def accuracy(golds: list[str], preds: list[str]) -> float:
    if not golds:
        return 0.0
    return sum(gold == pred for gold, pred in zip(golds, preds, strict=True)) / len(golds)


def macro_f1(golds: list[str], preds: list[str], labels: list[str] | None = None) -> float:
    labels = labels or LABELS
    scores = []
    for label in labels:
        tp = sum(gold == label and pred == label for gold, pred in zip(golds, preds, strict=True))
        fp = sum(gold != label and pred == label for gold, pred in zip(golds, preds, strict=True))
        fn = sum(gold == label and pred != label for gold, pred in zip(golds, preds, strict=True))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores) if scores else 0.0


def confusion(golds: list[str], preds: list[str], labels: list[str] | None = None) -> list[list[int]]:
    labels = labels or LABELS + ["Conflicting"]
    index = {label: position for position, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for gold, pred in zip(golds, preds, strict=True):
        if gold not in index or pred not in index:
            continue
        matrix[index[gold]][index[pred]] += 1
    return matrix


def format_matrix(matrix: list[list[int]], labels: list[str]) -> str:
    header = "| gold \\\\ pred | " + " | ".join(labels) + " |"
    rule = "|" + "|".join(["---"] * (len(labels) + 1)) + "|"
    rows = [header, rule]
    for label, row in zip(labels, matrix, strict=True):
        rows.append("| " + label + " | " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(rows)
