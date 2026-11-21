"""Run the pipeline on SciFact, a FEVER-format sample, and held-out answers."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from factcheck.checks import UrlChecker
from factcheck.config import Settings
from factcheck.eval.metrics import LABELS, accuracy, confusion, format_matrix, macro_f1, map_gold, map_verdict
from factcheck.judge import HeuristicRanker
from factcheck.mainstream import MainstreamChecker
from factcheck.retrievers import CorpusRetriever
from factcheck.storage import MemoryCache, Store, init_db, make_engine, make_session_factory
from factcheck.types import Evidence, ExtractedClaim, UsageMeter

FIXTURES = Path(__file__).parent / "fixtures"


def load_examples(name: str, sample_size: int, fixtures: Path = FIXTURES, seed: int = 21) -> list[dict]:
    path = fixtures / f"{name}_sample.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("label") or "")].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    mixed: list[dict] = []
    keys = sorted(groups)
    while any(groups.values()) and len(mixed) < sample_size:
        for key in keys:
            if groups[key]:
                mixed.append(groups[key].pop())
                if len(mixed) >= sample_size:
                    break
    return mixed


def passage_label(sources) -> str:
    entails = [source for source in sources if source.used and source.entailment == "entails"]
    contradicts = [source for source in sources if source.used and source.entailment == "contradicts"]
    if entails and contradicts:
        return "Conflicting"
    if entails:
        return "Supported"
    if contradicts:
        return "Contradicted"
    return "Unverified"


def _documents(examples: list[dict]) -> list[Evidence]:
    seen: set[str] = set()
    docs = []
    for example in examples:
        for doc in example.get("documents") or []:
            url = f"{example.get('dataset', 'corpus')}://{doc['id']}"
            if url in seen:
                continue
            seen.add(url)
            docs.append(
                Evidence(
                    url=url,
                    title=doc.get("title") or "",
                    snippet=doc.get("text") or "",
                    provider="corpus",
                    grade="A" if example.get("dataset") == "scifact" else "B",
                    source_type="primary" if example.get("dataset") == "scifact" else "secondary",
                )
            )
    return docs


def _service(examples: list[dict], settings: Settings):
    from factcheck.billing import NoBilling
    from factcheck.judge import LiveRanker
    from factcheck.llm import build_llm
    from factcheck.orchestrator import VerificationService

    engine = make_engine(settings.database_url)
    init_db(engine)
    factory = make_session_factory(engine)
    meter = UsageMeter()
    client = None if settings.llm_mode == "mock" else build_llm(settings)
    if client is None:
        from factcheck.extract import HeuristicExtractor as Extractor

        extractor = Extractor()
        ranker_a = HeuristicRanker("a", "mock-a/heuristic", meter)
        ranker_b = HeuristicRanker("b", "mock-b/heuristic", meter)
        ranker_tie = HeuristicRanker("tie", "mock-tie/heuristic", meter)
    else:
        from factcheck.extract import LiveExtractor

        extractor = LiveExtractor(client, settings.extract_model, meter, settings)
        ranker_a = LiveRanker(client, "a", settings.judge_a_model, meter, settings)
        ranker_b = LiveRanker(client, "b", settings.judge_b_model, meter, settings)
        ranker_tie = LiveRanker(client, "tie", settings.tiebreak_model, meter, settings)
    return VerificationService(
        settings=settings,
        extractor=extractor,
        retriever=CorpusRetriever(_documents(examples)),
        ranker_a=ranker_a,
        ranker_b=ranker_b,
        ranker_tie=ranker_tie,
        mainstream=MainstreamChecker(),
        store=Store(factory),
        cache=MemoryCache(),
        billing=NoBilling(),
        url_checker=UrlChecker(),
        capital_fetcher=None,
    )


def run_dataset(name: str, examples: list[dict], settings: Settings) -> dict:
    service = _service(examples, settings)
    golds = []
    verdicts = []
    passages = []
    latencies = []
    costs = []
    category = {"scifact": "T4", "fever": "T5", "heldout": "T5"}.get(name, "T5")
    for example in examples:
        claim_category = "T1" if example.get("tool") == "calculator" else category
        claim = ExtractedClaim(
            text=example["claim"],
            claim_type="fact",
            claim_category=claim_category,
            scientific=name == "scifact",
            hypothesis=False,
        )
        started = time.perf_counter()
        result = service.verify(text=None, claims=None, client=None, preclassified=[claim], charge=False)
        latencies.append(time.perf_counter() - started)
        checked = result.claims[0]
        golds.append(map_gold(example["label"]))
        verdicts.append(map_verdict(checked.verdict))
        if checked.tool and checked.verdict in {"Supported", "Contradicted"}:
            passages.append(checked.verdict)
        else:
            passages.append(passage_label(checked.sources))
        costs.append(result.usage.estimated_cost_usd)
    labels = LABELS + ["Conflicting"]
    return {
        "name": name,
        "n": len(examples),
        "accuracy": accuracy(golds, verdicts),
        "macro_f1": macro_f1(golds, verdicts),
        "entailment_accuracy": accuracy(golds, [map_verdict(item) for item in passages]),
        "entailment_macro_f1": macro_f1(golds, [map_verdict(item) for item in passages]),
        "confusion": confusion(golds, verdicts, labels),
        "labels": labels,
        "avg_latency_s": sum(latencies) / len(latencies) if latencies else 0.0,
        "avg_cost_usd": sum(costs) / len(costs) if costs else 0.0,
    }


def render_report(rows: list[dict], *, sample_size: int, mode: str) -> str:
    lines = [
        "# Evaluation",
        "",
        f"Mode: {mode}",
        f"Sample size requested: {sample_size}",
        "",
        "Verdicts are scored against the dataset labels, not against another model.",
        "Unsupported and Hypothesis are counted as Unverified so an abstention is not a lucky guess.",
        "Entailment accuracy asks whether a retrieved passage lines up with the gold label.",
        "Dry-run uses the offline passage labeler and the dataset corpus. It does not call a paid model.",
        "A live run, without --dry-run, uses the configured model client for extraction and passage labels.",
        "",
        "Grounding reduces errors. It cannot prove that no better evidence exists.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['name']}",
                "",
                f"- n: {row['n']}",
                f"- verdict accuracy: {row['accuracy']:.3f}",
                f"- verdict macro-F1: {row['macro_f1']:.3f}",
                f"- passage entailment accuracy: {row['entailment_accuracy']:.3f}",
                f"- passage entailment macro-F1: {row['entailment_macro_f1']:.3f}",
                f"- avg latency: {row['avg_latency_s']:.3f}s",
                f"- avg estimated cost: ${row['avg_cost_usd']:.6f}",
                "",
                format_matrix(row["confusion"], row["labels"]),
                "",
            ]
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate fact checks against labeled claims")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="reports/dry-run.md")
    parser.add_argument("--datasets", default="scifact,fever,heldout")
    args = parser.parse_args(argv)
    settings = Settings(
        _env_file=None,
        database_url="sqlite://",
        llm_mode="mock" if args.dry_run or not _live_keys() else "auto",
        cache_backend="memory",
    )
    mode = "dry-run" if settings.llm_mode == "mock" else "live"
    rows = []
    for name in [part.strip() for part in args.datasets.split(",") if part.strip()]:
        examples = load_examples(name, args.sample_size)
        rows.append(run_dataset(name, examples, settings))
    report = render_report(rows, sample_size=args.sample_size, mode=mode)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {output}")
    return 0


def _live_keys() -> bool:
    settings = Settings()
    return bool(settings.chutes_api_key or settings.openai_compat_api_key or (settings.soma_api_key and settings.soma_enabled))
