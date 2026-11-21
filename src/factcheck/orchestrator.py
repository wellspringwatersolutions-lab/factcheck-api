"""Run a check. The verdict is computed from passages and tools, never from a model's vote."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from uuid import uuid4

from factcheck.aggregate import RULES_VERSION, decide
from factcheck.categories import CATEGORIES, EFFORT_LIMITS, effort_for, normalize_claim
from factcheck.checks import UrlChecker, apply_passage_checks, passage_check_rows
from factcheck.cleanup import focus_query, priority
from factcheck.compute import recompute
from factcheck.credibility import annotate, dedupe_syndicated, dedupe_urls
from factcheck.judge import apply_verified_labels
from factcheck.retrievers import lookup_capital
from factcheck.schemas import (
    DISCLAIMER,
    CheckRow,
    ClaimOut,
    JudgeOut,
    JudgesOut,
    MainstreamCheck,
    MainstreamItem,
    SourceOut,
    UsageOut,
    VerificationOut,
)
from factcheck.textutil import is_fragile, jaccard, normalize_text, overlap
from factcheck.types import Evidence, ExtractedClaim, UsageMeter


def _labels_disagree(left, right) -> bool:
    urls = set(left.stances) | set(right.stances)
    for url in urls:
        a = left.stances.get(url)
        b = right.stances.get(url)
        if a in {"supports", "contradicts"} and b in {"supports", "contradicts"} and a != b:
            return True
    return False


def _cluster(claims: list[ExtractedClaim], threshold: float = 0.34) -> list[list[int]]:
    groups: list[list[int]] = []
    used: set[int] = set()
    for index, claim in enumerate(claims):
        if index in used:
            continue
        group = [index]
        used.add(index)
        for other in range(index + 1, len(claims)):
            if other in used:
                continue
            if jaccard(claim.text, claims[other].text) >= threshold:
                group.append(other)
                used.add(other)
        groups.append(group)
    return groups


def _cache_key(claim: ExtractedClaim) -> str:
    raw = f"{RULES_VERSION}|{claim.claim_category}|{claim.claim_type}|{normalize_text(claim.text)}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _source_out(evidence: Evidence) -> SourceOut:
    stance = {
        "entails": "supports",
        "contradicts": "contradicts",
    }.get(evidence.entailment, "not_relevant")
    return SourceOut(
        url=evidence.url,
        title=evidence.title,
        snippet=(evidence.snippet or evidence.quote)[:500],
        quote=evidence.quote,
        published_at=evidence.published_at,
        grade=evidence.grade,
        stance=stance,
        source_type=evidence.source_type,
        study_type=evidence.study_type,
        origin_id=evidence.origin_id,
        stale=evidence.stale,
        content_sha256=evidence.content_sha256,
        url_status=evidence.url_status,
        quote_match=evidence.quote_match,
        number_match=evidence.number_match,
        date_match=evidence.date_match,
        entailment=evidence.entailment,
        used=evidence.used,
        syndicated_also=list(evidence.syndicated_also),
        provider=evidence.provider or None,
    )


def _judge_out(vote) -> JudgeOut | None:
    if vote is None:
        return None
    return JudgeOut(
        model=vote.model,
        provider=vote.provider,
        role=vote.role,
        ranked_urls=list(vote.ranked_urls),
        filtered_urls=[url for url, stance in vote.stances.items() if stance == "not_relevant"],
        note=vote.reasoning,
    )


def _summary(claims: list[ClaimOut]) -> str:
    if not claims:
        return "No checkable claims were found. " + DISCLAIMER
    facts = [claim for claim in claims if claim.claim_type == "fact" and claim.verdict != "Hypothesis"]
    opinions = [claim for claim in claims if claim.mainstream_check]
    hypotheses = [claim for claim in claims if claim.verdict == "Hypothesis"]
    parts = []
    if facts:
        counts = Counter(claim.verdict for claim in facts)
        bits = ", ".join(f"{counts[name]} {name}" for name in ("Supported", "Contradicted", "Unsupported", "Unverified", "Conflicting") if counts[name])
        parts.append(f"{len(facts)} fact claim{'s' if len(facts) != 1 else ''}: {bits}")
    if opinions:
        counts = Counter(claim.mainstream_check.consensus for claim in opinions)
        bits = ", ".join(f"{count} {name.replace('_', ' ')}" for name, count in counts.items())
        parts.append(f"{len(opinions)} opinion claim{'s' if len(opinions) != 1 else ''} (what mainstream fact-checkers say: {bits})")
    if hypotheses:
        parts.append(f"{len(hypotheses)} hypothesis claim{'s' if len(hypotheses) != 1 else ''}")
    return ". ".join(parts) + ". " + DISCLAIMER


class VerificationService:
    def __init__(
        self,
        settings,
        extractor,
        retriever,
        ranker_a,
        ranker_b,
        ranker_tie,
        mainstream,
        store,
        cache,
        billing,
        url_checker=None,
        capital_fetcher=None,
    ):
        self.settings = settings
        self.extractor = extractor
        self.retriever = retriever
        self.ranker_a = ranker_a
        self.ranker_b = ranker_b
        self.ranker_tie = ranker_tie
        self.mainstream = mainstream
        self.store = store
        self.cache = cache
        self.billing = billing
        self.url_checker = url_checker or UrlChecker(timeout=min(8.0, getattr(settings, "retrieval_timeout_seconds", 8.0)))
        self.capital_fetcher = capital_fetcher
        self._disconnect = None
        self._meter_lock = __import__("threading").Lock()

    def health(self) -> dict:
        soma = any(getattr(provider, "name", "") == "soma" for provider in getattr(getattr(self.extractor, "client", None), "providers", []) or [])
        return {
            "status": "ok",
            "service": self.settings.service_name,
            "llm": "mock" if self.ranker_a.__class__.__name__.startswith("Heuristic") else "live",
            "database": "sqlite" if self.settings.database_url.startswith("sqlite") else "postgres",
            "cache": self.cache.kind,
            "soma_dev_only": soma,
        }

    def verify(self, *, text, claims, client, preclassified=None, charge=True, disconnect_check=None):
        reservation = self.billing.authorize(client) if charge else None
        self._disconnect = disconnect_check
        meter = UsageMeter()
        if hasattr(self.extractor, "meter"):
            self.extractor.meter = meter
        for ranker in (self.ranker_a, self.ranker_b, self.ranker_tie):
            if ranker is not None and hasattr(ranker, "meter"):
                ranker.meter = meter
        try:
            warnings: list[str] = []
            if preclassified is not None:
                extracted = [normalize_claim(item) for item in preclassified]
            else:
                extracted, warnings = self.extractor.extract(text, claims)
                extracted = [normalize_claim(item) for item in extracted]
            if len(extracted) > self.settings.max_output_claims:
                extracted = extracted[: self.settings.max_output_claims]
                warnings.append("claim_limit")
            outputs = self._run(extracted, meter, warnings)
            payload = self._assemble(text, claims, outputs, meter, warnings, client, charge)
            self.store.save(payload, fingerprint=getattr(client, "fingerprint", None), tier=payload.usage.tier, api_key_id=getattr(reservation, "api_key_id", None))
            if charge:
                if disconnect_check and disconnect_check():
                    self.billing.release(reservation)
                else:
                    self.billing.complete(reservation, payload.usage.estimated_cost_usd, payload.verification_id)
            return payload
        except Exception:
            if reservation is not None:
                self.billing.release(reservation)
            raise

    def get(self, verification_id: str) -> VerificationOut | None:
        return self.store.get(verification_id)

    def _disconnected(self) -> bool:
        return bool(self._disconnect and self._disconnect())

    def _not_checked(self, claim: ExtractedClaim, reason: str) -> ClaimOut:
        return ClaimOut(
            text=claim.text,
            claim_type="opinion" if claim.claim_category == "T8" else "fact",
            claim_category=claim.claim_category,
            category_label=CATEGORIES.get(claim.claim_category, ""),
            effort="skipped",
            verdict="Unverified",
            strength_of_support=0.0,
            confidence=0.0,
            reasoning=reason,
            attribution=claim.attribution,
            needs_human_review=False,
        )

    def _run(self, extracted: list[ExtractedClaim], meter: UsageMeter, warnings: list[str]) -> list[ClaimOut]:
        self._cost_mark = getattr(self.retriever, "cost_usd", 0.0)
        results: list[ClaimOut | None] = [None] * len(extracted)
        if self._disconnected():
            return [self._not_checked(claim, "Not checked: the client disconnected before the check finished.") for claim in extracted]
        deadline = time.monotonic() + float(getattr(self.settings, "request_budget_seconds", 40))
        order = sorted(range(len(extracted)), key=lambda index: priority(extracted[index]))
        cap = int(getattr(self.settings, "max_checked_claims", 6))
        selected = order[:cap]
        if len(extracted) > cap:
            warnings.append(f"Checked {cap} of {len(extracted)} claims. The rest were not checked.")
        for index in order[cap:]:
            results[index] = self._not_checked(
                extracted[index],
                "Not checked in this request. The check focused on the highest-stakes claims.",
            )
        jobs = self._jobs(extracted, selected, results, meter, warnings, deadline)
        workers = max(1, int(getattr(self.settings, "check_concurrency", 4)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = []
            for job in jobs:
                if time.monotonic() >= deadline or self._disconnected():
                    break
                futures.append(pool.submit(job))
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"claim_failed:{exc.__class__.__name__}")
        for index, claim in enumerate(extracted):
            if results[index] is None:
                reason = (
                    "Not checked: the client disconnected before the check finished."
                    if self._disconnected()
                    else "Not checked: the time budget ran out before this claim was retrieved."
                )
                results[index] = self._not_checked(claim, reason)
        return [item for item in results if item is not None]

    def _jobs(self, extracted, selected, results, meter, warnings, deadline):
        indexes = list(selected)
        buckets: dict[str, list[int]] = {name: [] for name in ("compute", "opinion", "hypothesis", "light", "standard", "strict")}
        for index in indexes:
            buckets[effort_for(extracted[index])].append(index)
        jobs = []
        for index in buckets["compute"]:
            claim = extracted[index]
            jobs.append(lambda claim=claim, index=index: results.__setitem__(index, self._math_claim(claim)))
        if buckets["opinion"]:
            opinion_indexes = list(buckets["opinion"])
            jobs.append(lambda opinion_indexes=opinion_indexes: self._opinions(extracted, opinion_indexes, results))
        for effort in ("hypothesis", "light", "standard", "strict"):
            subset_indexes = buckets[effort]
            if not subset_indexes:
                continue
            subset = [extracted[index] for index in subset_indexes]
            for cluster in _cluster(subset):
                global_indexes = [subset_indexes[position] for position in cluster]
                jobs.append(
                    lambda global_indexes=global_indexes, effort=effort: self._guarded_cluster(
                        extracted, global_indexes, results, effort, meter, warnings, deadline
                    )
                )
        return jobs

    def _guarded_cluster(self, extracted, indexes, results, effort, meter, warnings, deadline) -> None:
        if time.monotonic() >= deadline or self._disconnected():
            return
        self._evidence_cluster(extracted, indexes, results, effort, meter, warnings)

    def _math_claim(self, claim: ExtractedClaim) -> ClaimOut:
        tool = recompute(claim.text)
        if tool is None:
            decision = decide([], category="T1", tool=None)
            decision.reasoning = "Classified as math, but no arithmetic equality could be recomputed. " + decision.reasoning
        else:
            decision = decide([], category="T1", tool=tool)
        return self._claim_out(claim, "compute", decision, [], tool if tool and tool.get("actual") is not None else tool, None, [])

    def _opinions(self, extracted, indexes, results) -> None:
        claims = [extracted[index] for index in indexes]
        checked = self.mainstream.check(claims) if self.mainstream else [None] * len(claims)
        for index, claim, result in zip(indexes, claims, checked, strict=True):
            mainstream = None
            reasoning = "No mainstream fact-checker lookup was configured."
            strength = 0.0
            if result is not None:
                mainstream = MainstreamCheck(
                    consensus=result.consensus,
                    checks=[
                        MainstreamItem(outlet=item.outlet, rating=item.rating, url=item.url, published_at=item.published_at)
                        for item in result.checks
                    ],
                )
                reasoning = result.reasoning
                strength = result.strength_of_support
            results[index] = ClaimOut(
                text=claim.text,
                claim_type="opinion",
                claim_category="T8",
                category_label=CATEGORIES["T8"],
                effort="opinion",
                verdict=None,
                strength_of_support=strength,
                confidence=strength,
                reasoning=reasoning,
                attribution=claim.attribution,
                needs_human_review=False,
                mainstream_check=mainstream,
            )

    def _evidence_cluster(self, extracted, indexes, results, effort, meter, warnings) -> None:
        claims = [extracted[index] for index in indexes]
        limit = EFFORT_LIMITS[effort]["search"]
        query = focus_query(" ".join(claim.text for claim in claims))
        scientific = any(claim.scientific or claim.claim_category in {"T2", "T3", "T4"} for claim in claims)
        categories = [claim.claim_category for claim in claims]
        before = getattr(self.retriever, "cost_usd", 0.0)
        hits = list(self.retriever.search(query, limit=limit, scientific=scientific, categories=categories))
        with self._meter_lock:
            meter.add_usd(getattr(self.retriever, "cost_usd", 0.0) - before, retriever=True)
        warnings.extend(getattr(self.retriever, "warnings", [])[-3:])
        hits = dedupe_urls(hits)
        hits = self._resolve_all(claims, hits, effort)
        sensitive = any(claim.time_sensitive for claim in claims)
        for hit in hits:
            annotate(hit, time_sensitive=sensitive, stale_after_days=self.settings.stale_after_days)
        hits = dedupe_syndicated(hits)
        for offset, claim in enumerate(claims):
            chosen = self._select(claim, hits)
            tool = None
            if effort != "compute":
                tool = lookup_capital(claim.text, self.capital_fetcher)
            votes = []
            if chosen and self.ranker_a and self.ranker_b and tool is None:
                votes = self._label_concurrent(claim, chosen)
                if len(votes) >= 2 and _labels_disagree(votes[0], votes[1]) and self.ranker_tie:
                    tie = self.ranker_tie.rank([claim], chosen)
                    if tie:
                        votes.append(tie[0])
                chosen = apply_verified_labels(chosen, votes)
            decision = decide(
                chosen,
                category=claim.claim_category,
                hypothesis=claim.hypothesis or effort == "hypothesis",
                high_stakes=claim.claim_category in {"T4", "T7"} or claim.time_sensitive,
                tool=tool,
            )
            checks = passage_check_rows(claim.text, chosen)
            if any(not row["passed"] for row in checks):
                decision.needs_human_review = True
            if tool and tool.get("matches") is False:
                decision.needs_human_review = True
            results[indexes[offset]] = self._claim_out(claim, effort, decision, chosen, tool, votes, checks)

    def _label_concurrent(self, claim: ExtractedClaim, passages: list[Evidence]):
        async def _both():
            first, second = await asyncio.gather(
                asyncio.to_thread(self.ranker_a.rank, [claim], passages),
                asyncio.to_thread(self.ranker_b.rank, [claim], passages),
            )
            return first, second

        first, second = asyncio.run(_both())
        return [vote for vote in ((first[:1] if first else []) + (second[:1] if second else [])) if vote]

    def _resolve_all(self, claims: list[ExtractedClaim], hits: list[Evidence], effort: str) -> list[Evidence]:
        focus = " ".join(claim.text for claim in claims)
        crawl_budget = EFFORT_LIMITS[effort]["crawl"]
        resolved = []
        scholarly = {"openalex", "semantic_scholar", "pubmed", "europepmc"}
        for hit in hits:
            if hit.provider in scholarly and (hit.page_text or hit.snippet):
                body = hit.page_text or hit.snippet
                from factcheck.checks import UrlResolution, content_hash

                resolution = UrlResolution(hit.url, True, 200, hit.url, body, content_hash(body))
            else:
                resolution = self.url_checker.resolve(hit.url, body=hit.snippet)
            if resolution.ok and len(resolution.body) < 80 and crawl_budget > 0 and hasattr(self.retriever, "enrich"):
                extra = self.retriever.enrich(hit)
                if extra:
                    resolution.body = extra
                    resolution.content_sha256 = __import__("factcheck.checks", fromlist=["content_hash"]).content_hash(extra)
                    crawl_budget -= 1
            apply_passage_checks(focus, hit, resolution)
            if not hit.snippet:
                hit.snippet = hit.quote
            resolved.append(hit)
        return resolved

    def _select(self, claim: ExtractedClaim, hits: list[Evidence]) -> list[Evidence]:
        ranked = sorted(hits, key=lambda hit: overlap(claim.text, f"{hit.title} {hit.page_text or hit.snippet}"), reverse=True)
        chosen = []
        for hit in ranked[:5]:
            cloned = Evidence(**{key: getattr(hit, key) for key in hit.__dataclass_fields__})
            apply_passage_checks(claim.text, cloned, type("R", (), {
                "ok": hit.url_status == 200 or hit.url.startswith(("scifact://", "fever://", "heldout://", "corpus://")),
                "status": hit.url_status if hit.url_status is not None else (200 if hit.page_text else None),
                "body": hit.page_text or hit.snippet,
                "content_sha256": hit.content_sha256,
            })())
            chosen.append(cloned)
            if len(chosen) >= 5:
                break
        if not chosen:
            return []
        return chosen

    def _claim_out(self, claim, effort, decision, sources, tool, votes, checks) -> ClaimOut:
        strength = decision.strength_of_support
        judges = None
        if votes:
            judges = JudgesOut(
                a=_judge_out(votes[0]) if len(votes) > 0 else None,
                b=_judge_out(votes[1]) if len(votes) > 1 else None,
                tiebreak=_judge_out(votes[2]) if len(votes) > 2 else None,
            )
        fragile = is_fragile(claim.text)
        review = decision.needs_human_review or (fragile and decision.verdict in {"Unsupported", "Conflicting", "Unverified"})
        return ClaimOut(
            text=claim.text,
            claim_type="opinion" if claim.claim_category == "T8" else "fact",
            claim_category=claim.claim_category,
            category_label=CATEGORIES.get(claim.claim_category, ""),
            effort=effort,
            verdict=decision.verdict,
            strength_of_support=strength,
            confidence=strength,
            reasoning=decision.reasoning,
            attribution=claim.attribution,
            needs_human_review=review,
            hypothesis_status=decision.hypothesis_status,
            deterministic_checks=[CheckRow(**row) for row in checks],
            tool=tool,
            judges=judges,
            sources=[_source_out(item) for item in sources],
        )

    def _assemble(self, text, claims, outputs, meter, warnings, client, charge) -> VerificationOut:
        tier = "eval"
        credits = 0.0
        if charge:
            tier = "paid" if getattr(client, "api_key", None) else "free"
            credits = self.settings.credits_per_check if tier == "paid" else 0.0
        if text:
            compact = " ".join(text.split())
            summary_in = compact[:240] + ("…" if len(compact) > 240 else "")
        else:
            count = len(claims or [])
            summary_in = f"{count} claim{'s' if count != 1 else ''} submitted."
        return VerificationOut(
            verification_id=str(uuid4()),
            created_at=datetime.now(timezone.utc).isoformat(),
            input_summary=summary_in,
            claims=outputs,
            summary=_summary(outputs),
            attribution=DISCLAIMER,
            disclaimer=DISCLAIMER,
            usage=UsageOut(
                input_tokens=meter.input_tokens,
                output_tokens=meter.output_tokens,
                estimated_cost_usd=round(meter.estimated_cost_usd, 6),
                retriever_cost_usd=round(meter.retriever_cost_usd, 6),
                credits_charged=credits,
                tier=tier,
                models=meter.models,
            ),
            upgrade_url=self.settings.resolved_upgrade_url(),
            warnings=list(dict.fromkeys(warnings)),
        )
