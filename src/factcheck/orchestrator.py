"""Run a check. The verdict is computed from passages and tools, never from a model's vote."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from uuid import uuid4

from factcheck.aggregate import RULES_VERSION, decide
from factcheck.categories import CATEGORIES, EFFORT_LIMITS, effort_for, is_math, normalize_claim
from factcheck.checks import UrlChecker, apply_passage_checks, passage_check_rows
from factcheck.cleanup import focus_query, is_bare_question, peel_leading_question, priority, split_dialogue, strip_labels
from factcheck.extract import fallback_points, heuristic_extract, select_key_points, standalone_claim
from factcheck.queries import heuristic_queries, news_window, wikipedia_event_query
from factcheck.request_ctx import (
    add_phase,
    begin_request,
    bind_context,
    current_request,
    drop_counts,
    end_request,
    phase_seconds,
    retrieval_calls,
    set_phase,
)
from factcheck.checks import apply_debunk_labels, entailment, is_debunk
from factcheck.retain import (
    blocked_site,
    drop_invented_negative,
    entity_mismatch,
    is_fact_checker,
    is_official,
    person_names,
    screen_hits,
)
from factcheck.compute import recompute
from factcheck.credibility import annotate, dedupe_syndicated, dedupe_urls, drop_circular
from factcheck.judge import apply_verified_labels
from factcheck.retrievers import lookup_capital
from factcheck.questions import DraftError
from factcheck.schemas import (
    DISCLAIMER,
    CheckRow,
    ClaimOut,
    DraftAnswer,
    JudgeOut,
    JudgesOut,
    MainstreamCheck,
    MainstreamItem,
    SourceOut,
    UsageOut,
    VerificationOut,
)
from factcheck.synopsis import build_synopsis
from factcheck.textutil import is_fragile, jaccard, normalize_text, overlap, split_sentences
from factcheck.types import Evidence, ExtractedClaim, UsageMeter

logger = logging.getLogger("factcheck")


def _same_person(claim: str, quote: str) -> bool:
    names = person_names(claim)
    if not names:
        return True
    haystack = (quote or "").lower()
    return any(name.lower() in haystack for name in names)


def _encyclopedia_source(source) -> bool:
    if source.source_type == "encyclopedia":
        return True
    url = (source.url or "").lower()
    return "wikipedia.org" in url or "britannica.com" in url


def _share_factchecks(claims: list[ClaimOut]) -> None:
    """A fact-checker or encyclopedia quote found for one claim can back a related claim."""
    pool = []
    for claim in claims:
        for source in claim.sources:
            text = source.quote or source.snippet or ""
            if is_fact_checker(source.url, source.source_type) or is_debunk(text) or _encyclopedia_source(source):
                pool.append(source)
    for claim in claims:
        if claim.claim_type != "fact" or claim.verdict in {"Supported", "Contradicted"}:
            continue
        have = {source.url for source in claim.sources}
        extras = []
        for source in pool:
            if source.url in have:
                continue
            quoted = source.quote or source.snippet or ""
            if overlap(claim.text, quoted) < 0.2:
                continue
            if _encyclopedia_source(source) and not _same_person(claim.text, quoted):
                continue
            if _encyclopedia_source(source) and not is_debunk(quoted) and entailment(claim.text, quoted) != "entails":
                continue
            extras.append(source)
            have.add(source.url)
        if not extras:
            continue
        evidence = []
        for source in [*claim.sources, *extras]:
            text = source.quote or source.snippet or ""
            evidence.append(
                Evidence(
                    url=source.url,
                    title=source.title,
                    snippet=source.snippet,
                    quote=source.quote or text[:400],
                    page_text=text,
                    grade=source.grade or "B",
                    source_type=source.source_type or "fact_checker",
                    entailment="entails" if is_debunk(text) or _encyclopedia_source(source) else (source.entailment or "neutral"),
                    used=True,
                    origin_id=source.origin_id or source.url,
                )
            )
        evidence = apply_debunk_labels(claim.text, evidence)
        decision = decide(evidence, category=claim.claim_category or "T6", claim_text=claim.text)
        if decision.verdict not in {"Supported", "likely_true"}:
            continue
        claim.verdict = decision.verdict
        claim.reasoning = decision.reasoning
        claim.strength_of_support = decision.strength_of_support
        claim.confidence = decision.strength_of_support
        claim.sources.extend(extras)


def _clean_conjunctions(sentence: str) -> list[str]:
    """Split a model point only when both sides are already complete sentences."""
    parts = [part.strip() for part in re.split(r"\s+and\s+", sentence or "", flags=re.I) if part.strip()]
    if len(parts) != 2:
        return [sentence]
    finished = []
    for part in parts:
        if part[-1:] not in ".!?":
            part += "."
        if part[0].islower():
            part = part[0].upper() + part[1:]
        if not standalone_claim(part):
            return [sentence]
        finished.append(part)
    return finished


def _split_key_points(claims: list) -> list:
    """Split model points on a clean 'and' only. Fallback sentences are not split."""
    pieces = []
    for claim in claims:
        if not standalone_claim(claim.text) and not is_math(claim.text):
            continue
        for part in _clean_conjunctions(claim.text):
            if part == claim.text:
                pieces.append(claim)
                continue
            pieces.append(
                normalize_claim(
                    ExtractedClaim(
                        text=part,
                        claim_type=claim.claim_type,
                        claim_category=claim.claim_category,
                        time_sensitive=claim.time_sensitive,
                        scientific=claim.scientific,
                        attribution=claim.attribution,
                        fragile=claim.fragile,
                        hypothesis=claim.hypothesis,
                        span=claim.span or claim.text,
                    )
                )
            )
    return select_key_points(pieces, 6)


def _call_with_timeout(func, seconds: float, *args):
    """Run one call and stop waiting after `seconds`. The worker is a daemon."""
    box: dict = {}

    def run() -> None:
        try:
            box["value"] = func(*args)
        except Exception as exc:  # noqa: BLE001 - re-raised on the caller thread
            box["error"] = exc

    worker = threading.Thread(target=bind_context(run), daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise TimeoutError(f"extraction exceeded {seconds:.0f}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def request_budget(settings, text: str | None) -> float:
    """Long pasted answers get 75 seconds. Everything else stays near 45 so the review can run."""
    budget = float(getattr(settings, "request_budget_seconds", 45))
    if text and len(text) > 3500:
        budget = max(budget, float(getattr(settings, "long_text_budget_seconds", 75)))
    return budget


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
        bits = ", ".join(
            f"{counts[name]} {name}"
            for name in ("Supported", "likely_true", "Contradicted", "Unsupported", "Unverified", "Conflicting")
            if counts[name]
        )
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
        claimbuster=None,
        drafter=None,
        synopsizer=None,
        reviewer=None,
        query_writer=None,
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
        self.claimbuster = claimbuster
        self.drafter = drafter
        self.synopsizer = synopsizer
        self.reviewer = reviewer
        self.query_writer = query_writer
        self._draft_text = ""
        self._disconnect = None
        self._meter_lock = threading.Lock()

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

    def verify(self, *, text, claims, client, preclassified=None, charge=True, disconnect_check=None, question=None, started_at=None, budget=None, on_progress=None):
        reservation = self.billing.authorize(client) if charge else None
        meter = UsageMeter()
        state, token = begin_request(meter)
        if hasattr(self.retriever, "bind_budget"):
            self.retriever.bind_budget(state)
        state.disconnect = disconnect_check
        state.on_progress = on_progress
        started = started_at if started_at is not None else time.monotonic()
        budget = float(budget) if budget is not None else request_budget(self.settings, text)
        deadline = started + budget
        state.deadline = deadline
        try:
            warnings: list[str] = []
            mode = "text"
            asked = question
            draft_answer = None
            state.draft_text = ""
            check_text = text
            check_claims = claims
            if preclassified is None and not claims and text and not question and not is_bare_question(text):
                lead, body = peel_leading_question(text)
                if lead:
                    asked = lead
                    check_text = body
            if preclassified is None and not claims and (question or is_bare_question(text)):
                mode = "question"
                asked = (question or text or "").strip()
                model_name, drafted, stance = self._draft(asked)
                draft_answer = DraftAnswer(model=model_name, text=drafted, stance=stance)
                state.draft_text = drafted
                check_text = drafted
                check_claims = None
            state.question = asked or ""
            t_extract = time.monotonic()
            if preclassified is not None:
                extracted = [normalize_claim(item) for item in preclassified]
            else:
                try:
                    extracted, warnings = _call_with_timeout(
                        self.extractor.extract, 22.0, check_text, check_claims
                    )
                except Exception as exc:  # noqa: BLE001 - extraction must not consume the retrieval budget
                    snippet = (check_text or "")[:240]
                    logger.warning("extraction_fallback %s: %s snippet=%r", exc.__class__.__name__, exc, snippet)
                    extracted = []
                    warnings = [f"extraction_fallback:{exc.__class__.__name__}"]
                extracted = [normalize_claim(item) for item in extracted]
                if check_claims is None and (not extracted or any(item.startswith("extraction_fallback") for item in warnings)):
                    retried: list = []
                    if getattr(self.extractor, "retry_keypoints", None):
                        try:
                            retried = _call_with_timeout(self.extractor.retry_keypoints, 8.0, check_text) or []
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("extraction retry %s snippet=%r", exc.__class__.__name__, (check_text or "")[:240])
                            retried = []
                    if retried:
                        extracted = [normalize_claim(item) for item in retried]
                        warnings = ["extraction_retry:deepseek"]
                    else:
                        extracted = fallback_points(check_text)
                        if not any(item.startswith("extraction_fallback") for item in warnings):
                            warnings.append("extraction_fallback:TimeoutError")
            for item in extracted:
                item.text = strip_labels(item.text)
            if mode == "question":
                extracted = drop_invented_negative(extracted, asked)
                have = {" ".join(item.text.lower().split()) for item in extracted}
                for sentence in split_sentences(check_text or ""):
                    if len(extracted) >= 3:
                        break
                    sentence = strip_labels(sentence).strip()
                    if sentence and sentence[-1] not in ".!?":
                        sentence += "."
                    if not sentence or " ".join(sentence.lower().split()) in have:
                        continue
                    if drop_invented_negative([type("C", (), {"text": sentence})()], asked) == []:
                        continue
                    if not standalone_claim(sentence):
                        continue
                    extracted.append(normalize_claim(ExtractedClaim(text=sentence, claim_type="fact", span=sentence)))
                    have.add(" ".join(sentence.lower().split()))
            if check_claims is None and preclassified is None and not any(item.startswith("extraction_fallback") for item in warnings):
                extracted = _split_key_points(extracted)
            elif check_claims is None and preclassified is None:
                extracted = select_key_points(extracted, 6)
            set_phase("extract", time.monotonic() - t_extract)
            # Retrieval keeps 25s and the two-pass review keeps 16s on a real request.
            # A shorter budget, including a zero budget, is left alone so the request can stop.
            if budget >= 41 and deadline - time.monotonic() < 41:
                deadline = time.monotonic() + 41
                state.deadline = deadline
            if len(extracted) > self.settings.max_output_claims:
                extracted = extracted[: self.settings.max_output_claims]
                warnings.append("claim_limit")
            outputs = self._run(extracted, meter, warnings, deadline - 16)
            meter.retrieval_calls = retrieval_calls()
            _share_factchecks(outputs)
            payload = self._assemble(
                asked or text,
                claims,
                outputs,
                meter,
                warnings,
                client,
                charge,
                mode=mode,
                draft_answer=draft_answer,
                question=asked,
                deadline=deadline,
            )
            if deadline - time.monotonic() < 12:
                deadline = time.monotonic() + 12
                state.deadline = deadline
            t_review = time.monotonic()
            if self.reviewer is not None:
                self.reviewer.meter = meter
                from factcheck.review import run_review

                run_review(payload, reviewer=self.reviewer, deadline=deadline, meter=meter)
            set_phase("review", time.monotonic() - t_review)
            total = time.monotonic() - started
            logger.warning(
                "check phases extract=%.2fs retrieve=%.2fs judge=%.2fs review=%.2fs total=%.2fs budget=%.0fs chars=%d",
                phase_seconds("extract"),
                phase_seconds("retrieve"),
                phase_seconds("judge"),
                phase_seconds("review"),
                total,
                budget,
                len(text or ""),
            )
            from factcheck.reconcile import reconcile_answer

            reconcile_answer(payload)
            if payload.usage.tier == "owner":
                logger.info("verification %s tier=owner ip=%s", payload.verification_id, getattr(client, "ip", ""))
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
        finally:
            end_request(token)

    def get(self, verification_id: str) -> VerificationOut | None:
        return self.store.get(verification_id)

    def _disconnected(self) -> bool:
        state = current_request()
        check = state.disconnect if state is not None else None
        return bool(check and check())

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
            checked_at=datetime.now(timezone.utc).isoformat(),
        )

    def _dated_wikipedia(self, text: str) -> list:
        query = wikipedia_event_query(text)
        if not query:
            return []
        wiki = getattr(getattr(self.retriever, "references", None), "wikipedia", None)
        if wiki is None or not hasattr(wiki, "search_dated_event"):
            logger.warning(
                "wikipedia dated event dropped claim=%r query=%s reason=no_wikipedia_retriever",
                (text or "")[:160],
                query,
            )
            return []
        return list(wiki.search_dated_event(text) or [])

    def _search_queries(self, text: str) -> list[str]:
        writer = getattr(self, "query_writer", None)
        if writer is not None:
            try:
                written = writer.write(text)
            except Exception as exc:  # noqa: BLE001 - a query failure must not skip retrieval
                logger.warning("query writer failed %s", exc.__class__.__name__)
                written = []
            if written:
                return written[:2]
        written = heuristic_queries(text)[:2]
        return written or [focus_query(text)]

    def _draft(self, question: str) -> tuple[str, str, str]:
        from factcheck.questions import parse_stance

        if self.drafter is None:
            raise DraftError("No language model is configured, so a question cannot be drafted.")
        try:
            drafted = self.drafter(question)
        except DraftError:
            raise
        except Exception as exc:  # noqa: BLE001 - the caller gets one clean error, not a stack trace
            raise DraftError("The draft answer could not be written, so nothing was checked.") from exc
        if not isinstance(drafted, tuple) or len(drafted) not in {2, 3}:
            raise DraftError("The draft answer could not be written, so nothing was checked.")
        model_name, text = drafted[0], drafted[1]
        stance = drafted[2] if len(drafted) == 3 else parse_stance("", text)
        text = (text or "").strip()
        if not text:
            raise DraftError("The draft answer was empty, so nothing was checked.")
        return str(model_name or "unknown"), text, parse_stance(stance, text)

    def _run(self, extracted: list[ExtractedClaim], meter: UsageMeter, warnings: list[str], deadline: float | None = None) -> list[ClaimOut]:
        self._cost_mark = getattr(self.retriever, "cost_usd", 0.0)
        results: list[ClaimOut | None] = [None] * len(extracted)
        if self._disconnected():
            return [self._not_checked(claim, "Not checked: the client disconnected before the check finished.") for claim in extracted]
        if deadline is None:
            deadline = time.monotonic() + float(getattr(self.settings, "request_budget_seconds", 40))
        worthiness: dict[str, float] = {}
        if getattr(self, "claimbuster", None):
            try:
                worthiness = self.claimbuster.scores([claim.text for claim in extracted]) or {}
            except Exception:  # noqa: BLE001 - scoring is optional and must not stall the check
                worthiness = {}
        order = sorted(range(len(extracted)), key=lambda index: priority(extracted[index], worthiness.get(extracted[index].text, 0.0)))
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
                futures.append(pool.submit(bind_context(job)))
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
        callback = current_request().on_progress if current_request() is not None else None
        if callback is not None:
            callback(sum(1 for item in results if item is not None), len(results))
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
                checked_at=datetime.now(timezone.utc).isoformat(),
            )

    def _evidence_cluster(self, extracted, indexes, results, effort, meter, warnings) -> None:
        claims = [extracted[index] for index in indexes]
        limit = EFFORT_LIMITS[effort]["search"]
        scientific = any(claim.scientific or claim.claim_category in {"T2", "T3", "T4"} for claim in claims)
        categories = [claim.claim_category for claim in claims]
        t_retrieve = time.monotonic()
        hits = []
        seen_urls: set[str] = set()
        for claim in claims:
            queries = self._search_queries(claim.text)[:2]
            window = news_window(claim.text)
            found = list(
                self.retriever.search(
                    queries[0],
                    limit=min(limit, 10),
                    scientific=scientific,
                    categories=categories,
                    extra_queries=queries[1:2],
                    window=window,
                    broad=True,
                    budget=current_request(),
                )
            )
            found.extend(self._dated_wikipedia(claim.text))
            kept, reasons = screen_hits(claim.text, found, window, context=(current_request().question if current_request() else ""))
            logger.warning(
                "retrieval claim=%r queries=%s returned=%d kept=%d drops=%s",
                claim.text[:160],
                queries,
                len(found),
                len(kept),
                reasons,
            )
            for hit in kept:
                key = (hit.url or "").split("#")[0].rstrip("/")
                if not key or key in seen_urls:
                    continue
                seen_urls.add(key)
                hits.append(hit)
        add_phase("retrieve", time.monotonic() - t_retrieve)
        meter.retrieval_calls = retrieval_calls()
        warnings.extend(getattr(self.retriever, "warnings", [])[-3:])
        hits = dedupe_urls(hits)
        draft = current_request().draft_text if current_request() is not None else ""
        hits = drop_circular(hits, [claim.text for claim in claims], draft)
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
                t_judge = time.monotonic()
                votes = self._label_concurrent(claim, chosen)
                add_phase("judge", time.monotonic() - t_judge)
                if len(votes) >= 2 and _labels_disagree(votes[0], votes[1]) and self.ranker_tie:
                    tie = self.ranker_tie.rank([claim], chosen)
                    if tie:
                        votes.append(tie[0])
                chosen = apply_verified_labels(chosen, votes)
            chosen = apply_debunk_labels(claim.text, chosen)
            context = current_request().question if current_request() else ""
            chosen = [
                hit
                for hit in chosen
                if not blocked_site(hit.url)
                and not entity_mismatch(
                    claim.text,
                    " ".join(part for part in (hit.snippet, hit.page_text, hit.quote) if part),
                    context,
                    hit.url,
                    hit.title,
                )
            ]
            decision = decide(
                chosen,
                category=claim.claim_category,
                hypothesis=claim.hypothesis or effort == "hypothesis",
                high_stakes=claim.claim_category in {"T4", "T7"} or claim.time_sensitive,
                tool=tool,
                claim_text=claim.text,
            )
            checks = passage_check_rows(claim.text, chosen)
            if any(not row["passed"] for row in checks):
                decision.needs_human_review = True
            if tool and tool.get("matches") is False:
                decision.needs_human_review = True
            results[indexes[offset]] = self._claim_out(claim, effort, decision, chosen, tool, votes, checks)

    def _label_concurrent(self, claim: ExtractedClaim, passages: list[Evidence]):
        rank_a = bind_context(self.ranker_a.rank, [claim], passages)
        rank_b = bind_context(self.ranker_b.rank, [claim], passages)

        async def _both():
            first, second = await asyncio.gather(asyncio.to_thread(rank_a), asyncio.to_thread(rank_b))
            return first, second

        first, second = asyncio.run(_both())
        return [vote for vote in ((first[:1] if first else []) + (second[:1] if second else [])) if vote]

    def _resolve_all(self, claims: list[ExtractedClaim], hits: list[Evidence], effort: str) -> list[Evidence]:
        focus = " ".join(claim.text for claim in claims)
        crawl_budget = EFFORT_LIMITS[effort]["crawl"]
        resolved = []
        scholarly = {"openalex", "semantic_scholar", "pubmed", "europepmc", "wikidata", "loc", "google_factcheck", "wikipedia"}
        state = current_request()
        deadline = state.deadline if state is not None else None
        fetched = 0
        for hit in hits:
            if deadline is not None and time.monotonic() > deadline - 12:
                from factcheck.checks import UrlResolution, content_hash

                snippet = hit.page_text or hit.snippet or ""
                resolution = UrlResolution(hit.url, bool(snippet), 200 if snippet else None, hit.url, snippet, content_hash(snippet))
                apply_passage_checks(focus, hit, resolution)
                resolved.append(hit)
                continue
            if fetched >= 4 and hit.provider not in scholarly:
                from factcheck.checks import UrlResolution, content_hash

                snippet = hit.page_text or hit.snippet or ""
                resolution = UrlResolution(hit.url, bool(snippet), 200 if snippet else None, hit.url, snippet, content_hash(snippet))
                apply_passage_checks(focus, hit, resolution)
                resolved.append(hit)
                continue
            if hit.provider in scholarly and (hit.page_text or hit.snippet):
                body = hit.page_text or hit.snippet
                from factcheck.checks import UrlResolution, content_hash

                resolution = UrlResolution(hit.url, True, 200, hit.url, body, content_hash(body))
            else:
                resolution = self.url_checker.resolve(hit.url, body=hit.snippet)
                fetched += 1
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
        ranked = sorted(hits, key=lambda hit: overlap(claim.text, f"{hit.title} {hit.page_text or hit.snippet} {hit.quote}"), reverse=True)
        chosen = []
        for hit in ranked:
            blob = " ".join(part for part in (hit.title, hit.page_text, hit.snippet, hit.quote) if part)
            score = overlap(claim.text, blob)
            floor = 0.12 if is_fact_checker(hit.url, hit.source_type) or is_official(hit.url) else 0.2
            if score < floor:
                continue
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
            verdict="Unverified" if decision.verdict == "Unsupported" else decision.verdict,
            strength_of_support=strength,
            confidence=strength,
            reasoning=decision.reasoning,
            attribution=claim.attribution,
            span=getattr(claim, "span", "") or "",
            needs_human_review=review,
            hypothesis_status=decision.hypothesis_status,
            would_settle=decision.would_settle,
            checked_at=datetime.now(timezone.utc).isoformat(),
            deterministic_checks=[CheckRow(**row) for row in checks],
            tool=tool,
            judges=judges,
            sources=[_source_out(item) for item in sources],
        )

    def _assemble(self, text, claims, outputs, meter, warnings, client, charge, mode="text", draft_answer=None, question=None, deadline=None) -> VerificationOut:
        tier = "eval"
        credits = 0.0
        if charge:
            if getattr(client, "owner", False):
                tier = "owner"
                credits = 0.0
            elif getattr(client, "api_key", None):
                tier = "paid"
                credits = self.settings.credits_per_check
            else:
                tier = "free"
        if text:
            compact = " ".join(text.split())
            summary_in = compact[:240] + ("…" if len(compact) > 240 else "")
        else:
            count = len(claims or [])
            summary_in = f"{count} claim{'s' if count != 1 else ''} submitted."
        dialogue = split_dialogue(text) if text and mode != "question" else None
        synopsis = build_synopsis(
            outputs,
            question=question or (dialogue.question if dialogue else None),
            phrasing=self.synopsizer,
            deadline=deadline,
        )
        return VerificationOut(
            verification_id=str(uuid4()),
            created_at=datetime.now(timezone.utc).isoformat(),
            input_summary=summary_in,
            claims=outputs,
            mode=mode,
            draft_answer=draft_answer,
            synopsis=synopsis,
            summary=_summary(outputs),
            attribution=DISCLAIMER,
            disclaimer=DISCLAIMER,
            usage=UsageOut(
                input_tokens=meter.input_tokens,
                output_tokens=meter.output_tokens,
                estimated_cost_usd=round(meter.estimated_cost_usd, 6),
                retriever_cost_usd=round(meter.retriever_cost_usd, 6),
                retrieval_calls=int(getattr(meter, "retrieval_calls", 0)),
                drops=drop_counts(),
                credits_charged=credits,
                tier=tier,
                models=meter.models,
            ),
            upgrade_url=self.settings.resolved_upgrade_url(),
            warnings=list(dict.fromkeys(warnings)),
        )
