#!/usr/bin/env python3
"""Print per-claim verdicts and the bottom line from a live factcheck API.

This script calls the network. Tests do not run it.

Examples:
  python scripts/regress.py --base https://your-api.example washington
  python scripts/regress.py --base "$FACTCHECK_BASE_URL" adams
  python scripts/regress.py --base "$FACTCHECK_BASE_URL" --file gemini-paste.txt

Built-in names are `washington` and `adams`. Anything else is a file path.
A paste longer than 3,500 characters uses POST /check with {"async": true}
and polls GET /v1/jobs/{job_id}. Shorter inputs stay on the sync path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

_CASES = {
    "washington": "Was George Washington a criminal?",
    "adams": "was john adams a criminal",
}


def _headers(args: argparse.Namespace) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "factcheck-regress/1",
        "X-Client-Fingerprint": "regress",
    }
    if args.key:
        headers["Authorization"] = f"Bearer {args.key}"
    if args.owner_key:
        headers["X-Owner-Key"] = args.owner_key
    return headers


def _request(url: str, headers: dict[str, str], payload: dict | None = None, timeout: float = 30) -> tuple[int, dict]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if payload is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode()
            return response.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {"error": raw[:500]}
        return exc.code, parsed


def _poll(base: str, job_id: str, headers: dict[str, str], timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    url = base.rstrip("/") + f"/v1/jobs/{job_id}"
    last: dict = {}
    while time.monotonic() < deadline:
        status, last = _request(url, headers, None, timeout=30)
        if status != 200:
            raise SystemExit(f"GET {url} returned {status}: {json.dumps(last)[:400]}")
        print(f"job {job_id} {last.get('status')} {last.get('progress')}", file=sys.stderr)
        if last.get("status") in {"done", "error"}:
            return last
        time.sleep(2)
    raise SystemExit(f"job {job_id} was not done after {timeout:.0f}s: {json.dumps(last)[:400]}")


def check(base: str, text: str, headers: dict[str, str], timeout: float) -> dict:
    payload = {"text": text}
    if len(text) > 3500:
        payload["async"] = True
    status, body = _request(base.rstrip("/") + "/check", headers, payload, timeout=timeout)
    if status == 202:
        job_id = body.get("job_id")
        if not job_id:
            raise SystemExit(f"202 without job_id: {body}")
        polled = _poll(base, job_id, headers, timeout)
        if polled.get("status") != "done":
            raise SystemExit(f"job failed: {json.dumps(polled)[:800]}")
        report = polled.get("report") or {}
        report["_job_id"] = job_id
        return report
    if status != 200:
        raise SystemExit(f"POST /check returned {status}: {json.dumps(body)[:800]}")
    return body


def _load(name: str) -> tuple[str, str]:
    if name in _CASES:
        return name, _CASES[name]
    with open(name, encoding="utf-8") as handle:
        return name, handle.read()


def _print_report(label: str, report: dict) -> None:
    synopsis = report.get("synopsis") or {}
    print(f"\n== {label} ==")
    print(f"bottom line: {synopsis.get('answer')} / {synopsis.get('rating')}")
    print(f"tldr: {synopsis.get('tldr')}")
    counts = synopsis.get("counts") or {}
    if counts:
        print("counts: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    review = report.get("review") or {}
    print(f"review: passes={review.get('passes')} signed_off_by={review.get('signed_off_by')} changes={len(review.get('changes') or [])}")
    usage = report.get("usage") or {}
    if usage:
        print(
            "usage: "
            f"tokens={usage.get('input_tokens')}/{usage.get('output_tokens')} "
            f"usd={usage.get('estimated_cost_usd')} retrieval_calls={usage.get('retrieval_calls')}"
        )
    for claim in report.get("claims") or []:
        print(f"- {claim.get('verdict')}: {claim.get('claim')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run named fact-check cases against a live API and print the verdicts.")
    parser.add_argument("cases", nargs="*", help="washington, adams, or a path to a paste")
    parser.add_argument("--base", default=os.environ.get("FACTCHECK_BASE_URL", ""), help="API origin, no trailing path")
    parser.add_argument("--key", default=os.environ.get("FACTCHECK_API_KEY", ""), help="Optional bearer token")
    parser.add_argument("--owner-key", default=os.environ.get("OWNER_KEY", ""), help="Optional X-Owner-Key")
    parser.add_argument("--file", default="", help="Paste file, in addition to the named cases")
    parser.add_argument("--text", default="", help="One claim or question")
    parser.add_argument("--timeout", type=float, default=150, help="Seconds to wait, including async polling")
    args = parser.parse_args()
    if not args.base:
        raise SystemExit("Pass --base or set FACTCHECK_BASE_URL.")
    jobs: list[tuple[str, str]] = []
    for name in args.cases:
        jobs.append(_load(name))
    if args.file:
        jobs.append(_load(args.file))
    if args.text:
        jobs.append(("text", args.text))
    if not jobs:
        jobs = [_load("washington"), _load("adams")]
    headers = _headers(args)
    for label, text in jobs:
        report = check(args.base, text, headers, args.timeout)
        _print_report(label, report)


if __name__ == "__main__":
    main()
