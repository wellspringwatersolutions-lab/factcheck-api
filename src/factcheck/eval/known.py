"""Known-answer set. Dry-run checks the file. --live calls the checker and is not run in CI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_LABELS = {"Supported", "Contradicted", "Unverified", "Conflicting", "Hypothesis", "Opinion"}
_PATH = Path(__file__).with_name("known_answers.json")


def load_known(path: Path | None = None) -> list[dict]:
    rows = json.loads((path or _PATH).read_text())
    if len(rows) < 20:
        raise ValueError("the known-answer set needs at least 20 claims")
    for row in rows:
        if row.get("label") not in _LABELS or not row.get("text") or not row.get("id"):
            raise ValueError(f"bad known-answer row: {row.get('id')}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a small set of claims with known labels.")
    parser.add_argument("--live", action="store_true", help="Call the configured checker. This spends model and search calls.")
    parser.add_argument("--path", default="")
    args = parser.parse_args()
    rows = load_known(Path(args.path) if args.path else None)
    if not args.live:
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["label"]] = counts.get(row["label"], 0) + 1
        print(f"{len(rows)} known answers loaded. Not run live.")
        print(", ".join(f"{name} {count}" for name, count in sorted(counts.items())))
        return
    from factcheck.config import get_settings
    from factcheck.wiring import build_service

    service = build_service(get_settings())
    matched = 0
    for row in rows:
        result = service.verify(text=row["text"], claims=None, client=None, charge=False)
        got = "Opinion" if result.claims and result.claims[0].claim_type == "opinion" else (result.claims[0].verdict if result.claims else "Unverified")
        mark = "ok" if got == row["label"] else "miss"
        if mark == "ok":
            matched += 1
        print(f"{row['id']} {mark} expected {row['label']} got {got}")
    print(f"{matched}/{len(rows)} matched")


if __name__ == "__main__":
    main()
