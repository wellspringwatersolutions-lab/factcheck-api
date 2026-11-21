# factcheck

`factcheck` checks a passage or a list of claims for agents and researchers. It extracts atomic claims, recomputes math, fetches evidence, and returns structured JSON. A **Supported** verdict is shown only when a retrieved passage or a tool result backs the claim, and the report quotes that passage. If there is no passage, the verdict is **Unverified**, no matter what a model would have said.

Language models rank and filter evidence. They are not the basis of a verdict. Two rankers (different model families) can drop passages that are off topic. A third model is used only when those two disagree about relevance. None of them can create a Supported verdict on their own.

Verdicts are evidence summaries. Grounding reduces errors. It cannot prove that no better evidence exists, and a verdict is not proof.

The product name lives in `src/factcheck/branding.py` as `SERVICE_NAME`.

## What you get back

`POST /v1/verify` and `POST /check` accept `{"text": "..."}`, `{"claims": ["..."]}`, or `{"question": "..."}`. A short text that is only a question is treated the same as `question`. Both responses add these fields and keep the existing ones:

- `mode`: `"text"` or `"question"`
- `draft_answer`: `{"model", "text"}` when `mode` is `"question"`, otherwise `null`. The draft is not evidence and has no weight in a verdict.
- `synopsis`: `{"rating", "tldr", "summary", "caveats", "unchecked_count", "counts"}`. `rating` is one of Accurate, Mostly accurate, Mixed, Mostly inaccurate, Inaccurate, Not enough evidence. It is computed from the claim verdicts and cited quotes. `tldr` is one sentence of at most 30 words, written in the same phrasing call as `summary` and dropped if it adds a fact. `counts` is `{"supported", "contradicted", "conflicting", "unverified", "hypothesis", "opinion", "not_checked"}`. When a cited passage contradicts the draft, `summary` states that quote as the corrected position. A short question without a `?` still counts as a question when it starts with a question word.
- A failed draft is HTTP 503 `{"error": {"code": "draft_failed", ...}}`. That request is not charged.

`POST /v1/verify` also returns:

- `verification_id`, `created_at`, an input summary
- `claims[]` with `claim_type`, `claim_category` (T1–T8), `verdict`, `strength_of_support` (also copied to `confidence`), `reasoning`, sources, and deterministic checks
- for opinions, `mainstream_check`: what professional fact-checkers said, never a true/false label
- `summary`, `attribution`, `usage` (tokens, estimated USD, credits), `upgrade_url`
- `needs_human_review` when the claim is high-stakes or the evidence is thin or contradictory

`strength_of_support` is computed from source grade, how many independent sources agree, and whether the passages line up. It is not a model's self-reported confidence, and it is not a probability that the claim is true.

Verdicts:

| Verdict | When |
| --- | --- |
| Supported | A quoted passage from one authoritative source entails the claim, or two independent sources do. More independent sources raise `strength_of_support`. A tool result (calculator, or a structured API such as REST Countries) also counts. |
| Contradicted | A passage or a tool result conflicts with the claim, and the conflicting quote or trace is shown. |
| Unsupported | Some evidence lines up, but not enough independent sources to call it supported. The service abstains. |
| Unverified | No usable passage and no tool result. |
| Conflicting | Independent sources disagree. |
| Hypothesis | A prediction, novel conjecture, or business idea. Status is `supporting_evidence`, `contradicting_evidence`, or `untested`. Never true or false. |

Categories change how strict the check is:

- **T1** Math/logic is recomputed in code. No web search.
- **T2 / T3** Textbook laws and strongly supported theories. Lighter retrieval when the claim has no numbers, dates, or quotes.
- **T4** Medical and empirical results. Systematic reviews weigh more than trials, which weigh more than observational papers, which weigh more than expert opinion. That order is a default: a weak paper still loses to a stronger record.
- **T5** Historical claims. Primary documents and data outrank summaries. Syndicated copies of one story count once.
- **T6** Expert consensus. Weaker when sources conflict.
- **T7** Eyewitness accounts. Low weight.
- **T8** Opinions and anecdotes. These go to the mainstream fact-checker lane (Google Fact Check Tools, then Snopes, PolitiFact, FactCheck.org, AP Fact Check, and Reuters Fact Check).

Numbers, dates, exact quotes, citations, and anything time-sensitive get a stricter retrieval pass. Cited URLs are fetched, quoted text has to appear on the page (exact or fuzzy), and the page body is hashed for the audit trail. A dead URL or a quote that is not on the page cannot support a claim.

Human-readable report: `GET /v1/verification/{id}/report`. Facts, opinions, and hypotheses use different colors. The legend is on the page.

## Run locally without Docker

Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python -m factcheck.eval --dry-run --sample-size 50 --output reports/dry-run.md
python -m factcheck.cli serve --port 8741
```

With no API keys the service still runs. Math is recomputed locally. Fact claims come back Unverified until a retriever can fetch a page. SQLite (`./data/factcheck.db`) and an in-memory cache are the defaults, so Postgres and Redis are optional.

Open `http://127.0.0.1:8741/health`. Interactive API docs are at `/docs`.

Create a local key with credits:

```bash
python -m factcheck.cli create-key --credits 10
```

Send it as `Authorization: Bearer fc_...` or `X-API-Key`.

## Render

`render.yaml` is a Starter web service. It uses the Dockerfile, checks `/health`, and stores SQLite on a 1 GB disk at `/var/data`. Redis and Postgres are not required. CORS stays limited to `https://wellspringwatersolutions-lab.github.io`.

In the Render dashboard, create a Blueprint from this repo and fill the `sync: false` values (`CHUTES_API_KEY`, `DESEARCH_API_KEY`, `OPENALEX_EMAIL`, and any optional keys). Set `PUBLIC_BASE_URL` to the Render URL. `FREE_DAILY_CAP_USD` is already `20`.

## Docker

```bash
cp .env.example .env
docker compose up --build
```

That starts the API on port 8741, Postgres, and Redis. Put keys in `.env`. `.env.example` lists names only.

## Environment variables

| Name | Role |
| --- | --- |
| `CHUTES_API_KEY` | Primary OpenAI-compatible provider. Base URL `https://llm.chutes.ai/v1`. |
| `JUDGE_A_MODEL` | Ranker A. Default `deepseek-ai/DeepSeek-V3.2-TEE` (DeepSeek, TEE). |
| `JUDGE_B_MODEL` | Ranker B. Default `Qwen/Qwen3.5-397B-A17B-TEE` (Qwen, TEE). |
| `TIEBREAK_MODEL` | Used only when A and B disagree about which passages are relevant. Default `zai-org/GLM-5.1-TEE` (GLM, TEE). |
| `EXTRACT_MODEL` | Claimify-style extraction. Default same as judge A. |
| `SOMA_API_KEY` | Internal dev endpoint only (`https://app.thesoma.ai/v1`, model `openrouter/deepseek/deepseek-v4-pro`). Not for production. Ignored unless `SOMA_ENABLED=true`, or it is the only key available. |
| `OPENAI_COMPAT_BASE_URL` / `OPENAI_COMPAT_API_KEY` | Backup OpenAI-compatible provider. |
| `DESEARCH_API_KEY` | Web search (`GET /web`) and page text (`GET /web/extract`). `Authorization` is the raw key. |
| `GOOGLE_FACTCHECK_API_KEY` | Optional. Google Fact Check Tools `claims:search`. Skipped when blank. Ratings are cited as secondary evidence and do not decide the verdict. |
| `CLAIMBUSTER_API_KEY` | Optional. Check-worthiness scores used only to order claims under the cap. Skipped when blank or slow. |
| `NCBI_EMAIL` | Contact email sent with Wikidata, Chronicling America, and NCBI requests. |
| `SEMANTIC_SCHOLAR_API_KEY` | Optional. Without it, Semantic Scholar is limited to about 1 request per second. |
| `OPENALEX_EMAIL` | Mailto for the OpenAlex polite pool. Required for OpenAlex. |
| `OPENALEX_API_KEY` | Optional OpenAlex key. |
| `NCBI_API_KEY` | Optional. Raises the PubMed rate limit. |
| `DATABASE_URL` | Default SQLite. On Render use `sqlite:////var/data/factcheck.db`. |
| `FREE_DAILY_CAP_USD` | Same cap as `FREE_DAILY_SPEND_CAP_USD`. Default `20`. |
| `REDIS_URL` | Optional cache. Without it, the cache is in memory. |
| `FREE_DAILY_SPEND_CAP_USD` | Global cap for free checks. Default `20`. |
| `CREDITS_PER_CHECK` | Credits deducted per paid check. Default `1`. |
| `PUBLIC_BASE_URL` / `UPGRADE_URL` | Used in discovery docs and the upgrade line. |
| `OWNER_IPS` | Comma-separated client IPs that skip the one-free-check limit. They still count toward `FREE_DAILY_CAP_USD`. Behind Render, the client IP is the first `X-Forwarded-For` hop. |
| `OWNER_KEY` | Optional. Send `X-Owner-Key` for the same bypass. Leave this empty in the repo. |

Model ids were checked against `GET https://llm.chutes.ai/v1/models` on 21 Nov 2026. The catalog does not need a key. Refresh warnings with:

```bash
python -m factcheck.cli models
```

Prices used for estimates are the catalog's USD per million tokens on that date (DeepSeek-V3.2-TEE $1 / $1, Qwen3.5-397B-A17B-TEE $0.45 / $3, GLM-5.1-TEE $0.90 / $2.90). Unknown models use $1 / $3 per million so the estimate does not come out as zero. Desearch's `X-Desearch-Cost-Usd` header is added when present.

Failover order is Chutes, then the generic OpenAI-compatible backup, then Soma if it is enabled. A failed provider does not stop the check.

## Free tier and paid credits

The first check for a client is free. The client id is a hash of IP, User-Agent, and optional `X-Client-Fingerprint`. The next check from that client returns HTTP 402:

```json
{"error": {"code": "free_tier_exhausted", "message": "Free check used. More checks: http://127.0.0.1:8741/upgrade", "upgrade_url": "..."}}
```

Free checks also stop with HTTP 429 when the global daily spend cap is hit (`spend_cap_exceeded`). That reservation is not consumed, so the same client can retry after the cap resets. Paid calls with an API key use prepaid credits and do not touch the free cap. A missing key is 401. An empty balance is 402.

Stripe credit packs and TAO deposits are stub interfaces in `src/factcheck/payments.py`. `POST /v1/billing/checkout` returns 501 `payment_not_configured`. Nothing is charged.

## MCP

Streamable HTTP is at `/mcp`. No key is required for the one free check.

```text
add https://<your-host>/mcp to Claude/Cursor
```

Tools: `verify_text`, `verify_claims`, `get_verification`. Schemas set `additionalProperties` to false.

`/llms.txt` and `/.well-known/mcp.json` describe the server and that install line.

## Evaluation

The harness scores the pipeline against labels, not against another model agreeing with itself.

```bash
python -m factcheck.eval --dry-run --sample-size 50 --output reports/dry-run.md
```

Datasets:

- **SciFact** — 50 claims from the public AllenAI dev split (`claims_dev`) with the cited abstracts. Source: [allenai/scifact](https://github.com/allenai/scifact). This is a subset, not the full corpus.
- **FEVER** — an offline sample in FEVER's label format (`SUPPORTS`, `REFUTES`, `NOT ENOUGH INFO`) with the evidence text included. The official FEVER release points evidence at a 1.7 GB Wikipedia dump, which is not vendored. Replace `src/factcheck/eval/fixtures/fever_sample.jsonl` with your own resolved JSONL in the same shape (`id`, `claim`, `label`, `documents: [{id, title, text}]`) to score a larger set.
- **Held-out** — arithmetic with known answers, plus a few passage-backed facts.

Dry-run uses the deterministic passage check and the offline corpus. It does not call a paid model. The report includes accuracy, macro-F1, a confusion matrix, average latency, and average estimated cost. It also reports passage-level entailment separately from the final verdict, because Supported requires two independent sources and a single abstract often abstains (`Unsupported`, scored as Unverified so an abstention is not a lucky guess).

To run extraction and ranking against live models, set `CHUTES_API_KEY` (or another provider) and run:

```bash
python -m factcheck.eval --sample-size 50 --output reports/live.md
```

That still retrieves from the dataset corpus, which is what SciFact and FEVER measure. It does not spend Desearch credit. Token cost depends on how many claims the rankers see. At the default prices, a 50-claim sample is on the order of cents to a few dollars if every claim triggers extraction plus two rankers; the report prints the estimate from actual token counts. Keep `FREE_DAILY_SPEND_CAP_USD` in mind only for unauthenticated API traffic. The eval itself does not bill the free tier.

## Tests

```bash
pytest
```

Tests use mocked models and retrievers. No API key is required. They cover extraction parsing, verdict rules (no passage means Unverified, a fabricated quote and a dead URL are caught, T1 math is recomputed), T8 routing, syndicated-source dedupe, mainstream consensus, the free check, the spend cap, credits, and MCP tool schemas.

## Layout

```text
src/factcheck/          API, MCP, pipeline, storage, billing
src/factcheck/branding.py   SERVICE_NAME
src/factcheck/eval/     SciFact, FEVER-format, and held-out scoring
tests/                  mocked unit and API tests
docker-compose.yml      api, postgres, redis
```

## Cost notes

Retrieval is clustered: one web search per cluster of similar claims, not one search per claim. URLs are deduped, and syndicated copies of the same story count as one origin. Page extracts are capped (none on a light T2/T3 check, up to three on a strict check). Claim results are cached; time-sensitive claims expire sooner (30 minutes versus 24 hours).

A free check reserves `FREE_CHECK_RESERVE_USD` (default $0.25) against the daily cap before the models run, then records the actual estimate. Set the reserve near the high end of what one check costs so concurrent free checks cannot walk past the cap.
