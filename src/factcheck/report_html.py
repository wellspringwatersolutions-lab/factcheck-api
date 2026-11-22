"""HTML report. Fact checks and opinion checks use different colors."""

from __future__ import annotations

import html

from factcheck.branding import SERVICE_NAME
from factcheck.schemas import VerificationOut

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ margin: 0; font: 16px/1.5 "Iowan Old Style", Palatino, Georgia, serif; background: #f6f4ef; color: #1c1915; }}
  main {{ max-width: 840px; margin: 0 auto; padding: 28px 18px 64px; }}
  h1 {{ font-size: 1.7rem; margin-bottom: 0.2rem; }}
  .meta {{ color: #5c564c; font-size: 0.92rem; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 14px; margin: 18px 0; padding: 12px; background: white; border: 1px solid #e4dfd4; }}
  .swatch {{ width: 14px; height: 14px; display: inline-block; margin-right: 6px; vertical-align: -2px; }}
  .swatch-fact {{ background: #1d4e89; }}
  .swatch-opinion {{ background: #8a5a12; }}
  .swatch-hypothesis {{ background: #5c4d7a; }}
  article {{ margin: 14px 0; padding: 14px 16px; border-left: 6px solid #1d4e89; background: #e7f1fb; }}
  article.claim-opinion {{ border-left-color: #8a5a12; background: #fbf3e4; }}
  article.claim-hypothesis {{ border-left-color: #5c4d7a; background: #f3eef8; }}
  .kicker {{ font: 12px/1.3 ui-sans-serif, system-ui, sans-serif; letter-spacing: 0.04em; text-transform: uppercase; color: #3d3a34; }}
  blockquote {{ margin: 8px 0; padding-left: 10px; border-left: 3px solid #8d877c; color: #2a261f; }}
  .tag {{ font: 12px/1.3 ui-sans-serif, system-ui, sans-serif; background: white; border: 1px solid #d9d3c7; padding: 1px 6px; margin-right: 4px; }}
  .synopsis, .draft {{ margin-top: 22px; padding: 16px; background: white; border: 1px solid #e4dfd4; }}
  .draft {{ background: #f3f1ec; }}
  footer {{ margin-top: 28px; font-size: 0.92rem; color: #4d473e; }}
  a {{ color: #1d4e89; }}
</style>
</head>
<body>
<main>
  <h1>{title}</h1>
  <p class="meta">{meta}</p>
  <div class="legend">
    <span><i class="swatch swatch-fact"></i>Source-checked fact</span>
    <span><i class="swatch swatch-opinion"></i>Opinion / mainstream fact-checker coverage</span>
    <span><i class="swatch swatch-hypothesis"></i>Hypothesis (not true or false)</span>
  </div>
  {body}
  <footer>{footer}</footer>
</main>
</body>
</html>"""


def _article(claim) -> str:
    from factcheck.publish import map_verdict

    public, detail, label = map_verdict(claim.verdict, claim.claim_type, claim.verdict_detail)
    if public == "hypothesis":
        kind = "claim-hypothesis"
    elif public == "opinion" or claim.mainstream_check:
        kind = "claim-opinion"
    else:
        kind = "claim-fact"
    bits = [
        f'<p class="kicker">{html.escape(claim.claim_category)} · {html.escape(claim.category_label)}</p>',
        f"<h2>{html.escape(claim.text)}</h2>",
    ]
    if claim.mainstream_check:
        bits.append("<p><strong>What mainstream fact-checkers say:</strong> "
                    f"{html.escape(claim.mainstream_check.consensus.replace('_', ' '))}</p>")
        for item in claim.mainstream_check.checks:
            when = f" ({html.escape(item.published_at)})" if item.published_at else ""
            bits.append(
                f'<p><a href="{html.escape(item.url)}">{html.escape(item.outlet)}</a>: '
                f"{html.escape(item.rating)}{when}</p>"
            )
    else:
        bits.append(f"<p><strong>{html.escape(label)}</strong> · strength of support {claim.strength_of_support:.2f}</p>")
        if claim.hypothesis_status:
            bits.append(f"<p>Hypothesis status: {html.escape(claim.hypothesis_status.replace('_', ' '))}</p>")
        if claim.would_settle:
            bits.append(f"<p>{html.escape(claim.would_settle)}</p>")
    if claim.checked_at:
        bits.append(f"<p class=\"meta\">Checked {html.escape(claim.checked_at)}</p>")
    if claim.needs_human_review:
        bits.append("<p>Flagged for human review. The evidence trail is below.</p>")
    bits.append(f"<p>{html.escape(claim.reasoning)}</p>")
    for source in claim.sources:
        quote = source.quote or source.snippet
        bits.append(
            "<p>"
            f'<span class="tag">{html.escape(source.source_type)}</span>'
            + (f'<span class="tag">{html.escape(source.study_type.replace("_", " "))}</span>' if source.study_type else "")
            + f'<span class="tag">grade {html.escape(source.grade)}</span>'
            + f'<span class="tag">{html.escape(source.stance)}</span>'
            + f' <a href="{html.escape(source.url)}">{html.escape(source.title or source.url)}</a>'
            + "</p>"
        )
        if quote:
            bits.append(f"<blockquote>{html.escape(quote)}</blockquote>")
        if source.content_sha256:
            bits.append(f'<p class="meta">sha256 {html.escape(source.content_sha256[:16])}…</p>')
    for row in claim.deterministic_checks:
        mark = "passed" if row.passed else "failed"
        bits.append(f"<p>{html.escape(row.name)}: {mark}. {html.escape(row.detail)}</p>")
    return f'<article class="{kind}">{"".join(bits)}</article>'


def _draft(verification: VerificationOut) -> str:
    draft = verification.draft_answer
    if draft is None:
        return ""
    return (
        '<section class="draft">'
        '<p class="kicker">Drafted answer. This text is not evidence.</p>'
        f"<p>{html.escape(draft.text)}</p>"
        "</section>"
    )


def _synopsis(verification: VerificationOut) -> str:
    synopsis = verification.synopsis
    if synopsis is None:
        return ""
    bits = [
        '<section class="synopsis">',
        '<p class="kicker">Veracity synopsis</p>',
        f"<h2>{html.escape(synopsis.rating)}</h2>",
        (f"<p><strong>{html.escape(synopsis.tldr)}</strong></p>" if synopsis.tldr else ""),
        f"<p>{html.escape(synopsis.summary)}</p>",
    ]
    if synopsis.caveats:
        items = "".join(f"<li>{html.escape(item)}</li>" for item in synopsis.caveats)
        bits.append(f"<ul>{items}</ul>")
    if synopsis.unchecked_count:
        word = "claim" if synopsis.unchecked_count == 1 else "claims"
        verb = "was" if synopsis.unchecked_count == 1 else "were"
        bits.append(f"<p>{synopsis.unchecked_count} important {word} {verb} not checked.</p>")
    if verification.review.signed_off_by:
        bits.append(f"<p class=\"meta\">Signed off by {html.escape(verification.review.signed_off_by)}</p>")
    bits.append("</section>")
    return "".join(bits)


def render_report(verification: VerificationOut) -> str:
    articles = "".join(_article(claim) for claim in verification.claims) or "<p>No claims were extracted.</p>"
    body = _draft(verification) + articles + _synopsis(verification)
    meta = f"{html.escape(verification.verification_id)} · {html.escape(verification.created_at)}"
    return _PAGE.format(
        title=html.escape(SERVICE_NAME),
        meta=meta,
        body=body,
        footer=html.escape(verification.disclaimer),
    )


def render_home(base_url: str) -> str:
    install = f"add {base_url.rstrip('/')}/mcp to Claude/Cursor"
    text = f"""
    <h1>{html.escape(SERVICE_NAME)}</h1>
    <p>{html.escape(SERVICE_NAME)} checks a passage for agents and researchers.
    It separates checkable facts from opinions, recomputes math, and shows the passage a verdict rests on.</p>
    <p>Verdicts summarize evidence. They are not proof. Grounding reduces errors and cannot prove that no better evidence exists.</p>
    <p>Install the MCP server: <code>{html.escape(install)}</code></p>
    <p><a href="/docs">API docs</a> · <a href="/llms.txt">llms.txt</a> · <a href="/health">health</a></p>
    """
    return _PAGE.format(title=html.escape(SERVICE_NAME), meta="Fact-checking for agents", body=text, footer=html.escape(
        "Grounding reduces errors. It cannot prove that no better evidence exists."
    ))
