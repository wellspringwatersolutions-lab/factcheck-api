import httpx

from factcheck.scholarly import (
    OpenAlexRetriever,
    ScholarlyBundle,
    SemanticScholarRetriever,
    dedupe_papers,
    rank_score,
    study_type_of,
)
from factcheck.types import Evidence


def test_study_type_ranks_reviews_above_trials():
    assert study_type_of("A systematic review and meta-analysis of vaccine efficacy") == "systematic_review"
    assert study_type_of("A randomized controlled trial of the drug") == "rct"
    assert rank_score("systematic_review", 2, 0, False) > rank_score("rct", 5000, 40, True)


def test_semantic_scholar_uses_tldr_and_rate_limit_is_one_per_second():
    retriever = SemanticScholarRetriever(api_key="s2-key", min_interval=0)
    assert SemanticScholarRetriever().limiter.interval == 1.0

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "s2-key"
        assert "tldr" in request.url.params["fields"]
        assert "influentialCitationCount" in request.url.params["fields"]
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "paperId": "abc",
                        "title": "A randomized trial of treatment",
                        "abstract": "Patients improved.",
                        "tldr": {"text": "The trial found a benefit."},
                        "year": 2020,
                        "influentialCitationCount": 12,
                        "externalIds": {"DOI": "10.1000/abc", "PubMed": "99"},
                        "openAccessPdf": {"url": "https://example.org/abc.pdf"},
                    }
                ]
            },
        )

    retriever.client = httpx.Client(transport=httpx.MockTransport(handler))
    hits = retriever.search("trial")
    assert hits[0].source_type == "peer_reviewed"
    assert hits[0].study_type == "rct"
    assert hits[0].influential_citations == 12
    assert "benefit" in hits[0].page_text
    assert hits[0].doi == "10.1000/abc"
    assert hits[0].pmid == "99"


def test_rate_limit_returns_no_hits():
    retriever = SemanticScholarRetriever(min_interval=0)
    retriever.client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(429, json={"message": "slow"})))
    assert retriever.search("gravity") == []
    assert "rate_limited" in retriever.warnings[-1]


def test_bundle_picks_sources_dedupes_doi_and_caches():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        calls.append(host)
        if host == "api.openalex.org":
            assert request.url.params["mailto"] == "lab@example.com"
            assert request.url.params["api_key"] == "oa-key"
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": "https://openalex.org/W1",
                            "doi": "https://doi.org/10.1000/same",
                            "display_name": "A systematic review of germ theory evidence",
                            "publication_year": 2021,
                            "type": "review",
                            "cited_by_count": 4,
                            "open_access": {"is_oa": True, "oa_url": "https://example.org/review"},
                            "primary_location": {"source": {"display_name": "Nature"}},
                            "abstract_inverted_index": {"Germ": [0], "theory": [1], "holds": [2]},
                            "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/55"},
                        },
                        {
                            "id": "https://openalex.org/W2",
                            "doi": "https://doi.org/10.1000/rct",
                            "display_name": "A randomized trial of treatment",
                            "publication_year": 2022,
                            "type": "article",
                            "cited_by_count": 900,
                            "open_access": {"is_oa": False},
                            "primary_location": {"source": {"display_name": "Lancet"}},
                            "abstract_inverted_index": {"Patients": [0], "improved": [1]},
                            "ids": {},
                        },
                    ]
                },
            )
        if host == "api.semanticscholar.org":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "paperId": "dup",
                            "title": "A systematic review of germ theory evidence",
                            "abstract": "Germ theory holds in this longer abstract from Semantic Scholar.",
                            "year": 2021,
                            "influentialCitationCount": 3,
                            "externalIds": {"DOI": "10.1000/same", "PubMed": "55"},
                            "citationCount": 4,
                        }
                    ]
                },
            )
        if host == "eutils.ncbi.nlm.nih.gov" and "esearch" in request.url.path:
            assert request.url.params["api_key"] == "ncbi-key"
            return httpx.Response(200, json={"esearchresult": {"idlist": ["55"]}})
        if host == "eutils.ncbi.nlm.nih.gov":
            xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>55</PMID>
              <Article><ArticleTitle>A systematic review of germ theory evidence</ArticleTitle>
              <Abstract><AbstractText>Germ theory holds.</AbstractText></Abstract></Article>
              </MedlineCitation><PubmedData><ArticleIdList>
              <ArticleId IdType="doi">10.1000/same</ArticleId></ArticleIdList></PubmedData>
              </PubmedArticle></PubmedArticleSet>"""
            return httpx.Response(200, text=xml)
        if host == "www.ebi.ac.uk" and request.url.path.endswith("/fullTextXML"):
            return httpx.Response(200, text="<article><p>Germ theory holds in the open full text.</p></article>")
        if host == "www.ebi.ac.uk":
            return httpx.Response(
                200,
                json={
                    "resultList": {
                        "result": [
                            {
                                "title": "A systematic review of germ theory evidence",
                                "abstractText": "Short abstract.",
                                "doi": "10.1000/same",
                                "pmid": "55",
                                "pmcid": "PMC1",
                                "isOpenAccess": "Y",
                                "pubYear": "2021",
                                "pubType": "review",
                                "citedByCount": 4,
                                "journalTitle": "Nature",
                            }
                        ]
                    }
                },
            )
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    bundle = ScholarlyBundle(
        openalex_email="lab@example.com",
        openalex_api_key="oa-key",
        semantic_scholar_api_key="s2",
        ncbi_api_key="ncbi-key",
        client=client,
    )
    bundle.semantic_scholar.limiter.interval = 0
    bundle.pubmed.limiter.interval = 0
    bundle.openalex.limiter.interval = 0
    bundle.europepmc.limiter.interval = 0

    law = bundle.search("bodies fall under gravity", categories=["T2"])
    assert all(host not in calls for host in ("eutils.ncbi.nlm.nih.gov", "www.ebi.ac.uk"))
    assert law[0].study_type == "systematic_review"
    assert law[0].source_type == "peer_reviewed"
    assert law[0].study_type != law[-1].study_type

    calls.clear()
    medical = bundle.search("a randomized vaccine trial in patients", categories=["T4"])
    assert "eutils.ncbi.nlm.nih.gov" in calls
    assert "www.ebi.ac.uk" in calls
    identities = {hit.doi for hit in medical}
    assert identities == {"10.1000/same"} or "10.1000/same" in identities
    same = [hit for hit in medical if hit.doi == "10.1000/same"]
    assert len(same) == 1
    assert "full text" in same[0].page_text or "longer abstract" in same[0].page_text

    before = len(calls)
    again = bundle.search("a randomized vaccine trial in patients", categories=["T4"])
    assert len(calls) == before
    assert [hit.doi for hit in again] == [hit.doi for hit in medical]


def test_openalex_without_email_does_not_call():
    called = []
    retriever = OpenAlexRetriever(
        email="",
        client=httpx.Client(transport=httpx.MockTransport(lambda request: called.append(1) or httpx.Response(500))),
        min_interval=0,
    )
    assert retriever.search("gravity") == []
    assert called == []
    assert "missing_email" in retriever.warnings[0]


def test_dedupe_keeps_one_paper():
    first = Evidence(url="https://a.example", title="T", snippet="short", doi="10.1/x", source_type="peer_reviewed", rank_score=1)
    second = Evidence(
        url="https://b.example",
        title="T",
        snippet="a much longer abstract of the same paper",
        page_text="a much longer abstract of the same paper",
        doi="10.1/x",
        source_type="peer_reviewed",
        rank_score=5,
    )
    kept = dedupe_papers([first, second])
    assert len(kept) == 1
    assert kept[0].rank_score == 5
    assert "longer abstract" in (kept[0].page_text or kept[0].snippet)
