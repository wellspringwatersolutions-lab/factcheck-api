import httpx

from factcheck.retrievers import DesearchRetriever, PubMedRetriever, SemanticScholarRetriever


def test_desearch_search_and_extract_paths():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "test-key"
        if request.url.path == "/web":
            assert request.url.params["query"] == "taxes"
            return httpx.Response(
                200,
                json={"data": [{"link": "https://www.reuters.com/a", "title": "Taxes", "snippet": "The council raised taxes."}], "cost_usd": 0},
                headers={"X-Desearch-Cost-Usd": "0.01"},
            )
        if request.url.path == "/web/extract":
            return httpx.Response(
                200,
                text="The council raised taxes on 2019-05-01.\n\nMore text.",
                headers={"X-Desearch-Cost-Usd": "0.01"},
            )
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    retriever = DesearchRetriever("test-key", client=client)
    hits = retriever.search("taxes")
    assert hits[0].url == "https://www.reuters.com/a"
    page = retriever.extract(hits[0].url)
    assert "2019-05-01" in page
    assert retriever.cost_usd == 0.02


def test_academic_parsers():
    def handler(request: httpx.Request) -> httpx.Response:
        if "semanticscholar" in str(request.url):
            return httpx.Response(
                200,
                json={"data": [{"paperId": "abc", "title": "A trial", "abstract": "A randomized trial found a reduction.", "year": 2020}]},
            )
        if "esearch" in str(request.url):
            return httpx.Response(200, json={"esearchresult": {"idlist": ["1"]}})
        xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>1</PMID><Article><ArticleTitle>Trial</ArticleTitle><Abstract><AbstractText>A randomized trial.</AbstractText></Abstract><Journal><JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue></Journal></Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""
        return httpx.Response(200, text=xml)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    papers = SemanticScholarRetriever(client=client).search("trial")
    assert papers[0].url.endswith("/abc")
    assert papers[0].source_type == "peer_reviewed"
    articles = PubMedRetriever(client=client).search("trial")
    assert articles[0].url.endswith("/1/")
    assert "randomized" in articles[0].snippet
