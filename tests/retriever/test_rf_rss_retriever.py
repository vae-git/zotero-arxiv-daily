from types import SimpleNamespace

from omegaconf import open_dict

from zotero_arxiv_daily.retriever.base import get_retriever_cls
from zotero_arxiv_daily.retriever.rf_rss_retriever import RfRssRetriever


def test_rf_rss_retriever(config, monkeypatch):
    entries = [
        {
            "title": "A compact microwave filter for RF front ends",
            "link": "https://ieeexplore.ieee.org/document/1",
            "summary": "A microwave filter design for RF systems.",
            "authors": [{"name": "Author A"}],
            "published": "Fri, 20 Jun 2026 12:30:00 GMT",
        }
    ]
    parsed_feed = SimpleNamespace(entries=entries, bozo=False)
    monkeypatch.setattr("zotero_arxiv_daily.retriever.rf_rss_retriever.feedparser.parse", lambda _: parsed_feed)

    with open_dict(config.source):
        config.source.rf_rss = {
            "feeds": {"T-MTT": "https://example.com/rss.xml"},
            "max_entries_per_feed": 1,
        }

    retriever = RfRssRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == 1
    assert papers[0].source == "rf_rss"
    assert papers[0].title.startswith("[T-MTT]")
    assert "microwave filter" in papers[0].abstract
    assert papers[0].authors == ["Author A"]
    assert papers[0].published_date == "2026-06-20"
    assert papers[0].venue == "IEEE Transactions on Microwave Theory and Techniques"
    assert "Top RF/microwave journal" in papers[0].venue_rank
    assert "\u4e2d\u79d1\u9662 2 \u533a" in papers[0].cas_partition
    assert "JCR Q1" in papers[0].sci_quartile


def test_rf_rss_retriever_crossref_fallback(config, monkeypatch):
    parsed_feed = SimpleNamespace(entries=[], bozo=False)
    monkeypatch.setattr("zotero_arxiv_daily.retriever.rf_rss_retriever.feedparser.parse", lambda _: parsed_feed)
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    class StubResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "message": {
                    "items": [
                        {
                            "title": ["Table of Contents"],
                            "URL": "https://doi.org/10.1109/toc",
                            "DOI": "10.1109/toc",
                            "published-online": {"date-parts": [[2026, 7, 2]]},
                        },
                        {
                            "title": ["A Broadband RF Power Amplifier With Digital Predistortion"],
                            "URL": "https://doi.org/10.1109/example",
                            "DOI": "10.1109/example",
                            "author": [{"given": "Author", "family": "A"}],
                            "abstract": "<jats:p>A power amplifier paper.</jats:p>",
                            "published-online": {"date-parts": [[2026, 7, 1]]},
                        }
                    ]
                }
            }

    monkeypatch.setattr("zotero_arxiv_daily.retriever.rf_rss_retriever.requests.get", lambda *a, **kw: StubResponse())

    with open_dict(config.source):
        config.source.rf_rss = {
            "feeds": {"T-MTT": "https://example.com/rss.xml"},
            "crossref_issn": {"T-MTT": "0018-9480"},
            "max_entries_per_feed": 1,
        }

    retriever = RfRssRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == 1
    assert papers[0].title.startswith("[T-MTT]")
    assert "RF Power Amplifier" in papers[0].title
    assert papers[0].authors == ["Author A"]
    assert papers[0].published_date == "2026-07-01"
    assert "power amplifier paper" in papers[0].abstract


def test_rf_rss_retriever_registered():
    assert get_retriever_cls("rf_rss") is RfRssRetriever


def _stub_crossref(monkeypatch, items, recorded_params=None):
    class StubResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"items": items}}

    def stub_get(url, params=None, headers=None, timeout=None):
        if recorded_params is not None:
            recorded_params.append((url, params))
        return StubResponse()

    monkeypatch.setattr("zotero_arxiv_daily.retriever.rf_rss_retriever.requests.get", stub_get)


def test_crossref_query_applies_lookback_window(config, monkeypatch):
    recorded = []
    _stub_crossref(monkeypatch, [], recorded)
    monkeypatch.setattr("zotero_arxiv_daily.retriever.rf_rss_retriever.feedparser.parse",
                        lambda _: SimpleNamespace(entries=[], bozo=False))

    with open_dict(config.source):
        config.source.rf_rss = {
            "feeds": {},
            "crossref_issn": {"T-MTT": "0018-9480"},
            "crossref_queries": ["power amplifier"],
            "max_entries_per_feed": 5,
            "lookback_days": 30,
        }

    RfRssRetriever(config)._retrieve_raw_papers()

    assert recorded
    _, params = recorded[0]
    assert "from-pub-date:" in params["filter"]
    assert "type:journal-article" in params["filter"]


def test_crossref_supports_proceedings_and_prefix_sources(config, monkeypatch):
    recorded = []
    _stub_crossref(monkeypatch, [], recorded)
    monkeypatch.setattr("zotero_arxiv_daily.retriever.rf_rss_retriever.feedparser.parse",
                        lambda _: SimpleNamespace(entries=[], bozo=False))

    with open_dict(config.source):
        config.source.rf_rss = {
            "feeds": {},
            "crossref_issn": {},
            "crossref_queries": ["power amplifier"],
            "max_entries_per_feed": 5,
            "crossref_sources": {
                "IMS": {"kind": "proceedings", "container": "IEEE MTT-S International Microwave Symposium"},
                "TechRxiv": {"kind": "prefix", "prefix": "10.36227"},
            },
        }

    RfRssRetriever(config)._retrieve_raw_papers()

    urls = [url for url, _ in recorded]
    params_list = [params for _, params in recorded]
    assert any("prefixes/10.36227" in url for url in urls)
    assert any(p.get("query.container-title", "").startswith("IEEE MTT-S") for p in params_list)
    assert any("type:proceedings-article" in p["filter"] for p in params_list)


def test_crossref_sources_are_not_queried_when_rss_succeeds(config, monkeypatch):
    recorded = []
    _stub_crossref(monkeypatch, [], recorded)
    entries = [{
        "title": "An RF power amplifier",
        "link": "https://ieeexplore.ieee.org/document/1",
        "summary": "A PA design.",
        "authors": [{"name": "Author A"}],
        "published": "Fri, 20 Jun 2026 12:30:00 GMT",
    }]
    monkeypatch.setattr("zotero_arxiv_daily.retriever.rf_rss_retriever.feedparser.parse",
                        lambda _: SimpleNamespace(entries=entries, bozo=False))
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    with open_dict(config.source):
        config.source.rf_rss = {
            "feeds": {"T-MTT": "https://example.com/rss.xml"},
            "crossref_issn": {"T-MTT": "0018-9480"},
            "max_entries_per_feed": 5,
        }

    papers = RfRssRetriever(config).retrieve_papers()

    assert len(papers) == 1
    assert recorded == []


def test_missing_abstract_is_recovered_via_enricher(config, monkeypatch):
    recovered = "A broadband Doherty power amplifier achieving high back-off efficiency across the band."
    _stub_crossref(monkeypatch, [{
        "title": ["A Broadband Doherty Power Amplifier"],
        "URL": "https://doi.org/10.1109/example",
        "DOI": "10.1109/example",
        "published-online": {"date-parts": [[2026, 7, 1]]},
    }])
    monkeypatch.setattr("zotero_arxiv_daily.retriever.rf_rss_retriever.feedparser.parse",
                        lambda _: SimpleNamespace(entries=[], bozo=False))
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    monkeypatch.setattr("zotero_arxiv_daily.retriever.rf_rss_retriever.enrich_abstract",
                        lambda doi_or_url: recovered)

    with open_dict(config.source):
        config.source.rf_rss = {
            "feeds": {},
            "crossref_issn": {"T-MTT": "0018-9480"},
            "crossref_queries": ["power amplifier"],
            "max_entries_per_feed": 1,
        }

    papers = RfRssRetriever(config).retrieve_papers()

    assert len(papers) == 1
    assert papers[0].abstract == recovered
    assert papers[0].abstract_is_placeholder is False


def test_placeholder_abstract_is_flagged_when_recovery_fails(config, monkeypatch):
    _stub_crossref(monkeypatch, [{
        "title": ["A Broadband Doherty Power Amplifier"],
        "URL": "https://doi.org/10.1109/example",
        "DOI": "10.1109/example",
        "published-online": {"date-parts": [[2026, 7, 1]]},
    }])
    monkeypatch.setattr("zotero_arxiv_daily.retriever.rf_rss_retriever.feedparser.parse",
                        lambda _: SimpleNamespace(entries=[], bozo=False))
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    monkeypatch.setattr("zotero_arxiv_daily.retriever.rf_rss_retriever.enrich_abstract",
                        lambda doi_or_url: None)

    with open_dict(config.source):
        config.source.rf_rss = {
            "feeds": {},
            "crossref_issn": {"T-MTT": "0018-9480"},
            "crossref_queries": ["power amplifier"],
            "max_entries_per_feed": 1,
        }

    papers = RfRssRetriever(config).retrieve_papers()

    assert len(papers) == 1
    assert papers[0].abstract.startswith("Latest T-MTT")
    assert papers[0].abstract_is_placeholder is True


def test_one_failing_source_does_not_kill_the_others(config, monkeypatch):
    def stub_get(url, params=None, headers=None, timeout=None):
        if "prefixes" in url:
            raise RuntimeError("prefix source is down")

        class StubResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"message": {"items": [{
                    "title": ["A Doherty Power Amplifier"],
                    "URL": "https://doi.org/10.1109/ok",
                    "DOI": "10.1109/ok",
                    "abstract": "A power amplifier paper with a sufficiently long abstract body.",
                    "published-online": {"date-parts": [[2026, 7, 1]]},
                }]}}

        return StubResponse()

    monkeypatch.setattr("zotero_arxiv_daily.retriever.rf_rss_retriever.requests.get", stub_get)
    monkeypatch.setattr("zotero_arxiv_daily.retriever.rf_rss_retriever.feedparser.parse",
                        lambda _: SimpleNamespace(entries=[], bozo=False))
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    with open_dict(config.source):
        config.source.rf_rss = {
            "feeds": {},
            "crossref_issn": {"T-MTT": "0018-9480"},
            "crossref_queries": ["power amplifier"],
            "max_entries_per_feed": 1,
            "crossref_sources": {"TechRxiv": {"kind": "prefix", "prefix": "10.36227"}},
        }

    papers = RfRssRetriever(config).retrieve_papers()

    # TechRxiv 抛错只应跳过它自己，T-MTT 仍要返回结果。
    assert [p.title for p in papers] == ["[T-MTT] A Doherty Power Amplifier"]

