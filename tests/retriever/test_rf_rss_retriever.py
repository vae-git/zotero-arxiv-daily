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
