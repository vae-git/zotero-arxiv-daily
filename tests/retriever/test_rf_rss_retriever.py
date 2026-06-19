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


def test_rf_rss_retriever_registered():
    assert get_retriever_cls("rf_rss") is RfRssRetriever
