"""Tests for zotero_arxiv_daily.executor: normalize_path_patterns, filter_corpus, fetch_zotero_corpus, E2E."""

import json
from datetime import datetime
from email import message_from_string

import pytest
from omegaconf import OmegaConf

from zotero_arxiv_daily.executor import Executor, _paper_identifier_keys, normalize_path_patterns
from zotero_arxiv_daily.protocol import CorpusPaper


def _decode_email_body(raw_email: str) -> str:
    message = message_from_string(raw_email)
    payload = message.get_payload(decode=True)
    if payload is None:
        return raw_email
    return payload.decode(message.get_content_charset() or "utf-8", errors="replace")


# ---------------------------------------------------------------------------
# normalize_path_patterns — migrated from test_include_path.py
# ---------------------------------------------------------------------------


def test_normalize_path_patterns_rejects_single_string_for_include_path():
    with pytest.raises(TypeError, match="config.zotero.include_path must be a list"):
        normalize_path_patterns("2026/survey/**", "include_path")


def test_normalize_path_patterns_accepts_list_config_for_include_path():
    include_path = OmegaConf.create(["2026/survey/**", "2026/reading-group/**"])
    assert normalize_path_patterns(include_path, "include_path") == [
        "2026/survey/**",
        "2026/reading-group/**",
    ]


def test_normalize_path_patterns_rejects_single_string_for_ignore_path():
    with pytest.raises(TypeError, match="config.zotero.ignore_path must be a list"):
        normalize_path_patterns("archive/**", "ignore_path")


def test_normalize_path_patterns_accepts_list_config_for_ignore_path():
    ignore_path = OmegaConf.create(["archive/**", "2025/**"])
    assert normalize_path_patterns(ignore_path, "ignore_path") == ["archive/**", "2025/**"]


def test_normalize_path_patterns_accepts_empty_list():
    assert normalize_path_patterns([], "ignore_path") == []


def test_normalize_path_patterns_accepts_none():
    assert normalize_path_patterns(None, "include_path") is None


# ---------------------------------------------------------------------------
# filter_corpus — migrated from test_include_path.py
# ---------------------------------------------------------------------------


def _make_executor(include_patterns=None, ignore_patterns=None):
    executor = Executor.__new__(Executor)
    executor.include_path_patterns = normalize_path_patterns(include_patterns, "include_path") if include_patterns else None
    executor.ignore_path_patterns = normalize_path_patterns(ignore_patterns, "ignore_path") if ignore_patterns else None
    return executor


def test_filter_corpus_matches_any_path_against_any_pattern():
    executor = _make_executor(include_patterns=["2026/survey/**", "2026/reading-group/**"])
    corpus = [
        CorpusPaper(title="Survey Paper", abstract="", added_date=datetime(2026, 1, 1), paths=["2026/survey/topic-a", "archive/misc"]),
        CorpusPaper(title="Reading Group Paper", abstract="", added_date=datetime(2026, 1, 2), paths=["notes/inbox", "2026/reading-group/week-1"]),
        CorpusPaper(title="Excluded Paper", abstract="", added_date=datetime(2026, 1, 3), paths=["2025/other/topic"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert [p.title for p in filtered] == ["Survey Paper", "Reading Group Paper"]


def test_filter_corpus_excludes_papers_matching_ignore_path():
    executor = _make_executor(ignore_patterns=["archive/**", "2025/**"])
    corpus = [
        CorpusPaper(title="Active Paper", abstract="", added_date=datetime(2026, 1, 1), paths=["2026/survey/topic-a"]),
        CorpusPaper(title="Archived Paper", abstract="", added_date=datetime(2026, 1, 2), paths=["archive/misc"]),
        CorpusPaper(title="Old Paper", abstract="", added_date=datetime(2026, 1, 3), paths=["2025/other/topic"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert [p.title for p in filtered] == ["Active Paper"]


def test_filter_corpus_ignore_path_takes_precedence_over_include_path():
    executor = _make_executor(include_patterns=["2026/**"], ignore_patterns=["2026/ignore/**"])
    corpus = [
        CorpusPaper(title="Included Paper", abstract="", added_date=datetime(2026, 1, 1), paths=["2026/survey/topic-a"]),
        CorpusPaper(title="Ignored Paper", abstract="", added_date=datetime(2026, 1, 2), paths=["2026/ignore/topic-b"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert [p.title for p in filtered] == ["Included Paper"]


def test_filter_corpus_no_filters_returns_all():
    executor = _make_executor()
    corpus = [
        CorpusPaper(title="Paper A", abstract="", added_date=datetime(2026, 1, 1), paths=["foo"]),
        CorpusPaper(title="Paper B", abstract="", added_date=datetime(2026, 1, 2), paths=["bar"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert filtered == corpus


# ---------------------------------------------------------------------------
# fetch_zotero_corpus
# ---------------------------------------------------------------------------


def test_fetch_zotero_corpus(config, monkeypatch):
    from tests.canned_responses import make_stub_zotero_client

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    executor = Executor.__new__(Executor)
    executor.config = config
    corpus = executor.fetch_zotero_corpus()

    assert len(corpus) == 2
    assert corpus[0].title == "Stub Paper 1"
    assert "survey/topic-a" in corpus[0].paths[0]


def test_fetch_zotero_corpus_paper_with_zero_collections(config, monkeypatch):
    from tests.canned_responses import make_stub_zotero_client

    items = [
        {
            "data": {
                "title": "No Collection Paper",
                "abstractNote": "Abstract.",
                "dateAdded": "2026-03-01T00:00:00Z",
                "collections": [],
            }
        }
    ]
    stub_zot = make_stub_zotero_client(items=items)
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    executor = Executor.__new__(Executor)
    executor.config = config
    corpus = executor.fetch_zotero_corpus()

    assert len(corpus) == 1
    assert corpus[0].paths == []


def test_fetch_zotero_corpus_keeps_title_only_papers(config, monkeypatch):
    from tests.canned_responses import make_stub_zotero_client

    items = [
        {
            "data": {
                "title": "Title Only Paper",
                "abstractNote": "",
                "dateAdded": "2026-03-01T00:00:00Z",
                "collections": [],
            }
        },
        {
            "data": {
                "title": "",
                "abstractNote": "",
                "dateAdded": "2026-03-02T00:00:00Z",
                "collections": [],
            }
        },
    ]
    stub_zot = make_stub_zotero_client(items=items)
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    executor = Executor.__new__(Executor)
    executor.config = config
    corpus = executor.fetch_zotero_corpus()

    assert len(corpus) == 1
    assert corpus[0].title == "Title Only Paper"
    assert corpus[0].abstract == ""


# ---------------------------------------------------------------------------
# E2E: Executor.run()
# ---------------------------------------------------------------------------


def test_run_end_to_end(config, monkeypatch):
    """Full pipeline: Zotero fetch -> filter -> retrieve -> rerank -> TLDR -> email."""
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import (
        make_sample_corpus,
        make_sample_paper,
        make_stub_openai_client,
        make_stub_smtp,
        make_stub_zotero_client,
    )

    # Config: source=["arxiv"], reranker="api", send_empty=false
    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = False

    # 1. Stub pyzotero
    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    # 2. Stub OpenAI (for reranker + TLDR/affiliations)
    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)
    retrieved = [
        make_sample_paper(title="E2E Paper 1", score=None),
        make_sample_paper(title="E2E Paper 2", score=None),
    ]

    # Import to register the arxiv retriever
    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401

    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(
        registered_retrievers["arxiv"],
        "retrieve_papers",
        lambda self: retrieved,
    )

    # 4. Stub SMTP
    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))

    # 5. Stub sleep (reranker/retriever)
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    # 6. Run
    executor = Executor(config)
    executor.run()

    # Assertions
    assert len(sent) == 1, "Email should have been sent"
    _, _, email_body = sent[0]
    assert "text/html" in email_body


def test_run_no_papers_send_empty_false(config, monkeypatch):
    """When no papers are found and send_empty=false, no email is sent."""
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import make_stub_openai_client, make_stub_smtp, make_stub_zotero_client

    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = False

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401

    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(registered_retrievers["arxiv"], "retrieve_papers", lambda self: [])

    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    executor = Executor(config)
    executor.run()

    assert len(sent) == 0, "No email should be sent when no papers and send_empty=false"


def test_run_no_papers_send_empty_true(config, monkeypatch):
    """When no papers are found and send_empty=true, empty email is sent."""
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import make_stub_openai_client, make_stub_smtp, make_stub_zotero_client

    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = True

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401

    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(registered_retrievers["arxiv"], "retrieve_papers", lambda self: [])

    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    executor = Executor(config)
    executor.run()

    assert len(sent) == 1, "Email should be sent even with no papers when send_empty=true"
    _, _, body = sent[0]
    assert "text/html" in body


def test_run_retriever_failure_send_empty_true(config, monkeypatch):
    """A transient source failure should not fail the whole email run."""
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import make_stub_openai_client, make_stub_smtp, make_stub_zotero_client

    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = True

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401

    from zotero_arxiv_daily.retriever.base import registered_retrievers

    def _raise(self):
        raise RuntimeError("source temporarily down")

    monkeypatch.setattr(registered_retrievers["arxiv"], "retrieve_papers", _raise)

    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    executor = Executor(config)
    executor.run()

    assert len(sent) == 1, "Empty email should still be sent when source retrieval fails"
    _, _, body = sent[0]
    assert "text/html" in body


def test_run_reranker_failure_sends_unranked_papers(config, monkeypatch):
    """If reranking fails, the run should still email retrieved papers."""
    import smtplib
    from types import SimpleNamespace

    from omegaconf import open_dict

    from tests.canned_responses import (
        make_sample_paper,
        make_stub_openai_client,
        make_stub_smtp,
        make_stub_zotero_client,
    )

    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = False

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401

    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(
        registered_retrievers["arxiv"],
        "retrieve_papers",
        lambda self: [make_sample_paper(title="Retrieved Paper")],
    )

    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    executor = Executor(config)
    executor.reranker = SimpleNamespace(
        rerank=lambda papers, corpus: (_ for _ in ()).throw(RuntimeError("reranker temporarily down"))
    )
    executor.run()

    assert len(sent) == 1, "Email should still be sent when reranking fails"
    _, _, body = sent[0]
    assert "text/html" in body


def test_apply_source_quotas_keeps_rf_rss_papers(config):
    """RF journal papers should keep reserved slots when they match the focus."""
    from omegaconf import open_dict

    from tests.canned_responses import make_sample_paper

    with open_dict(config):
        config.executor.max_paper_num = 5
        config.executor.source_quotas = {
            "rf_rss": {"min": 2, "max": 3, "require_focus_match": True},
        }
        config.reranker.focus.primary_keywords = ["power amplifier"]

    executor = Executor.__new__(Executor)
    executor.config = config

    arxiv_1 = make_sample_paper(source="arxiv", title="Highly related arXiv paper", score=9.8)
    arxiv_2 = make_sample_paper(source="arxiv", title="Another arXiv paper", score=9.7)
    rf_1 = make_sample_paper(
        source="rf_rss",
        title="Doherty RF power amplifier in T-MTT",
        abstract="A power amplifier design.",
        score=6.0,
    )
    rf_2 = make_sample_paper(
        source="rf_rss",
        title="Digital predistortion for power amplifiers",
        abstract="A DPD method for RF power amplifier linearization.",
        score=5.5,
    )
    rf_3 = make_sample_paper(
        source="rf_rss",
        title="AI assisted RF power amplifier design",
        abstract="A machine learning method for RF power amplifier design.",
        score=5.0,
    )
    rf_non_focus = make_sample_paper(source="rf_rss", title="Microwave filter", abstract="A filter.", score=8.0)

    selected = executor._apply_source_quotas(
        ranked_papers=[arxiv_1, arxiv_2, rf_non_focus, rf_1, rf_2, rf_3],
        all_papers=[arxiv_1, arxiv_2, rf_1, rf_2, rf_3, rf_non_focus],
    )

    assert selected[:3] == [rf_1, rf_2, rf_3]
    assert len(selected) == 5
    assert rf_non_focus not in selected[:3]


def test_sent_history_filters_previous_and_same_run_duplicates(config, tmp_path):
    from omegaconf import open_dict

    from tests.canned_responses import make_sample_paper

    history_path = tmp_path / "sent_history.json"
    history_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "title": "Already sent",
                        "keys": ["doi:10.1109/tmtt.2026.1234567"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with open_dict(config):
        config.executor.sent_history = {
            "enabled": True,
            "path": str(history_path),
            "max_items": 20,
        }

    executor = Executor.__new__(Executor)
    executor.config = config

    already_sent = make_sample_paper(
        title="Already Sent RF Power Amplifier",
        url="https://doi.org/10.1109/TMTT.2026.1234567?utm_source=email",
    )
    arxiv_once = make_sample_paper(
        title="Fresh arXiv RF Power Amplifier",
        url="https://arxiv.org/abs/2607.01234v1",
        pdf_url="https://arxiv.org/pdf/2607.01234v1",
    )
    arxiv_duplicate = make_sample_paper(
        title="Fresh arXiv RF Power Amplifier",
        url="https://arxiv.org/pdf/2607.01234v2",
        pdf_url="https://arxiv.org/pdf/2607.01234v2",
    )
    new_paper = make_sample_paper(
        title="Fresh Doherty RF Power Amplifier",
        url="https://doi.org/10.1109/TMTT.2026.7654321",
    )

    filtered = executor._filter_sent_papers([already_sent, arxiv_once, arxiv_duplicate, new_paper])

    assert filtered == [arxiv_once, new_paper]


def test_run_skips_sent_history_and_records_new_papers(config, monkeypatch, tmp_path):
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import (
        make_sample_paper,
        make_stub_openai_client,
        make_stub_smtp,
        make_stub_zotero_client,
    )

    history_path = tmp_path / "sent_history.json"
    old_paper = make_sample_paper(
        title="Already Sent RF Power Amplifier",
        url="https://doi.org/10.1109/TMTT.2026.1111111",
    )
    new_paper = make_sample_paper(
        title="Fresh RF Power Amplifier",
        url="https://doi.org/10.1109/TMTT.2026.2222222",
    )
    history_path.write_text(
        json.dumps({"entries": [{"title": old_paper.title, "keys": _paper_identifier_keys(old_paper)}]}),
        encoding="utf-8",
    )

    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = False
        config.executor.max_paper_num = 2
        config.executor.sent_history = {
            "enabled": True,
            "path": str(history_path),
            "max_items": 20,
        }

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401

    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(registered_retrievers["arxiv"], "retrieve_papers", lambda self: [old_paper, new_paper])

    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    executor = Executor(config)
    executor.run()

    assert len(sent) == 1
    _, _, email_body = sent[0]
    decoded_body = _decode_email_body(email_body)
    assert "Fresh RF Power Amplifier" in decoded_body
    assert "Already Sent RF Power Amplifier" not in decoded_body

    history = json.loads(history_path.read_text(encoding="utf-8"))
    assert any(entry["title"] == "Fresh RF Power Amplifier" for entry in history["entries"])
