"""Tests for abstract_enricher: inverted-index restore, fallback chain, degradation."""

from types import SimpleNamespace

import pytest
import requests

import zotero_arxiv_daily.retriever.abstract_enricher as enricher
from zotero_arxiv_daily.retriever.abstract_enricher import (
    enrich_abstract,
    extract_doi,
    restore_inverted_abstract,
)

LONG_ABSTRACT = (
    "This paper presents a broadband Doherty power amplifier that achieves high "
    "efficiency and linearity across the entire operating band."
)
DOI = "10.1109/tmtt.2026.1234567"


def _to_inverted_index(text: str) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for position, word in enumerate(text.split()):
        index.setdefault(word, []).append(position)
    return index


def _make_response(payload: dict, status: int = 200):
    class StubResponse:
        def raise_for_status(self):
            if status >= 400:
                raise requests.HTTPError(response=SimpleNamespace(status_code=status))

        def json(self):
            return payload

    return StubResponse()


@pytest.fixture(autouse=True)
def _clear_cache():
    """enrich_abstract is lru_cached; keep tests independent of each other."""
    enrich_abstract.cache_clear()
    yield
    enrich_abstract.cache_clear()


def test_restore_inverted_abstract_recovers_word_order():
    text = "a broadband Doherty power amplifier"
    assert restore_inverted_abstract(_to_inverted_index(text)) == text


def test_restore_inverted_abstract_handles_repeated_words():
    text = "the amplifier and the network"
    assert restore_inverted_abstract(_to_inverted_index(text)) == text


@pytest.mark.parametrize("value", [None, {}])
def test_restore_inverted_abstract_empty(value):
    assert restore_inverted_abstract(value) is None


@pytest.mark.parametrize(
    "value,expected",
    [
        ("10.1109/TMTT.2026.1234567", "10.1109/tmtt.2026.1234567"),
        ("https://doi.org/10.1109/TMTT.2026.1234567", "10.1109/tmtt.2026.1234567"),
        ("https://ieeexplore.ieee.org/document/12345", None),
        ("", None),
        (None, None),
    ],
)
def test_extract_doi(value, expected):
    assert extract_doi(value) == expected


def test_enrich_abstract_uses_openalex(monkeypatch):
    calls = []

    def stub_get(url, params=None, headers=None, timeout=None):
        calls.append(url)
        return _make_response({"abstract_inverted_index": _to_inverted_index(LONG_ABSTRACT)})

    monkeypatch.setattr(enricher.requests, "get", stub_get)

    assert enrich_abstract(DOI) == LONG_ABSTRACT
    assert len(calls) == 1
    assert "openalex" in calls[0]


def test_enrich_abstract_falls_back_to_semantic_scholar(monkeypatch):
    calls = []

    def stub_get(url, params=None, headers=None, timeout=None):
        calls.append(url)
        if "openalex" in url:
            return _make_response({"abstract_inverted_index": None})
        return _make_response({"abstract": LONG_ABSTRACT})

    monkeypatch.setattr(enricher.requests, "get", stub_get)

    assert enrich_abstract(DOI) == LONG_ABSTRACT
    assert len(calls) == 2
    assert "semanticscholar" in calls[1]


def test_enrich_abstract_rejects_too_short_result(monkeypatch):
    def stub_get(url, params=None, headers=None, timeout=None):
        if "openalex" in url:
            return _make_response({"abstract_inverted_index": _to_inverted_index("Too short.")})
        return _make_response({"abstract": ""})

    monkeypatch.setattr(enricher.requests, "get", stub_get)

    assert enrich_abstract(DOI) is None


def test_enrich_abstract_strips_markup(monkeypatch):
    def stub_get(url, params=None, headers=None, timeout=None):
        if "openalex" in url:
            return _make_response({"abstract_inverted_index": None})
        return _make_response({"abstract": f"<jats:p>{LONG_ABSTRACT}</jats:p>"})

    monkeypatch.setattr(enricher.requests, "get", stub_get)

    assert enrich_abstract(DOI) == LONG_ABSTRACT


def test_enrich_abstract_survives_network_failure(monkeypatch):
    def stub_get(url, params=None, headers=None, timeout=None):
        raise requests.ConnectionError("network down")

    monkeypatch.setattr(enricher.requests, "get", stub_get)

    # 补摘要失败不能中断当天的运行，只能安静降级。
    assert enrich_abstract(DOI) is None


def test_enrich_abstract_survives_http_404(monkeypatch):
    def stub_get(url, params=None, headers=None, timeout=None):
        return _make_response({}, status=404)

    monkeypatch.setattr(enricher.requests, "get", stub_get)

    assert enrich_abstract(DOI) is None


def test_enrich_abstract_without_doi_makes_no_request(monkeypatch):
    def stub_get(*args, **kwargs):
        raise AssertionError("should not perform a request without a DOI")

    monkeypatch.setattr(enricher.requests, "get", stub_get)

    assert enrich_abstract("https://ieeexplore.ieee.org/document/12345") is None
    assert enrich_abstract(None) is None


def test_enrich_abstract_is_cached(monkeypatch):
    calls = []

    def stub_get(url, params=None, headers=None, timeout=None):
        calls.append(url)
        return _make_response({"abstract_inverted_index": _to_inverted_index(LONG_ABSTRACT)})

    monkeypatch.setattr(enricher.requests, "get", stub_get)

    enrich_abstract(DOI)
    enrich_abstract(DOI)
    assert len(calls) == 1
