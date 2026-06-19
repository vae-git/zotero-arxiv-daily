"""Tests for BaseReranker: scoring, sorting, time decay, unknown reranker."""

import numpy as np
import pytest
from types import SimpleNamespace

from zotero_arxiv_daily.reranker.base import BaseReranker, get_reranker_cls
from tests.canned_responses import make_sample_paper, make_sample_corpus


class StubReranker(BaseReranker):
    """Reranker with a controlled similarity matrix for deterministic tests."""

    def __init__(self, sim_matrix: np.ndarray, config=None):
        self.config = config
        self._sim = sim_matrix
        self.seen_s1 = None
        self.seen_s2 = None

    def get_similarity_score(self, s1, s2):
        self.seen_s1 = s1
        self.seen_s2 = s2
        return self._sim


def test_rerank_scores_and_sorts():
    corpus = make_sample_corpus(3)
    papers = [make_sample_paper(title=f"Paper {i}") for i in range(2)]

    # Paper 1 has higher similarity to all corpus papers
    sim = np.array([
        [0.1, 0.1, 0.1],  # paper 0 — low
        [0.9, 0.9, 0.9],  # paper 1 — high
    ])
    reranker = StubReranker(sim)
    ranked = reranker.rerank(papers, corpus)
    assert ranked[0].title == "Paper 1"
    assert ranked[1].title == "Paper 0"
    assert ranked[0].score > ranked[1].score


def test_rerank_time_decay_weighting():
    corpus = make_sample_corpus(3)
    papers = [make_sample_paper(title="P")]

    # Only similar to the oldest paper (index 2 after reverse-sort by date)
    sim = np.array([[0.0, 0.0, 1.0]])
    reranker = StubReranker(sim)
    ranked_old = reranker.rerank(papers, corpus)
    score_old = ranked_old[0].score

    # Only similar to the newest paper (index 0 after reverse-sort by date)
    papers2 = [make_sample_paper(title="P")]
    sim2 = np.array([[1.0, 0.0, 0.0]])
    reranker2 = StubReranker(sim2)
    ranked_new = reranker2.rerank(papers2, corpus)
    score_new = ranked_new[0].score

    # Newest corpus paper gets higher time-decay weight, so score should be higher
    assert score_new > score_old


def test_rerank_single_candidate_single_corpus():
    corpus = make_sample_corpus(1)
    papers = [make_sample_paper()]
    sim = np.array([[0.5]])
    reranker = StubReranker(sim)
    ranked = reranker.rerank(papers, corpus)
    assert len(ranked) == 1
    assert ranked[0].score is not None


def test_rerank_uses_title_and_abstract_text():
    corpus = make_sample_corpus(1)
    papers = [make_sample_paper(title="Candidate Title", abstract="Candidate abstract.")]
    corpus[0].title = "Corpus Title"
    corpus[0].abstract = "Corpus abstract."
    sim = np.array([[0.5]])

    reranker = StubReranker(sim)
    reranker.rerank(papers, corpus)

    assert reranker.seen_s1 == ["Title: Candidate Title\nAbstract: Candidate abstract."]
    assert reranker.seen_s2 == ["Title: Corpus Title\nAbstract: Corpus abstract."]


def test_rerank_top_k_prefers_strong_nearest_match():
    corpus = make_sample_corpus(25)
    papers = [
        make_sample_paper(title="Strong nearest match"),
        make_sample_paper(title="Broad weak match"),
    ]
    config = SimpleNamespace(reranker={"top_k": 1})

    sim = np.array([
        [1.0] + [0.0] * 24,
        [0.2] * 25,
    ])
    reranker = StubReranker(sim, config=config)
    ranked = reranker.rerank(papers, corpus)

    assert ranked[0].title == "Strong nearest match"
    assert ranked[0].score > ranked[1].score


def test_get_reranker_cls_unknown():
    with pytest.raises(ValueError, match="not found"):
        get_reranker_cls("nonexistent_reranker_xyz")
