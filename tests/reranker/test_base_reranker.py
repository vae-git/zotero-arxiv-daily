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


def test_rerank_focus_boosts_ai_power_amplifier_papers():
    corpus = make_sample_corpus(2)
    papers = [
        make_sample_paper(
            title="Generic microwave antenna design",
            abstract="This paper studies an antenna for wireless systems.",
        ),
        make_sample_paper(
            title="Deep learning digital predistortion for RF power amplifiers",
            abstract="A neural network improves PA linearization and efficiency.",
        ),
    ]
    config = SimpleNamespace(reranker={
        "top_k": 2,
        "nearest_weight": 0.7,
        "focus": {
            "enabled": True,
            "primary_keywords": ["power amplifier", "digital predistortion", "pa"],
            "secondary_keywords": ["efficiency", "linearization"],
            "ai_keywords": ["deep learning", "neural network"],
            "drop_without_primary": False,
        },
    })

    sim = np.array([
        [0.6, 0.6],
        [0.6, 0.6],
    ])
    reranker = StubReranker(sim, config=config)
    ranked = reranker.rerank(papers, corpus)

    assert ranked[0].title == "Deep learning digital predistortion for RF power amplifiers"
    assert ranked[0].score > ranked[1].score


def test_rerank_focus_can_drop_non_power_amplifier_papers():
    corpus = make_sample_corpus(1)
    papers = [
        make_sample_paper(title="Generic microwave antenna design", abstract="Wireless antenna study."),
        make_sample_paper(title="Doherty power amplifier linearization", abstract="A PA efficiency method."),
    ]
    config = SimpleNamespace(reranker={
        "top_k": 1,
        "focus": {
            "enabled": True,
            "primary_keywords": ["power amplifier", "doherty", "pa"],
            "drop_without_primary": True,
        },
    })

    sim = np.array([[0.9], [0.5]])
    reranker = StubReranker(sim, config=config)
    ranked = reranker.rerank(papers, corpus)

    assert [p.title for p in ranked] == ["Doherty power amplifier linearization"]


def test_get_reranker_cls_unknown():
    with pytest.raises(ValueError, match="not found"):
        get_reranker_cls("nonexistent_reranker_xyz")


def _tiered_focus_config():
    return SimpleNamespace(reranker={
        "top_k": 1,
        "nearest_weight": 0.7,
        "focus": {
            "enabled": True,
            "primary_keywords": ["power amplifier", "doherty"],
            "domain_keywords": ["antenna", "microwave", "mmic"],
            "primary_boost_per_match": 0.14,
            "domain_boost_per_match": 0.04,
            "domain_only_base": 0.8,
            "domain_max_boost": 0.2,
            "no_primary_penalty": 0.35,
            "drop_without_primary": False,
        },
    })


def test_rerank_tiers_primary_above_domain_above_unrelated():
    corpus = make_sample_corpus(1)
    papers = [
        make_sample_paper(title="A study of protein folding", abstract="Molecular biology."),
        make_sample_paper(title="A microwave antenna array", abstract="An antenna for radar."),
        make_sample_paper(title="A Doherty power amplifier", abstract="High efficiency design."),
    ]
    # 相似度完全相同，排序差异只能来自 focus 分档。
    sim = np.array([[0.5], [0.5], [0.5]])

    reranker = StubReranker(sim, config=_tiered_focus_config())
    ranked = reranker.rerank(papers, corpus)

    assert [p.title for p in ranked] == [
        "A Doherty power amplifier",
        "A microwave antenna array",
        "A study of protein folding",
    ]


def test_domain_only_paper_is_kept_but_ranked_below_primary():
    corpus = make_sample_corpus(1)
    papers = [make_sample_paper(title="A microwave antenna array", abstract="An antenna for radar.")]
    sim = np.array([[0.5]])

    reranker = StubReranker(sim, config=_tiered_focus_config())
    ranked = reranker.rerank(papers, corpus)

    # domain-only 的倍率固定压在 1.0 以下，确保永远排在命中 primary 的论文之后。
    assert len(ranked) == 1
    assert ranked[0].score < 0.5 * 10


def test_unrelated_paper_is_penalised_harder_than_domain_paper():
    corpus = make_sample_corpus(1)
    domain_paper = make_sample_paper(title="A microwave antenna array", abstract="An antenna.")
    unrelated_paper = make_sample_paper(title="A study of protein folding", abstract="Biology.")
    sim = np.array([[0.5], [0.5]])

    reranker = StubReranker(sim, config=_tiered_focus_config())
    ranked = reranker.rerank([domain_paper, unrelated_paper], corpus)

    assert ranked[0].title == "A microwave antenna array"
    assert ranked[0].score > ranked[1].score


def test_power_allocation_paper_is_not_treated_as_power_amplifier():
    """The bare "pa" keyword used to match "Power Allocation" in eess.SP papers."""
    corpus = make_sample_corpus(1)
    papers = [
        make_sample_paper(
            title="Robust Beamforming and Power Allocation for Cell-Free Massive MIMO",
            abstract="We optimise PA across users to maximise throughput.",
        ),
    ]
    config = SimpleNamespace(reranker={
        "top_k": 1,
        "focus": {
            "enabled": True,
            # 当前生产配置不再收录裸 "pa"。
            "primary_keywords": ["power amplifier", "doherty", "digital predistortion"],
            "domain_keywords": [],
            "no_primary_penalty": 0.35,
            "drop_without_primary": True,
        },
    })

    sim = np.array([[0.9]])
    reranker = StubReranker(sim, config=config)

    assert reranker.rerank(papers, corpus) == []
