from abc import ABC, abstractmethod
from omegaconf import DictConfig
from ..protocol import Paper, CorpusPaper
import numpy as np
from typing import Type
import re


DEFAULT_TOP_K = 5
DEFAULT_NEAREST_WEIGHT = 0.7
DEFAULT_FOCUS_PRIMARY_BOOST = 0.12
DEFAULT_FOCUS_SECONDARY_BOOST = 0.04
DEFAULT_FOCUS_AI_COMBO_BOOST = 0.35
DEFAULT_FOCUS_MAX_BOOST = 0.9
DEFAULT_FOCUS_NO_PRIMARY_PENALTY = 0.35


class BaseReranker(ABC):
    def __init__(self, config:DictConfig):
        self.config = config

    def rerank(self, candidates:list[Paper], corpus:list[CorpusPaper]) -> list[Paper]:
        corpus = sorted(corpus,key=lambda x: x.added_date,reverse=True)
        time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
        time_decay_weight: np.ndarray = time_decay_weight / time_decay_weight.sum()
        sim = self.get_similarity_score(
            [self._candidate_text(c) for c in candidates],
            [self._corpus_text(c) for c in corpus],
        )
        assert sim.shape == (len(candidates), len(corpus))
        scores = self._aggregate_scores(sim, time_decay_weight) * 10 # [n_candidate]
        for s,c in zip(scores,candidates):
            c.score = s * self._focus_multiplier(c)
        candidates = self._filter_by_focus(candidates)
        candidates = sorted(candidates,key=lambda x: x.score,reverse=True)
        return candidates

    def _filter_by_focus(self, candidates: list[Paper]) -> list[Paper]:
        focus_config = self._get_reranker_config_value("focus")
        if (
            not focus_config
            or self._focus_config_get(focus_config, "enabled", False) is False
            or not self._focus_config_get(focus_config, "drop_without_primary", False)
        ):
            return candidates
        return [c for c in candidates if self._primary_focus_matches(c) > 0]

    def _focus_multiplier(self, paper: Paper) -> float:
        focus_config = self._get_reranker_config_value("focus")
        if not focus_config or self._focus_config_get(focus_config, "enabled", False) is False:
            return 1.0

        text = self._candidate_text(paper).lower()
        primary_matches = self._primary_focus_matches(paper, focus_config)
        secondary_matches = self._count_term_matches(text, self._focus_config_get(focus_config, "secondary_keywords", []))
        ai_matches = self._count_term_matches(text, self._focus_config_get(focus_config, "ai_keywords", []))

        if primary_matches == 0:
            return float(self._focus_config_get(focus_config, "no_primary_penalty", DEFAULT_FOCUS_NO_PRIMARY_PENALTY))

        boost = (
            primary_matches * float(self._focus_config_get(focus_config, "primary_boost_per_match", DEFAULT_FOCUS_PRIMARY_BOOST))
            + secondary_matches * float(self._focus_config_get(focus_config, "secondary_boost_per_match", DEFAULT_FOCUS_SECONDARY_BOOST))
        )
        if ai_matches > 0:
            boost += float(self._focus_config_get(focus_config, "ai_combo_boost", DEFAULT_FOCUS_AI_COMBO_BOOST))
        max_boost = float(self._focus_config_get(focus_config, "max_boost", DEFAULT_FOCUS_MAX_BOOST))
        return 1.0 + min(boost, max_boost)

    def _primary_focus_matches(self, paper: Paper, focus_config=None) -> int:
        focus_config = focus_config or self._get_reranker_config_value("focus")
        if not focus_config:
            return 0
        text = self._candidate_text(paper).lower()
        return self._count_term_matches(text, self._focus_config_get(focus_config, "primary_keywords", []))

    @staticmethod
    def _focus_config_get(config, key: str, default=None):
        if hasattr(config, "get"):
            return config.get(key, default)
        return getattr(config, key, default)

    @classmethod
    def _count_term_matches(cls, text: str, terms) -> int:
        return sum(1 for term in terms if cls._term_matches(text, str(term).lower()))

    @staticmethod
    def _term_matches(text: str, term: str) -> bool:
        term = term.strip()
        if not term:
            return False
        if len(term) <= 3 and re.fullmatch(r"[a-z0-9]+", term):
            return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text))
        return term in text

    def _aggregate_scores(self, sim: np.ndarray, time_decay_weight: np.ndarray) -> np.ndarray:
        top_k = self._get_top_k(sim.shape[1])
        nearest_weight = self._get_nearest_weight()
        if top_k >= sim.shape[1]:
            top_indices = np.tile(np.arange(sim.shape[1]), (sim.shape[0], 1))
        else:
            top_indices = np.argpartition(sim, -top_k, axis=1)[:, -top_k:]
        scores = []
        for row, indices in enumerate(top_indices):
            selected_weight = time_decay_weight[indices]
            selected_weight = selected_weight / selected_weight.sum()
            weighted_top_score = (sim[row, indices] * selected_weight).sum()
            nearest_score = sim[row, indices].max()
            scores.append(nearest_weight * nearest_score + (1 - nearest_weight) * weighted_top_score)
        return np.array(scores)

    def _get_top_k(self, corpus_size: int) -> int:
        top_k = self._get_reranker_config_value("top_k")
        if top_k is None:
            top_k = corpus_size if self.config is None else DEFAULT_TOP_K

        top_k = int(top_k)
        if top_k <= 0:
            return corpus_size
        return min(top_k, corpus_size)

    def _get_nearest_weight(self) -> float:
        nearest_weight = self._get_reranker_config_value("nearest_weight")
        if nearest_weight is None:
            nearest_weight = 0 if self.config is None else DEFAULT_NEAREST_WEIGHT
        nearest_weight = float(nearest_weight)
        return min(max(nearest_weight, 0.0), 1.0)

    def _get_reranker_config_value(self, key: str):
        if self.config is None or getattr(self.config, "reranker", None) is None:
            return None
        reranker_config = self.config.reranker
        if hasattr(reranker_config, "get"):
            return reranker_config.get(key)
        return getattr(reranker_config, key, None)

    @staticmethod
    def _candidate_text(paper: Paper) -> str:
        return BaseReranker._paper_text(paper.title, paper.abstract)

    @staticmethod
    def _corpus_text(paper: CorpusPaper) -> str:
        return BaseReranker._paper_text(paper.title, paper.abstract)

    @staticmethod
    def _paper_text(title: str, abstract: str) -> str:
        parts = []
        if title:
            parts.append(f"Title: {title}")
        if abstract:
            parts.append(f"Abstract: {abstract}")
        return "\n".join(parts)
    
    @abstractmethod
    def get_similarity_score(self, s1:list[str], s2:list[str]) -> np.ndarray:
        raise NotImplementedError

registered_rerankers = {}

def register_reranker(name:str):
    def decorator(cls):
        registered_rerankers[name] = cls
        return cls
    return decorator

def get_reranker_cls(name:str) -> Type[BaseReranker]:
    if name not in registered_rerankers:
        raise ValueError(f"Reranker {name} not found")
    return registered_rerankers[name]
