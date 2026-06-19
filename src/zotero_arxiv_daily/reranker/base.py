from abc import ABC, abstractmethod
from omegaconf import DictConfig
from ..protocol import Paper, CorpusPaper
import numpy as np
from typing import Type


DEFAULT_TOP_K = 5
DEFAULT_NEAREST_WEIGHT = 0.7


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
            c.score = s
        candidates = sorted(candidates,key=lambda x: x.score,reverse=True)
        return candidates

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
