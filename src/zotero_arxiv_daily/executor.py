from loguru import logger
from pyzotero import zotero
from omegaconf import DictConfig, ListConfig
from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper, Paper, normalize_llm_base_url
import random
from datetime import datetime
import re
from .reranker import get_reranker_cls
from .construct_email import render_email
from .utils import send_email
from openai import OpenAI
from tqdm import tqdm


def normalize_path_patterns(patterns: list[str] | ListConfig | None, config_key: str) -> list[str] | None:
    if patterns is None:
        return None

    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.zotero.{config_key} must be a list of glob patterns or null, "
            'for example ["2026/survey/**"]. Single strings are not supported.'
        )

    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(f"config.zotero.{config_key} must contain only glob pattern strings.")

    return list(patterns)


def _config_get(config, key: str, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _term_matches(text: str, term: str) -> bool:
    term = str(term or "").strip().lower()
    if not term:
        return False
    if len(term) <= 3 and re.fullmatch(r"[a-z0-9]+", term):
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text))
    return term in text


class Executor:
    def __init__(self, config:DictConfig):
        self.config = config
        self.include_path_patterns = normalize_path_patterns(config.zotero.include_path, "include_path")
        self.ignore_path_patterns = normalize_path_patterns(config.zotero.ignore_path, "ignore_path")
        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in config.executor.source
        }
        self.reranker = get_reranker_cls(config.executor.reranker)(config)
        llm_base_url = normalize_llm_base_url(config.llm.api.base_url)
        self.openai_client = OpenAI(
            api_key=config.llm.api.key,
            base_url=llm_base_url or None,
            timeout=30,
            max_retries=2,
        )
    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")
        zot = zotero.Zotero(self.config.zotero.user_id, 'user', self.config.zotero.api_key)
        collections = zot.everything(zot.collections())
        collections = {c['key']:c for c in collections}
        corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
        corpus = [
            c for c in corpus
            if c['data'].get('title', '').strip() or c['data'].get('abstractNote', '').strip()
        ]
        def get_collection_path(col_key:str) -> str:
            if p := collections[col_key]['data']['parentCollection']:
                return get_collection_path(p) + '/' + collections[col_key]['data']['name']
            else:
                return collections[col_key]['data']['name']
        for c in corpus:
            paths = [get_collection_path(col) for col in c['data']['collections']]
            c['paths'] = paths
        logger.info(f"Fetched {len(corpus)} zotero papers")
        return [CorpusPaper(
            title=c['data'].get('title', ''),
            abstract=c['data'].get('abstractNote', ''),
            added_date=datetime.strptime(c['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),
            paths=c['paths']
        ) for c in corpus]
    
    def filter_corpus(self, corpus:list[CorpusPaper]) -> list[CorpusPaper]:
        if self.include_path_patterns:
            logger.info(f"Selecting zotero papers matching include_path: {self.include_path_patterns}")
            corpus = [
                c for c in corpus
                if any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.include_path_patterns
                )
            ]
        if self.ignore_path_patterns:
            logger.info(f"Excluding zotero papers matching ignore_path: {self.ignore_path_patterns}")
            corpus = [
                c for c in corpus
                if not any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.ignore_path_patterns
                )
            ]
        if self.include_path_patterns or self.ignore_path_patterns:
            samples = random.sample(corpus, min(5, len(corpus)))
            samples = '\n'.join([c.title + ' - ' + '\n'.join(c.paths) for c in samples])
            logger.info(f"Selected {len(corpus)} zotero papers:\n{samples}\n...")
        return corpus

    def _paper_focus_match_count(self, paper: Paper) -> int:
        focus_config = _config_get(self.config.reranker, "focus", {}) or {}
        primary_keywords = _config_get(focus_config, "primary_keywords", []) or []
        text = "\n".join([paper.title or "", paper.abstract or ""]).lower()
        return sum(1 for term in primary_keywords if _term_matches(text, term))

    def _source_quota_candidates(self, source: str, papers: list[Paper], quota_config) -> list[Paper]:
        require_focus_match = bool(_config_get(quota_config, "require_focus_match", True))
        candidates = [p for p in papers if p.source == source]
        if require_focus_match:
            candidates = [p for p in candidates if self._paper_focus_match_count(p) > 0]
        return sorted(
            candidates,
            key=lambda p: (
                p.score if p.score is not None else -1.0,
                p.published_date or "",
            ),
            reverse=True,
        )

    def _apply_source_quotas(self, ranked_papers: list[Paper], all_papers: list[Paper]) -> list[Paper]:
        source_quotas = _config_get(self.config.executor, "source_quotas", {}) or {}
        if not source_quotas:
            return ranked_papers

        max_paper_num = int(self.config.executor.max_paper_num)
        selected: list[Paper] = []
        selected_ids: set[int] = set()

        def add_paper(paper: Paper) -> bool:
            paper_id = id(paper)
            if paper_id in selected_ids or len(selected) >= max_paper_num:
                return False
            selected.append(paper)
            selected_ids.add(paper_id)
            return True

        for source, quota_config in dict(source_quotas).items():
            min_count = int(_config_get(quota_config, "min", 0) or 0)
            max_count = int(_config_get(quota_config, "max", min_count) or min_count)
            if min_count <= 0 and max_count <= 0:
                continue
            source_selected = 0
            for paper in self._source_quota_candidates(source, all_papers, quota_config):
                if source_selected >= max_count:
                    break
                if add_paper(paper):
                    source_selected += 1
            logger.info(f"Source quota {source}: selected {source_selected} papers (min={min_count}, max={max_count})")

        for paper in ranked_papers:
            add_paper(paper)
        return selected

    
    def run(self):
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)
        if len(corpus) == 0:
            logger.error(f"No zotero papers found. Please check your zotero settings:\n{self.config.zotero}")
            return
        all_papers = []
        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            try:
                papers = retriever.retrieve_papers()
            except Exception as exc:
                logger.exception(f"Failed to retrieve {source} papers; skip this source and continue: {exc}")
                continue
            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue
            logger.info(f"Retrieved {len(papers)} {source} papers")
            all_papers.extend(papers)
        logger.info(f"Total {len(all_papers)} papers retrieved from all sources")
        reranked_papers = []
        if len(all_papers) > 0:
            logger.info("Reranking papers...")
            try:
                reranked_papers = self.reranker.rerank(all_papers, corpus)
            except Exception as exc:
                logger.exception(f"Reranking failed; send unranked retrieved papers instead: {exc}")
                reranked_papers = all_papers
            reranked_papers = self._apply_source_quotas(reranked_papers, all_papers)
            reranked_papers = reranked_papers[:self.config.executor.max_paper_num]
            llm_kwargs = Paper._llm_generation_kwargs(self.config.llm)
            logger.info(f"Using LLM base_url={self.config.llm.api.base_url}, model={llm_kwargs.get('model')}")
            logger.info("Generating translated titles, TLDR and affiliations...")
            for p in tqdm(reranked_papers):
                p.generate_title_translation(self.openai_client, self.config.llm)
                p.generate_tldr(self.openai_client, self.config.llm)
                p.generate_affiliations(self.openai_client, self.config.llm)
        elif not self.config.executor.send_empty:
            logger.info("No new papers found. No email will be sent.")
            return
        logger.info("Sending email...")
        email_content = render_email(reranked_papers)
        send_email(self.config, email_content)
        logger.info("Email sent successfully")
