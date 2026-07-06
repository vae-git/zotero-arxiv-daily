from loguru import logger
from pyzotero import zotero
from omegaconf import DictConfig, ListConfig
from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper, Paper, normalize_llm_base_url
import random
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlparse
from .reranker import get_reranker_cls
from .construct_email import render_email
from .utils import send_email
from openai import OpenAI
from tqdm import tqdm


DEFAULT_SENT_HISTORY_PATH = ".cache/zotero-arxiv-daily/sent_history.json"
DEFAULT_SENT_HISTORY_MAX_ITEMS = 1000
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s<>\"'\?#]+", re.IGNORECASE)
ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)


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


def _config_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _term_matches(text: str, term: str) -> bool:
    term = str(term or "").strip().lower()
    if not term:
        return False
    if len(term) <= 3 and re.fullmatch(r"[a-z0-9]+", term):
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text))
    return term in text


def _normalize_title_for_history(title: str | None) -> str:
    title = re.sub(r"^\[[^\]]+\]\s*", "", str(title or ""))
    title = re.sub(r"[\W_]+", " ", title.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", title).strip()


def _extract_doi(value: str | None) -> str | None:
    value = unquote(str(value or ""))
    parsed = urlparse(value)
    candidates = [value]
    if "doi.org" in parsed.netloc.lower() and parsed.path:
        candidates.insert(0, parsed.path.lstrip("/"))
    for candidate in candidates:
        match = DOI_RE.search(candidate)
        if match:
            return match.group(0).rstrip(".,);]").lower()
    return None


def _paper_identifier_keys(paper: Paper) -> list[str]:
    doi_keys: list[str] = []
    arxiv_keys: list[str] = []
    url_keys: list[str] = []

    def add_key(keys: list[str], key: str | None):
        if key and key not in keys:
            keys.append(key)

    for value in (paper.url, paper.pdf_url):
        doi = _extract_doi(value)
        if doi:
            add_key(doi_keys, f"doi:{doi}")
            continue

        value = str(value or "").strip()
        if not value:
            continue
        parsed = urlparse(value)
        normalized_path = parsed.path.rstrip("/")
        if "arxiv.org" in parsed.netloc.lower():
            arxiv_match = ARXIV_ID_RE.search(normalized_path)
            if arxiv_match:
                add_key(arxiv_keys, f"arxiv:{arxiv_match.group(1).lower()}")
                continue
        normalized_url = parsed._replace(query="", fragment="", path=normalized_path).geturl().lower()
        add_key(url_keys, f"url:{normalized_url}")

    keys = doi_keys or arxiv_keys or url_keys

    title_key = _normalize_title_for_history(paper.title)
    if title_key:
        title_hash = hashlib.sha1(title_key.encode("utf-8")).hexdigest()[:20]
        add_key(keys, f"title:{title_hash}")
    return keys


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

    def _sent_history_config(self):
        return _config_get(self.config.executor, "sent_history", {}) or {}

    def _sent_history_enabled(self) -> bool:
        return _config_bool(_config_get(self._sent_history_config(), "enabled", False), default=False)

    def _sent_history_path(self) -> Path:
        history_path = _config_get(self._sent_history_config(), "path", DEFAULT_SENT_HISTORY_PATH)
        return Path(str(history_path or DEFAULT_SENT_HISTORY_PATH)).expanduser()

    def _sent_history_max_items(self) -> int:
        try:
            return max(1, int(_config_get(self._sent_history_config(), "max_items", DEFAULT_SENT_HISTORY_MAX_ITEMS)))
        except (TypeError, ValueError):
            return DEFAULT_SENT_HISTORY_MAX_ITEMS

    def _load_sent_history(self) -> dict:
        if not self._sent_history_enabled():
            return {"entries": []}

        history_path = self._sent_history_path()
        if not history_path.exists():
            return {"entries": []}

        try:
            data = json.loads(history_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Failed to load sent history from {history_path}: {exc}")
            return {"entries": []}

        entries = data if isinstance(data, list) else data.get("entries", [])
        if not isinstance(entries, list):
            entries = []
        return {"entries": [entry for entry in entries if isinstance(entry, dict)]}

    @staticmethod
    def _sent_history_keys(history: dict) -> set[str]:
        keys: set[str] = set()
        for entry in history.get("entries", []):
            entry_keys = entry.get("keys", [])
            if isinstance(entry_keys, str):
                entry_keys = [entry_keys]
            for key in entry_keys:
                if key:
                    keys.add(str(key))
        return keys

    def _filter_sent_papers(self, papers: list[Paper]) -> list[Paper]:
        if not self._sent_history_enabled():
            return papers

        sent_keys = self._sent_history_keys(self._load_sent_history())
        run_keys: set[str] = set()
        filtered: list[Paper] = []
        skipped_history = 0
        skipped_run_duplicate = 0

        for paper in papers:
            keys = set(_paper_identifier_keys(paper))
            if keys and keys.intersection(sent_keys):
                skipped_history += 1
                continue
            if keys and keys.intersection(run_keys):
                skipped_run_duplicate += 1
                continue
            filtered.append(paper)
            run_keys.update(keys)

        logger.info(
            "Sent-history filter kept "
            f"{len(filtered)}/{len(papers)} papers "
            f"(skipped_history={skipped_history}, skipped_run_duplicate={skipped_run_duplicate})"
        )
        return filtered

    def _record_sent_papers(self, papers: list[Paper]) -> None:
        if not self._sent_history_enabled() or not papers:
            return

        history = self._load_sent_history()
        entries = history.get("entries", [])
        known_keys = self._sent_history_keys(history)
        sent_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        added = 0

        for paper in papers:
            keys = _paper_identifier_keys(paper)
            if not keys or known_keys.intersection(keys):
                continue
            entries.append(
                {
                    "sent_at": sent_at,
                    "source": paper.source,
                    "title": paper.title,
                    "url": paper.url,
                    "published_date": paper.published_date,
                    "keys": keys,
                }
            )
            known_keys.update(keys)
            added += 1

        if added == 0:
            logger.info("No new sent-history entries to record")
            return

        max_items = self._sent_history_max_items()
        history = {"entries": entries[-max_items:]}
        history_path = self._sent_history_path()
        try:
            history_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = history_path.with_suffix(history_path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_path.replace(history_path)
            logger.info(f"Recorded {added} sent papers to {history_path}")
        except Exception as exc:
            logger.warning(f"Failed to record sent history to {history_path}: {exc}")

    
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
        all_papers = self._filter_sent_papers(all_papers)
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
        self._record_sent_papers(reranked_papers)
