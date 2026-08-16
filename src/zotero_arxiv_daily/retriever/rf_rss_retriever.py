from typing import Any

import feedparser
import re
import requests
from datetime import date, timedelta
from html import unescape
from loguru import logger

from ..protocol import Paper
from .abstract_enricher import enrich_abstract
from .base import BaseRetriever, register_retriever
from .date_utils import format_published_date


DEFAULT_RF_RSS_FEEDS = {
    "T-MTT": "https://ieeexplore.ieee.org/rss/TOC22.xml",
    "T-AP": "https://ieeexplore.ieee.org/rss/TOC8.xml",
    "MWTL": "https://ieeexplore.ieee.org/rss/TOC7260.xml",
    "JSSC": "https://ieeexplore.ieee.org/rss/TOC4.xml",
}

DEFAULT_RF_CROSSREF_ISSN = {
    "T-MTT": "0018-9480",
    "T-AP": "0018-926X",
    "MWTL": "2771-957X",
    "JSSC": "0018-9200",
}

DEFAULT_CROSSREF_QUERIES = [
    "RF power amplifier",
    "power amplifier",
    "digital predistortion",
    "Doherty",
    "linearization",
    "envelope tracking",
    "outphasing",
    "machine learning power amplifier",
    "neural network power amplifier",
    "power amplifier behavior modeling",
]

DEFAULT_RF_RSS_VENUE_INFO = {
    "T-MTT": {
        "venue": "IEEE Transactions on Microwave Theory and Techniques",
        "rank": "Top RF/microwave journal / \u5c04\u9891\u5fae\u6ce2\u9886\u57df\u9876\u7ea7\u671f\u520a",
        "cas_partition": "\u4e2d\u79d1\u9662 2 \u533a\uff08\u5de5\u7a0b\u6280\u672f\uff0c\u9ed8\u8ba4\u53c2\u8003\uff09",
        "sci_quartile": "JCR Q1 / SCI \u4e00\u533a",
    },
    "T-AP": {
        "venue": "IEEE Transactions on Antennas and Propagation",
        "rank": "Top antennas/propagation journal / \u5929\u7ebf\u4e0e\u4f20\u64ad\u9886\u57df\u9876\u7ea7\u671f\u520a",
        "cas_partition": "\u4e2d\u79d1\u9662 2 \u533a\uff08\u5de5\u7a0b\u6280\u672f\uff0c\u9ed8\u8ba4\u53c2\u8003\uff09",
        "sci_quartile": "JCR Q1 / SCI \u4e00\u533a",
    },
    "MWTL": {
        "venue": "IEEE Microwave and Wireless Technology Letters",
        "rank": "IEEE RF/microwave letters journal / IEEE \u5c04\u9891\u5fae\u6ce2\u5feb\u62a5\u671f\u520a",
        "cas_partition": "\u4e2d\u79d1\u9662 3 \u533a\uff08\u5de5\u7a0b\u6280\u672f\uff0c\u9ed8\u8ba4\u53c2\u8003\uff09",
        "sci_quartile": "JCR Q2 / SCI \u4e8c\u533a",
    },
    "JSSC": {
        "venue": "IEEE Journal of Solid-State Circuits",
        "rank": "Top integrated circuits journal / \u96c6\u6210\u7535\u8def\u9886\u57df\u9876\u7ea7\u671f\u520a",
        "cas_partition": "\u4e2d\u79d1\u9662 1 \u533a\uff08\u5de5\u7a0b\u6280\u672f\uff0c\u9ed8\u8ba4\u53c2\u8003\uff09",
        "sci_quartile": "JCR Q1 / SCI \u4e00\u533a",
    },
}

CROSSREF_TIMEOUT = 30
# Crossref 单次最多可取 1000 条，因此一次请求即可覆盖 max_entries，无需翻页。
DEFAULT_LOOKBACK_DAYS = 120
DEFAULT_MAX_CROSSREF_QUERY_ENTRIES = 8
CROSSREF_NON_ARTICLE_TITLE_PATTERNS = (
    "table of contents",
    "front cover",
    "back cover",
    "cover",
    "editorial",
    "erratum",
    "correction",
    "comments on",
    "reply to",
)


@register_retriever("rf_rss")
class RfRssRetriever(BaseRetriever):
    def _get_venue_info(self, journal: str) -> tuple[str, str, str, str]:
        venue_info = self.retriever_config.get("venue_info") or DEFAULT_RF_RSS_VENUE_INFO
        info = dict(venue_info).get(journal, {})
        if hasattr(info, "get"):
            venue = info.get("venue") or journal
            rank = info.get("rank") or "Unknown"
            cas_partition = info.get("cas_partition") or "Unknown"
            sci_quartile = info.get("sci_quartile") or "Unknown"
        else:
            venue = journal
            rank = str(info or "Unknown")
            cas_partition = "Unknown"
            sci_quartile = "Unknown"
        return str(venue), str(rank), str(cas_partition), str(sci_quartile)

    @staticmethod
    def _clean_crossref_text(text: str | None) -> str:
        text = unescape(str(text or ""))
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @classmethod
    def _is_research_crossref_item(cls, item: dict[str, Any]) -> bool:
        title_values = item.get("title") or []
        title = cls._clean_crossref_text(title_values[0] if title_values else "").lower()
        if not title:
            return False
        return not any(
            re.search(rf"(?<![a-z0-9]){re.escape(pattern)}(?![a-z0-9])", title)
            for pattern in CROSSREF_NON_ARTICLE_TITLE_PATTERNS
        )

    @staticmethod
    def _crossref_date(item: dict[str, Any]) -> str | None:
        for key in ("published-print", "published-online", "published", "created", "issued"):
            date_parts = item.get(key, {}).get("date-parts")
            if date_parts and date_parts[0]:
                parts = [int(p) for p in date_parts[0]]
                year = parts[0]
                month = parts[1] if len(parts) > 1 else 1
                day = parts[2] if len(parts) > 2 else 1
                return f"{year:04d}-{month:02d}-{day:02d}"
        return None

    def _lookback_filter(self) -> str:
        """Restrict Crossref results to a recent window so old issues drop out."""
        try:
            days = int(self.retriever_config.get("lookback_days") or DEFAULT_LOOKBACK_DAYS)
        except (TypeError, ValueError):
            days = DEFAULT_LOOKBACK_DAYS
        days = max(days, 1)
        return f"from-pub-date:{(date.today() - timedelta(days=days)).isoformat()}"

    def _crossref_sources(self) -> dict[str, dict[str, Any]]:
        """Normalise the legacy ``crossref_issn`` map and the richer ``crossref_sources``.

        ``crossref_sources`` entries carry a ``kind`` telling us how to query:
        ``journal`` (by ISSN), ``proceedings`` (by container title) or ``prefix``
        (by DOI prefix, used for preprint servers such as TechRxiv).
        """
        sources: dict[str, dict[str, Any]] = {}
        legacy_issn = self.retriever_config.get("crossref_issn") or DEFAULT_RF_CROSSREF_ISSN
        for name, issn in dict(legacy_issn).items():
            if issn:
                sources[name] = {"kind": "journal", "issn": str(issn)}

        for name, spec in dict(self.retriever_config.get("crossref_sources") or {}).items():
            spec = dict(spec) if hasattr(spec, "keys") else {}
            if spec:
                spec.setdefault("kind", "journal")
                sources[name] = spec
        return sources

    def _crossref_request(self, spec: dict[str, Any], rows: int, query_title: str | None) -> list[dict[str, Any]]:
        kind = str(spec.get("kind", "journal"))
        params: dict[str, Any] = {
            "sort": "published",
            "order": "desc",
            "rows": rows,
            "select": "DOI,URL,title,author,abstract,published,published-print,published-online,issued,created",
        }
        filters = [self._lookback_filter()]

        if kind == "journal":
            issn = spec.get("issn")
            if not issn:
                return []
            url = f"https://api.crossref.org/journals/{issn}/works"
            filters.append("type:journal-article")
        elif kind == "proceedings":
            container = spec.get("container")
            if not container:
                return []
            url = "https://api.crossref.org/works"
            filters.append("type:proceedings-article")
            params["query.container-title"] = container
        elif kind == "prefix":
            prefix = spec.get("prefix")
            if not prefix:
                return []
            url = f"https://api.crossref.org/prefixes/{prefix}/works"
        else:
            logger.warning(f"Unknown Crossref source kind {kind!r}; skipping")
            return []

        params["filter"] = ",".join(filters)
        if query_title:
            params["query.title"] = query_title

        response = requests.get(
            url,
            params=params,
            headers={"User-Agent": "zotero-arxiv-daily/1.0 (mailto:no-reply@example.com)"},
            timeout=CROSSREF_TIMEOUT,
        )
        response.raise_for_status()
        return response.json().get("message", {}).get("items", [])

    def _fetch_crossref_items(self, issn: str, rows: int, query_title: str | None = None) -> list[dict[str, Any]]:
        """Backwards-compatible helper for querying a journal by ISSN."""
        return self._crossref_request({"kind": "journal", "issn": issn}, rows, query_title)

    def _retrieve_crossref_entries(
        self,
        journal: str,
        max_entries: int,
        spec: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if spec is None:
            crossref_issn = self.retriever_config.get("crossref_issn") or DEFAULT_RF_CROSSREF_ISSN
            issn = dict(crossref_issn).get(journal)
            if not issn:
                return []
            spec = {"kind": "journal", "issn": issn}

        queries = list(self.retriever_config.get("crossref_queries") or DEFAULT_CROSSREF_QUERIES)
        query_rows = int(
            self.retriever_config.get("max_crossref_query_entries") or DEFAULT_MAX_CROSSREF_QUERY_ENTRIES
        )
        items: list[dict[str, Any]] = []
        try:
            # 主题检索先行，保证 PA 相关论文排在候选池前面，不会被整期目录挤掉。
            for query in queries:
                items.extend(self._crossref_request(spec, query_rows, query_title=query))
            items.extend(self._crossref_request(spec, max_entries, query_title=None))
        except Exception as exc:
            logger.warning(f"Crossref fetch failed for {journal} ({spec}): {exc}")
            return []

        deduped = []
        seen = set()
        for item in items:
            if not self._is_research_crossref_item(item):
                continue
            key = item.get("DOI") or item.get("URL") or str(item.get("title", ""))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= max_entries:
                break
        logger.info(f"Retrieved {len(deduped)} RF Crossref entries from {journal}")
        return [{"journal": journal, "entry": item, "entry_source": "crossref"} for item in deduped]

    def _retrieve_raw_papers(self) -> list[dict[str, Any]]:
        feeds = self.retriever_config.get("feeds") or DEFAULT_RF_RSS_FEEDS
        max_entries_per_feed = int(self.retriever_config.get("max_entries_per_feed") or 5)
        if self.config.executor.debug:
            max_entries_per_feed = min(max_entries_per_feed, 2)

        raw_papers = []
        satisfied_by_rss: set[str] = set()
        for journal, url in dict(feeds).items():
            feed = feedparser.parse(url)
            if getattr(feed, "bozo", False):
                logger.warning(f"RF RSS feed parse warning for {journal}: {getattr(feed, 'bozo_exception', '')}")
            entries = list(getattr(feed, "entries", []))[:max_entries_per_feed]
            logger.info(f"Retrieved {len(entries)} RF RSS entries from {journal}")
            if entries:
                satisfied_by_rss.add(journal)
                for entry in entries:
                    raw_papers.append({"journal": journal, "entry": entry, "entry_source": "rss"})
            else:
                # IEEE 的 RSS 端点长期对自动化访问返回 418，Crossref 是常态路径而非异常路径。
                logger.warning(
                    f"No RF RSS entries from {journal} via {url}; "
                    f"status={getattr(feed, 'status', 'unknown')}. Falling back to Crossref."
                )

        # 没有被 RSS 满足的源（含没有配置 feed 的期刊/会议/预印本）统一走 Crossref。
        for name, spec in self._crossref_sources().items():
            if name in satisfied_by_rss:
                continue
            raw_papers.extend(self._retrieve_crossref_entries(name, max_entries_per_feed, spec))
        return raw_papers

    def convert_to_paper(self, raw_paper: dict[str, Any]) -> Paper | None:
        journal = raw_paper["journal"]
        entry = raw_paper["entry"]
        entry_source = raw_paper.get("entry_source", "rss")

        if entry_source == "crossref":
            title_values = entry.get("title") or []
            title = self._clean_crossref_text(title_values[0] if title_values else "")
            url = entry.get("URL") or (f"https://doi.org/{entry.get('DOI')}" if entry.get("DOI") else "")
            summary = self._clean_crossref_text(entry.get("abstract", ""))
            authors = []
            for author in entry.get("author", [])[:12]:
                name = " ".join(part for part in [author.get("given"), author.get("family")] if part)
                if name:
                    authors.append(name)
            published_date = self._crossref_date(entry)
        else:
            title = getattr(entry, "title", None) or entry.get("title", "")
            url = getattr(entry, "link", None) or entry.get("link", "")
            summary = getattr(entry, "summary", None) or entry.get("summary", "")
            authors = getattr(entry, "authors", None) or entry.get("authors", [])
            if isinstance(authors, str):
                authors = [authors]
            elif authors and isinstance(authors[0], dict):
                authors = [a.get("name", "") for a in authors if a.get("name")]
            published_date = (
                format_published_date(entry.get("published"))
                or format_published_date(entry.get("updated"))
                or format_published_date(entry.get("published_parsed"))
                or format_published_date(entry.get("updated_parsed"))
            )

        if not title or not url:
            return None

        # Crossref 不提供 IEEE 期刊的摘要，RSS 条目也常常只有标题。缺摘要会让
        # reranker 只能对标题做 embedding，并让 TLDR 无中生有，因此先尝试补回。
        if not summary:
            doi = entry.get("DOI") if entry_source == "crossref" else None
            summary = enrich_abstract(doi or url) or ""

        abstract = summary or f"Latest {journal} RF/microwave journal paper: {title}"
        venue, venue_rank, cas_partition, sci_quartile = self._get_venue_info(journal)
        return Paper(
            source=self.name,
            title=f"[{journal}] {title}",
            authors=authors or [journal],
            abstract=abstract,
            abstract_is_placeholder=not summary,
            url=url,
            pdf_url=url,
            full_text=None,
            published_date=published_date,
            venue=venue,
            venue_rank=venue_rank,
            cas_partition=cas_partition,
            sci_quartile=sci_quartile,
        )
