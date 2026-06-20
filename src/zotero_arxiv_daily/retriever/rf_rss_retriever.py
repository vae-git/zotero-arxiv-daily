from typing import Any

import feedparser
from loguru import logger

from ..protocol import Paper
from .base import BaseRetriever, register_retriever
from .date_utils import format_published_date


DEFAULT_RF_RSS_FEEDS = {
    "T-MTT": "https://ieeexplore.ieee.org/rss/TOC22.xml",
    "T-AP": "https://ieeexplore.ieee.org/rss/TOC8.xml",
    "MWTL": "https://ieeexplore.ieee.org/rss/TOC7260.xml",
    "JSSC": "https://ieeexplore.ieee.org/rss/TOC4.xml",
}


@register_retriever("rf_rss")
class RfRssRetriever(BaseRetriever):
    def _retrieve_raw_papers(self) -> list[dict[str, Any]]:
        feeds = self.retriever_config.get("feeds") or DEFAULT_RF_RSS_FEEDS
        max_entries_per_feed = int(self.retriever_config.get("max_entries_per_feed") or 5)
        if self.config.executor.debug:
            max_entries_per_feed = min(max_entries_per_feed, 2)

        raw_papers = []
        for journal, url in dict(feeds).items():
            feed = feedparser.parse(url)
            if getattr(feed, "bozo", False):
                logger.warning(f"RF RSS feed parse warning for {journal}: {getattr(feed, 'bozo_exception', '')}")
            entries = list(getattr(feed, "entries", []))[:max_entries_per_feed]
            logger.info(f"Retrieved {len(entries)} RF RSS entries from {journal}")
            for entry in entries:
                raw_papers.append({"journal": journal, "entry": entry})
        return raw_papers

    def convert_to_paper(self, raw_paper: dict[str, Any]) -> Paper | None:
        journal = raw_paper["journal"]
        entry = raw_paper["entry"]
        title = getattr(entry, "title", None) or entry.get("title", "")
        url = getattr(entry, "link", None) or entry.get("link", "")
        summary = getattr(entry, "summary", None) or entry.get("summary", "")
        authors = getattr(entry, "authors", None) or entry.get("authors", [])
        if isinstance(authors, str):
            authors = [authors]
        elif authors and isinstance(authors[0], dict):
            authors = [a.get("name", "") for a in authors if a.get("name")]

        if not title or not url:
            return None

        abstract = summary or f"Latest {journal} RF/microwave journal paper: {title}"
        published_date = (
            format_published_date(entry.get("published"))
            or format_published_date(entry.get("updated"))
            or format_published_date(entry.get("published_parsed"))
            or format_published_date(entry.get("updated_parsed"))
        )
        return Paper(
            source=self.name,
            title=f"[{journal}] {title}",
            authors=authors or [journal],
            abstract=abstract,
            url=url,
            pdf_url=url,
            full_text=None,
            published_date=published_date,
        )
