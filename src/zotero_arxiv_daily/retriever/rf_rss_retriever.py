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
        venue, venue_rank, cas_partition, sci_quartile = self._get_venue_info(journal)
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
            venue=venue,
            venue_rank=venue_rank,
            cas_partition=cas_partition,
            sci_quartile=sci_quartile,
        )
