"""Recover abstracts for papers whose metadata source does not provide one.

Crossref does not carry abstracts for IEEE journals (measured: 0/20 on T-MTT, T-AP,
MWTL and JSSC), so RF journal entries would otherwise fall back to a placeholder
sentence. That placeholder poisons two things at once: the reranker embeds title-only
text and produces meaningless similarity scores, and the LLM generates a TLDR out of
nothing. Filling the abstract back in from OpenAlex fixes both.

OpenAlex is the primary source (measured ~75% coverage on recent IEEE articles, free,
no API key). Semantic Scholar is the fallback for the remainder.
"""

import re
from functools import lru_cache
from urllib.parse import quote, unquote, urlparse

import requests
from loguru import logger

OPENALEX_URL = "https://api.openalex.org/works/doi:{doi}"
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
REQUEST_TIMEOUT = 20
CONTACT_EMAIL = "no-reply@example.com"
USER_AGENT = f"zotero-arxiv-daily/1.0 (mailto:{CONTACT_EMAIL})"
# 低于这个长度的多半是残缺片段（版权声明、单句导语），不如让调用方退回占位文案。
MIN_ABSTRACT_LENGTH = 80
# 与 executor._extract_doi 中的模式一致。此处重复定义是为了让本模块不依赖
# executor —— 后者会 import retriever，反向引用会形成循环导入。
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s<>\"'\?#]+", re.IGNORECASE)


def extract_doi(value: str | None) -> str | None:
    """Pull a bare DOI out of a raw DOI string or a doi.org/publisher URL."""
    value = unquote(str(value or "")).strip()
    if not value:
        return None
    parsed = urlparse(value)
    candidates = [value]
    if "doi.org" in parsed.netloc.lower() and parsed.path:
        candidates.insert(0, parsed.path.lstrip("/"))
    for candidate in candidates:
        match = DOI_RE.search(candidate)
        if match:
            return match.group(0).rstrip(".,);]").lower()
    return None


def _clean_abstract_text(text: str | None) -> str:
    """Strip JATS/HTML markup and collapse whitespace."""
    text = str(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def restore_inverted_abstract(inverted_index: dict | None) -> str | None:
    """Rebuild plain text from OpenAlex's ``abstract_inverted_index``.

    The index maps each word to the list of positions it occupies. Sorting by
    position reconstructs the original word order without assuming the positions
    are contiguous.
    """
    if not inverted_index:
        return None
    positions = [(index, word) for word, indices in inverted_index.items() for index in indices]
    if not positions:
        return None
    positions.sort(key=lambda item: item[0])
    return " ".join(word for _, word in positions)


def _get_json(url: str, params: dict) -> dict:
    response = requests.get(
        url,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def _fetch_openalex_abstract(doi: str) -> str | None:
    payload = _get_json(OPENALEX_URL.format(doi=quote(doi, safe="")), {"mailto": CONTACT_EMAIL})
    return restore_inverted_abstract(payload.get("abstract_inverted_index"))


def _fetch_semantic_scholar_abstract(doi: str) -> str | None:
    payload = _get_json(SEMANTIC_SCHOLAR_URL.format(doi=quote(doi, safe="")), {"fields": "abstract"})
    return payload.get("abstract")


_SOURCES = (
    ("OpenAlex", _fetch_openalex_abstract),
    ("Semantic Scholar", _fetch_semantic_scholar_abstract),
)


@lru_cache(maxsize=512)
def enrich_abstract(doi_or_url: str | None) -> str | None:
    """Return a recovered abstract for *doi_or_url*, or ``None`` if unavailable.

    Failures are non-fatal by design: a missing abstract degrades ranking quality
    but must never break the daily run. Results (including misses) are cached so a
    DOI is looked up at most once per run.
    """
    doi = extract_doi(doi_or_url)
    if not doi:
        return None

    for source_name, fetcher in _SOURCES:
        try:
            abstract = fetcher(doi)
        except requests.HTTPError as exc:
            # 404 只是说明该源没收录这篇，属正常情况，不值得 warning 刷屏。
            status = getattr(exc.response, "status_code", None)
            log = logger.debug if status == 404 else logger.warning
            log(f"{source_name} abstract lookup failed for {doi}: {exc}")
            continue
        except Exception as exc:
            logger.warning(f"{source_name} abstract lookup failed for {doi}: {exc}")
            continue

        abstract = _clean_abstract_text(abstract)
        if len(abstract) >= MIN_ABSTRACT_LENGTH:
            logger.debug(f"Recovered abstract for {doi} from {source_name} ({len(abstract)} chars)")
            return abstract

    logger.debug(f"No abstract recovered for {doi}")
    return None
