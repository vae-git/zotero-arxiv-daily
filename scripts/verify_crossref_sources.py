"""One-off checker for the Crossref sources configured in ``config/custom.yaml``.

IEEE's RSS endpoints return HTTP 418 to automated clients, so every RF source in
this project actually resolves through Crossref. Before trusting a newly added
ISSN, conference container title or DOI prefix, run this script to confirm it
returns real, on-topic records:

    python scripts/verify_crossref_sources.py

Sources that come back empty should be removed from ``crossref_sources`` rather
than left in place emitting a daily warning.
"""

import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta

UA = {"User-Agent": "zotero-arxiv-daily/1.0 (mailto:no-reply@example.com)"}
LOOKBACK_DAYS = 120
PROBE_QUERY = "power amplifier"

# 与 config/custom.yaml 的 crossref_sources 保持一致。
SOURCES: dict[str, dict[str, str]] = {
    "T-MTT": {"kind": "journal", "issn": "0018-9480"},
    "T-AP": {"kind": "journal", "issn": "0018-926X"},
    "MWTL": {"kind": "journal", "issn": "2771-957X"},
    "JSSC": {"kind": "journal", "issn": "0018-9200"},
    "T-CAS-I": {"kind": "journal", "issn": "1549-8328"},
    "T-CAS-II": {"kind": "journal", "issn": "1549-7747"},
    "JMW": {"kind": "journal", "issn": "2692-8388"},
    "T-ED": {"kind": "journal", "issn": "0018-9383"},
    "IMS": {"kind": "proceedings", "container": "IEEE MTT-S International Microwave Symposium"},
    "RFIC": {"kind": "proceedings", "container": "IEEE Radio Frequency Integrated Circuits Symposium"},
    "TechRxiv": {"kind": "prefix", "prefix": "10.36227"},
}


def build_url(spec: dict[str, str], rows: int, query_title: str | None) -> str | None:
    params: dict[str, object] = {
        "sort": "published",
        "order": "desc",
        "rows": rows,
        "select": "DOI,title,abstract",
    }
    filters = [f"from-pub-date:{(date.today() - timedelta(days=LOOKBACK_DAYS)).isoformat()}"]
    kind = spec.get("kind", "journal")

    if kind == "journal":
        url = f"https://api.crossref.org/journals/{spec['issn']}/works"
        filters.append("type:journal-article")
    elif kind == "proceedings":
        url = "https://api.crossref.org/works"
        filters.append("type:proceedings-article")
        params["query.container-title"] = spec["container"]
    elif kind == "prefix":
        url = f"https://api.crossref.org/prefixes/{spec['prefix']}/works"
    else:
        return None

    params["filter"] = ",".join(filters)
    if query_title:
        params["query.title"] = query_title
    return f"{url}?{urllib.parse.urlencode(params)}"


def probe(name: str, spec: dict[str, str]) -> bool:
    url = build_url(spec, rows=30, query_title=PROBE_QUERY)
    if url is None:
        print(f"  {name:10s} SKIP  unknown kind {spec.get('kind')!r}")
        return False

    try:
        request = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(request, timeout=45) as response:
            message = json.load(response).get("message", {})
    except Exception as exc:
        print(f"  {name:10s} FAIL  {type(exc).__name__}: {exc}")
        return False

    items = message.get("items", [])
    on_topic = [i for i in items if "amplifier" in ((i.get("title") or [""])[0]).lower()]
    with_abstract = sum(1 for i in items if i.get("abstract"))
    status = "OK  " if items else "EMPTY"
    print(
        f"  {name:10s} {status} total={message.get('total-results')} "
        f"returned={len(items)} on-topic={len(on_topic)} with-abstract={with_abstract}"
    )
    for item in on_topic[:2]:
        print(f"             + {((item.get('title') or [''])[0])[:70]}")
    return bool(items)


def main() -> int:
    print(f"Probing Crossref sources with query.title={PROBE_QUERY!r}, lookback={LOOKBACK_DAYS}d\n")
    failed = []
    for name, spec in SOURCES.items():
        if not probe(name, spec):
            failed.append(name)
        time.sleep(0.5)

    print()
    if failed:
        print(f"Sources returning nothing: {', '.join(failed)}")
        print("Remove these from config/custom.yaml -> source.rf_rss.crossref_sources.")
    else:
        print("All configured sources returned results.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
