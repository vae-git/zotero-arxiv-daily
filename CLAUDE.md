# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Zotero-arXiv-Daily recommends new arXiv/bioRxiv/medRxiv papers based on a user's Zotero library. It computes embedding similarity between new papers and the user's existing library, generates TLDRs via LLM, and delivers results by email. Designed to run as a GitHub Actions workflow at zero cost.

## Commands

```bash
# Run the application
uv run src/zotero_arxiv_daily/main.py

# Run tests (excludes slow tests by default)
uv run pytest

# Run all tests including slow ones
uv run pytest -m ""

# Run a single test
uv run pytest tests/test_utils.py::TestGlobMatch -v

# Install/sync dependencies
uv sync
```

No linter or formatter is configured.

## Architecture

The app follows a linear pipeline orchestrated by `Executor` (`src/zotero_arxiv_daily/executor.py`):

1. **Fetch Zotero corpus** — retrieves user's library papers via pyzotero API
2. **Filter corpus** — applies `include_path` glob patterns to select relevant collections
3. **Retrieve new papers** — fetches from configured sources (arXiv RSS, bioRxiv/medRxiv REST API, RF journals/conferences via Crossref)
4. **Rerank** — scores candidates by weighted similarity to corpus (newer Zotero papers weighted higher), then applies the tiered keyword focus multiplier
5. **Apply source quotas + truncate** to `max_paper_num`
6. **Fetch full text** — only for the papers that survived step 5, via `BaseRetriever.fetch_full_text`
7. **Generate translated title, translated abstract, TLDR + affiliations** — via OpenAI-compatible LLM API
8. **Render + send email** — HTML email via SMTP

### RF/PA specifics

This fork is tuned for RF power amplifier literature:

- **Crossref is the real path for RF journals.** IEEE's RSS endpoints return HTTP 418 to automated clients, so `rf_rss` almost always falls through to Crossref. `source.rf_rss.crossref_sources` supports three kinds: `journal` (ISSN), `proceedings` (container title) and `prefix` (DOI prefix, for preprint servers).
- **Crossref carries no abstracts for IEEE.** `retriever/abstract_enricher.py` recovers them from OpenAlex, falling back to Semantic Scholar. Papers where recovery fails get `abstract_is_placeholder=True` and are skipped by abstract translation and email rendering.
- **Keyword focus is tiered.** `primary_keywords` (PA core) boost, `domain_keywords` (Tier2, general RF) keep a paper alive at a multiplier below 1.0, and everything else takes `no_primary_penalty`. Do not add bare `"pa"` to `primary_keywords` — it matches "Power Allocation" in eess.SP.
- **Translation uses a glossary.** `rf_glossary.py` injects only the RF terms that actually appear in a paper, keeping token cost proportional to jargon density.

Run `python scripts/verify_crossref_sources.py` before trusting a newly added ISSN, conference or DOI prefix.

### Plugin Systems

**Retrievers** (`src/zotero_arxiv_daily/retriever/`): Register via `@register_retriever` decorator, discovered by `get_retriever_cls()`. Each retriever implements `_retrieve_raw_papers()` and `convert_to_paper()`.

**Rerankers** (`src/zotero_arxiv_daily/reranker/`): Register via `@register_reranker` decorator, discovered by `get_reranker_cls()`. Two implementations: `local` (sentence-transformers) and `api` (OpenAI-compatible embeddings endpoint).

### Configuration

Uses Hydra + OmegaConf. Config is composed from `config/base.yaml` (defaults) + `config/custom.yaml` (user overrides). Environment variables are interpolated via `${oc.env:VAR_NAME,default}` syntax. Entry point uses `@hydra.main`.

### Data Classes

`Paper` and `CorpusPaper` in `src/zotero_arxiv_daily/protocol.py`. `Paper` has LLM-powered methods (`generate_tldr`, `generate_affiliations`) that call the OpenAI API directly.

## Testing

Tests marked `@pytest.mark.slow` require heavy dependencies (e.g., sentence-transformers model download) and are skipped locally by default (`addopts = "-m 'not slow'"` in pyproject.toml). All other tests run with pure Python stubs (no Docker containers needed).

```bash
# Run tests (excludes slow tests)
uv run pytest

# Run all tests including slow ones
uv run pytest -m ""

# Run with coverage
uv run pytest --cov=src/zotero_arxiv_daily --cov-report=term-missing
```

## gstack

Use the `/browse` skill from gstack for all web browsing. Never use `mcp__claude-in-chrome__*` tools.

Available skills: `/office-hours`, `/plan-ceo-review`, `/plan-eng-review`, `/plan-design-review`, `/design-consultation`, `/design-shotgun`, `/design-html`, `/review`, `/ship`, `/land-and-deploy`, `/canary`, `/benchmark`, `/browse`, `/connect-chrome`, `/qa`, `/qa-only`, `/design-review`, `/setup-browser-cookies`, `/setup-deploy`, `/retro`, `/investigate`, `/document-release`, `/codex`, `/cso`, `/autoplan`, `/plan-devex-review`, `/devex-review`, `/careful`, `/freeze`, `/guard`, `/unfreeze`, `/gstack-upgrade`, `/learn`.

If gstack skills aren't working, run `cd .claude/skills/gstack && ./setup` to build the binary and register skills.

## Git Workflow

- PRs should target the `dev` branch, not `main`
- Current development branch: `dev`
