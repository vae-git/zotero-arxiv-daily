"""Tests for zotero_arxiv_daily.protocol: Paper.generate_tldr, Paper.generate_affiliations."""

import pytest
from types import SimpleNamespace

from tests.canned_responses import make_sample_paper, make_stub_openai_client
from zotero_arxiv_daily.protocol import (
    DEFAULT_TLDR_MAX_TOKENS,
    SILICONFLOW_DEFAULT_MODEL,
    Paper,
    ZH_LABEL,
    contains_chinese,
    wants_bilingual_tldr,
)


@pytest.fixture()
def llm_params():
    return {
        "language": "English",
        "generation_kwargs": {"model": "gpt-4o-mini", "max_tokens": 16384},
    }


# ---------------------------------------------------------------------------
# generate_tldr
# ---------------------------------------------------------------------------


def test_tldr_returns_response(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_tldr(client, llm_params)
    assert result == "Hello! How can I assist you today?"
    assert paper.tldr == result


def test_tldr_without_abstract_or_fulltext(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(abstract="", full_text=None)
    result = paper.generate_tldr(client, llm_params)
    assert "Failed to generate TLDR" in result


def test_tldr_falls_back_to_abstract_on_error(llm_params):
    paper = make_sample_paper()

    # Client whose create() raises
    from types import SimpleNamespace

    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("API down")))
        )
    )
    result = paper.generate_tldr(broken_client, llm_params)
    assert result == paper.abstract


def test_tldr_truncates_long_prompt(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(full_text="word " * 10000)
    result = paper.generate_tldr(client, llm_params)
    assert result is not None


def test_tldr_bilingual_prompt_requests_english_and_chinese(llm_params):
    captured = {}
    bilingual_tldr = f"English: Summary.\n{ZH_LABEL}: \u6458\u8981\u3002"

    def create(**kwargs):
        captured["messages"] = kwargs["messages"]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=bilingual_tldr))]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    llm_params["language"] = "English and Chinese"
    paper = make_sample_paper()

    result = paper.generate_tldr(client, llm_params)

    assert result == bilingual_tldr
    request = str(captured["messages"])
    assert "English:" in request
    assert f"{ZH_LABEL}:" in request


def test_tldr_bilingual_prompt_repairs_english_only_answer(llm_params):
    responses = [
        "English: Summary only.",
        f"English: Summary.\n{ZH_LABEL}: \u6458\u8981\u3002",
    ]

    def create(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=responses.pop(0)))]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    llm_params["language"] = "English and Chinese"
    paper = make_sample_paper()

    result = paper.generate_tldr(client, llm_params)

    assert f"{ZH_LABEL}:" in result
    assert contains_chinese(result)


def test_wants_bilingual_tldr():
    assert wants_bilingual_tldr("English and Chinese")
    assert wants_bilingual_tldr("\u4e2d\u82f1\u6587")
    assert not wants_bilingual_tldr("Chinese")


def test_llm_generation_kwargs_adapts_siliconflow_defaults():
    kwargs = Paper._llm_generation_kwargs({
        "api": {"base_url": "https://api.siliconflow.cn/v1"},
        "generation_kwargs": {"model": "gpt-4o-mini", "max_tokens": 16384},
    })

    assert kwargs["model"] == SILICONFLOW_DEFAULT_MODEL
    assert kwargs["max_tokens"] == DEFAULT_TLDR_MAX_TOKENS


# ---------------------------------------------------------------------------
# generate_affiliations
# ---------------------------------------------------------------------------


def test_affiliations_returns_parsed_list(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    assert isinstance(result, list)
    assert "TsingHua University" in result
    assert "Peking University" in result


def test_affiliations_none_without_fulltext(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(full_text=None)
    result = paper.generate_affiliations(client, llm_params)
    assert result is None


def test_affiliations_deduplicates(llm_params):
    """The stub returns two distinct affiliations, so no dedup needed.
    But confirm the set() dedup in the code doesn't break anything.
    """
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    assert len(result) == len(set(result))


def test_affiliations_malformed_llm_output(llm_params):
    """LLM returns affiliations without JSON brackets. Should fall back gracefully."""
    from types import SimpleNamespace

    def create_no_brackets(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="TsingHua University, Peking University"),
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_no_brackets)
        )
    )
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    # re.search for [...] will fail -> AttributeError -> caught -> returns None
    assert result is None


def test_affiliations_error_returns_none(llm_params):
    from types import SimpleNamespace

    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        )
    )
    paper = make_sample_paper()
    result = paper.generate_affiliations(broken_client, llm_params)
    assert result is None
    assert paper.affiliations is None
