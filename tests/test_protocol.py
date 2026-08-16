"""Tests for zotero_arxiv_daily.protocol: Paper.generate_tldr, Paper.generate_affiliations."""

import pytest
from types import SimpleNamespace

from tests.canned_responses import make_sample_paper, make_stub_openai_client
from zotero_arxiv_daily.protocol import (
    DEFAULT_TLDR_MAX_TOKENS,
    SILICONFLOW_DEFAULT_MODEL,
    TASK_MAX_TOKENS,
    Paper,
    TITLE_ZH_FAILURE,
    ZH_LABEL,
    contains_chinese,
    normalize_llm_base_url,
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


def test_title_translation_returns_chinese_title(llm_params):
    captured = {}

    def create(**kwargs):
        captured["messages"] = kwargs["messages"]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="\u7528\u4e8e\u5c04\u9891\u524d\u7aef\u7684\u7d27\u51d1\u578b\u5fae\u6ce2\u6ee4\u6ce2\u5668")
                )
            ]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    paper = make_sample_paper(title="A compact microwave filter for RF front ends")

    result = paper.generate_title_translation(client, llm_params)

    assert result == "\u7528\u4e8e\u5c04\u9891\u524d\u7aef\u7684\u7d27\u51d1\u578b\u5fae\u6ce2\u6ee4\u6ce2\u5668"
    assert paper.title_zh == result
    assert "Translate the following scientific paper title" in str(captured["messages"])


def test_title_translation_strips_chinese_label(llm_params):
    def create(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="\u4e2d\u6587\u6807\u9898\uff1a\u7528\u4e8e\u5c04\u9891\u524d\u7aef\u7684\u7d27\u51d1\u578b\u5fae\u6ce2\u6ee4\u6ce2\u5668")
                )
            ]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    paper = make_sample_paper(title="A compact microwave filter for RF front ends")

    result = paper.generate_title_translation(client, llm_params)

    assert result == "\u7528\u4e8e\u5c04\u9891\u524d\u7aef\u7684\u7d27\u51d1\u578b\u5fae\u6ce2\u6ee4\u6ce2\u5668"


def test_title_translation_repairs_non_chinese_answer(llm_params):
    responses = [
        "A compact microwave filter for RF front ends",
        "\u7528\u4e8e\u5c04\u9891\u524d\u7aef\u7684\u7d27\u51d1\u578b\u5fae\u6ce2\u6ee4\u6ce2\u5668",
    ]

    def create(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=responses.pop(0)))]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    paper = make_sample_paper(title="A compact microwave filter for RF front ends")

    result = paper.generate_title_translation(client, llm_params)

    assert result == "\u7528\u4e8e\u5c04\u9891\u524d\u7aef\u7684\u7d27\u51d1\u578b\u5fae\u6ce2\u6ee4\u6ce2\u5668"
    assert paper.title_zh == result


def test_title_translation_error_is_visible_in_email_title(llm_params):
    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("API down")))
        )
    )
    paper = make_sample_paper(title="A compact microwave filter for RF front ends")

    result = paper.generate_title_translation(broken_client, llm_params)

    assert result is None
    assert TITLE_ZH_FAILURE in paper.title_zh


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


def test_contains_chinese():
    assert contains_chinese("\u5c04\u9891\u524d\u7aef")
    assert not contains_chinese("RF front-end")


def test_llm_generation_kwargs_adapts_siliconflow_defaults():
    kwargs = Paper._llm_generation_kwargs({
        "api": {"base_url": "https://api.siliconflow.cn/v1"},
        "generation_kwargs": {"model": "gpt-4o-mini", "max_tokens": 16384},
    })

    assert kwargs["model"] == SILICONFLOW_DEFAULT_MODEL
    assert kwargs["max_tokens"] == DEFAULT_TLDR_MAX_TOKENS


def test_normalize_llm_base_url_adds_siliconflow_v1():
    assert normalize_llm_base_url("https://api.siliconflow.cn") == "https://api.siliconflow.cn/v1"
    assert normalize_llm_base_url("https://api.siliconflow.cn/v1") == "https://api.siliconflow.cn/v1"


def test_llm_generation_kwargs_scales_max_tokens_per_task():
    """Abstract translation must not be squeezed into the 512-token TLDR budget."""
    params = {
        "api": {"base_url": "https://api.siliconflow.cn/v1"},
        "generation_kwargs": {"model": "gpt-4o-mini", "max_tokens": 16384},
    }

    assert Paper._llm_generation_kwargs(params, task="tldr")["max_tokens"] == TASK_MAX_TOKENS["tldr"]
    assert Paper._llm_generation_kwargs(params, task="title")["max_tokens"] == TASK_MAX_TOKENS["title"]
    assert Paper._llm_generation_kwargs(params, task="abstract")["max_tokens"] == TASK_MAX_TOKENS["abstract"]
    assert TASK_MAX_TOKENS["abstract"] > DEFAULT_TLDR_MAX_TOKENS


def test_llm_generation_kwargs_defaults_to_tldr_budget():
    kwargs = Paper._llm_generation_kwargs({"generation_kwargs": {"max_tokens": 128}})
    assert kwargs["max_tokens"] == 128


# ---------------------------------------------------------------------------
# generate_abstract_translation
# ---------------------------------------------------------------------------


def test_abstract_translation_returns_chinese(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(abstract="A broadband Doherty power amplifier design.")

    result = paper.generate_abstract_translation(client, llm_params)

    assert result is not None
    assert contains_chinese(result)
    assert paper.abstract_zh == result


def test_abstract_translation_skips_placeholder(llm_params):
    """rf_rss falls back to a placeholder sentence; translating it is pointless."""

    def fail_create(**kwargs):
        raise AssertionError("placeholder abstracts must not reach the LLM")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail_create)))
    paper = make_sample_paper(
        abstract="Latest T-MTT RF/microwave journal paper: Some Title",
        abstract_is_placeholder=True,
    )

    assert paper.generate_abstract_translation(client, llm_params) is None
    assert paper.abstract_zh is None


def test_abstract_translation_skips_empty_abstract(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(abstract="")

    assert paper.generate_abstract_translation(client, llm_params) is None


def test_abstract_translation_keeps_already_chinese_abstract(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(abstract="本文提出一种功率放大器。")

    assert paper.generate_abstract_translation(client, llm_params) == "本文提出一种功率放大器。"


def test_abstract_translation_error_returns_none(llm_params):
    def raising_create(**kwargs):
        raise RuntimeError("boom")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=raising_create)))
    paper = make_sample_paper(abstract="A broadband Doherty power amplifier design.")

    # 摘要翻译失败不应影响这封邮件的其余内容。
    assert paper.generate_abstract_translation(client, llm_params) is None
    assert paper.abstract_zh is None


@pytest.mark.parametrize(
    "raw",
    [
        "中文摘要：本文提出一种功率放大器。",
        "```\n本文提出一种功率放大器。\n```",
        "  本文提出一种功率放大器。  ",
    ],
)
def test_clean_abstract_translation_strips_wrappers(raw):
    assert Paper._clean_abstract_translation(raw) == "本文提出一种功率放大器。"


def test_abstract_translation_prompt_carries_rf_glossary(llm_params):
    seen = {}

    def capture_create(**kwargs):
        seen["messages"] = kwargs["messages"]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="本文提出一种多赫蒂功率放大器。"))]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=capture_create)))
    paper = make_sample_paper(
        title="A Doherty Power Amplifier",
        abstract="This Doherty power amplifier improves back-off efficiency.",
    )
    paper.generate_abstract_translation(client, llm_params)

    prompt = str(seen["messages"])
    assert "多赫蒂" in prompt
    assert "功率放大器" in prompt
    assert "功率回退" in prompt


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
