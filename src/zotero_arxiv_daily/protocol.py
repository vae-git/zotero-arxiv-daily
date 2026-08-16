from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json
from .rf_glossary import glossary_hint_for
RawPaperItem = TypeVar('RawPaperItem')

ZH_LABEL = "\u4e2d\u6587"
TITLE_ZH_FAILURE = "\u4e2d\u6587\u6807\u9898\u751f\u6210\u5931\u8d25\uff0c\u8bf7\u68c0\u67e5 LLM \u914d\u7f6e\u6216\u8fd0\u884c\u65e5\u5fd7\u3002"
DEFAULT_TLDR_MAX_TOKENS = 512
# \u5404\u7c7b\u4efb\u52a1\u7684\u8f93\u51fa\u957f\u5ea6\u9700\u6c42\u5dee\u522b\u5f88\u5927\uff1a\u6807\u9898\u4e00\u884c\u5c31\u591f\uff0c\u800c\u6458\u8981\u7ffb\u8bd1\u4e00\u65e6\u88ab\u622a\u65ad\u5c31\u6beb\u65e0\u4ef7\u503c\u3002
TASK_MAX_TOKENS = {
    "title": 256,
    "tldr": DEFAULT_TLDR_MAX_TOKENS,
    "abstract": 1536,
    "affiliations": 512,
}
SILICONFLOW_DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"


def normalize_llm_base_url(base_url: str | None) -> str:
    base_url = str(base_url or "").strip().rstrip("/")
    if "siliconflow.cn" in base_url and not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def wants_chinese(language: str) -> bool:
    language = str(language or "").lower()
    return "chinese" in language or "\u4e2d\u6587" in language or "\u4e2d" in language or "zh" in language


def wants_bilingual_tldr(language: str) -> bool:
    language_lower = str(language or "").lower()
    has_english = "english" in language_lower or "\u82f1\u6587" in language_lower or "\u82f1" in language_lower
    return has_english and wants_chinese(language)


def contains_chinese(text: str | None) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in (text or ""))

@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    title_zh: Optional[str] = None
    abstract_zh: Optional[str] = None
    tldr: Optional[str] = None
    affiliations: Optional[list[str]] = None
    score: Optional[float] = None
    published_date: Optional[str] = None
    venue: Optional[str] = None
    venue_rank: Optional[str] = None
    cas_partition: Optional[str] = None
    sci_quartile: Optional[str] = None
    # 检索源拿不到真实摘要时会填入占位文案。翻译和邮件渲染都应跳过这类内容。
    abstract_is_placeholder: bool = False

    def _generate_tldr_with_llm(self, openai_client:OpenAI,llm_params:dict) -> str:
        lang = llm_params.get('language', 'English')
        if wants_bilingual_tldr(lang):
            prompt = (
                "Given the following information of a paper, generate a bilingual TLDR summary.\n"
                "Return exactly two concise lines in this format:\n"
                "English: <one-sentence English TLDR>\n"
                f"{ZH_LABEL}: <one-sentence Simplified Chinese TLDR>\n\n"
            )
        else:
            prompt = f"Given the following information of a paper, generate a one-sentence TLDR summary in {lang}:\n\n"

        # 术语表放在最前面：下面的 tiktoken 截断会砍掉尾部，全文预览可以丢，术语约束不能丢。
        if wants_chinese(lang):
            prompt += glossary_hint_for(self.title, self.abstract)

        if self.title:
            prompt += f"Title:\n {self.title}\n\n"

        if self.abstract:
            prompt += f"Abstract: {self.abstract}\n\n"

        if self.full_text:
            prompt += f"Preview of main content:\n {self.full_text}\n\n"

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate TLDR. Neither full text nor abstract is provided"
        
        # use gpt-4o tokenizer for estimation
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
        prompt = enc.decode(prompt_tokens)
        
        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": self._tldr_system_prompt(lang),
                },
                {"role": "user", "content": prompt},
            ],
            **self._llm_generation_kwargs(llm_params, task="tldr")
        )
        tldr = response.choices[0].message.content
        if wants_bilingual_tldr(lang) and not self._has_bilingual_tldr(tldr):
            tldr = self._repair_bilingual_tldr(openai_client, llm_params, tldr)
        return tldr

    @staticmethod
    def _tldr_system_prompt(language: str) -> str:
        if wants_bilingual_tldr(language):
            return (
                "You are an assistant who perfectly summarizes scientific papers. "
                "Answer with exactly two lines: one English TLDR line and one Simplified Chinese TLDR line."
            )
        return f"You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user. Your answer should be in {language}."

    @staticmethod
    def _config_get(config, key: str, default=None):
        if config is None:
            return default
        if hasattr(config, "get"):
            return config.get(key, default)
        return getattr(config, key, default)

    @staticmethod
    def _llm_generation_kwargs(llm_params: dict, task: str = "tldr") -> dict:
        kwargs = dict(Paper._config_get(llm_params, "generation_kwargs", {}) or {})
        api_config = Paper._config_get(llm_params, "api", {}) or {}
        base_url = normalize_llm_base_url(Paper._config_get(api_config, "base_url", "")).lower()

        if kwargs.get("model") == "gpt-4o-mini" and "siliconflow" in base_url:
            kwargs["model"] = SILICONFLOW_DEFAULT_MODEL

        # 配置里的 max_tokens 被当作 TLDR 的预算；其余任务用各自的默认值，
        # 否则摘要翻译会被压在 512 token 而中途截断。
        task_default = TASK_MAX_TOKENS.get(task, DEFAULT_TLDR_MAX_TOKENS)
        if task == "tldr":
            try:
                configured = int(kwargs.get("max_tokens", task_default))
            except (TypeError, ValueError):
                configured = task_default
            kwargs["max_tokens"] = min(configured, task_default)
        else:
            kwargs["max_tokens"] = task_default

        return kwargs

    @staticmethod
    def _has_bilingual_tldr(tldr: str | None) -> bool:
        return bool(tldr and "English:" in tldr and f"{ZH_LABEL}:" in tldr and contains_chinese(tldr))

    @staticmethod
    def _safe_error_message(exc: Exception) -> str:
        message = f"{type(exc).__name__}: {exc}"
        message = re.sub(r"(sk-[A-Za-z0-9_-]+)", "sk-***", message)
        return message[:240]

    @staticmethod
    def _chinese_error_hint(safe_error: str) -> str:
        if "APIConnectionError" in safe_error:
            return (
                "\u65e0\u6cd5\u8fde\u63a5\u5230 LLM API\u3002"
                "\u8bf7\u786e\u8ba4 OPENAI_API_BASE \u4e3a https://api.siliconflow.cn/v1\uff0c"
                "\u5e76\u67e5\u770b Action \u65e5\u5fd7\u4e2d\u7684 LLM connectivity precheck\u3002"
            )
        return "\u4e2d\u6587\u6458\u8981\u751f\u6210\u5931\u8d25\uff0c\u8bf7\u68c0\u67e5 LLM \u914d\u7f6e\u6216\u8fd0\u884c\u65e5\u5fd7\u3002"

    def _repair_bilingual_tldr(self, openai_client:OpenAI, llm_params:dict, current_tldr:str) -> str:
        prompt = (
            "The previous answer did not follow the required bilingual format.\n"
            "Rewrite it into exactly two concise lines, preserving the scientific meaning:\n"
            "English: <one-sentence English TLDR>\n"
            f"{ZH_LABEL}: <one-sentence Simplified Chinese TLDR>\n\n"
        )
        if self.title:
            prompt += f"Title:\n{self.title}\n\n"
        if self.abstract:
            prompt += f"Abstract:\n{self.abstract}\n\n"
        prompt += f"Previous answer:\n{current_tldr}\n"

        try:
            response = openai_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self._tldr_system_prompt("English and Chinese")},
                    {"role": "user", "content": prompt},
                ],
                **self._llm_generation_kwargs(llm_params, task="tldr")
            )
        except Exception as e:
            logger.warning(f"Failed to repair bilingual tldr of {self.url}: {e}")
            return current_tldr
        repaired_tldr = response.choices[0].message.content
        if self._has_bilingual_tldr(repaired_tldr):
            return repaired_tldr
        return current_tldr

    @staticmethod
    def _clean_title_translation(title: str) -> str:
        title = (title or "").strip().strip("`").strip().strip('"').strip("'").strip()
        lines = [line.strip().strip("-*").strip() for line in title.splitlines() if line.strip()]
        if lines:
            chinese_lines = [line for line in lines if contains_chinese(line)]
            title = chinese_lines[0] if chinese_lines else lines[0]
        label_pattern = (
            "^(?:Chinese|Simplified Chinese|Chinese title|Translated title|Translation|"
            "\u4e2d\u6587|\u4e2d\u6587\u6807\u9898|\u6807\u9898|\u8bd1\u6587|\u7ffb\u8bd1)"
            "\\s*[:\uff1a]\\s*"
        )
        title = re.sub(label_pattern, "", title, flags=re.IGNORECASE)
        return title.strip().strip('"').strip("'").strip()

    def _generate_title_translation_with_llm(self, openai_client:OpenAI, llm_params:dict) -> str:
        if not self.title:
            return ""
        if contains_chinese(self.title):
            return self.title

        prompt = (
            "Translate the following scientific paper title into Simplified Chinese.\n"
            "Return only the translated title in Simplified Chinese. The answer must contain Chinese characters.\n"
            "Do not add quotes, labels, explanations, or markdown.\n"
            "Preserve technical acronyms, formulas, model names, journal abbreviations, and proper nouns when appropriate.\n\n"
            f"{glossary_hint_for(self.title)}"
            f"\nTitle:\n{self.title}"
        )
        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You translate scientific paper titles accurately into concise Simplified Chinese. "
                        "You are familiar with RF and microwave engineering terminology. "
                        "Return only the Chinese title."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            **self._llm_generation_kwargs(llm_params, task="title")
        )
        return self._clean_title_translation(response.choices[0].message.content)

    def _repair_title_translation(self, openai_client:OpenAI, llm_params:dict, current_answer:str) -> str:
        prompt = (
            "The previous answer did not produce a Simplified Chinese paper title.\n"
            "Translate the title below into Simplified Chinese now.\n"
            "Return only the translated Chinese title, with no labels, quotes, explanations, or markdown.\n"
            "Preserve technical acronyms, formulas, model names, journal abbreviations, and proper nouns when appropriate.\n\n"
            f"Title:\n{self.title}\n\n"
            f"Previous answer:\n{current_answer}"
        )
        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You must answer with only a Simplified Chinese scientific paper title.",
                },
                {"role": "user", "content": prompt},
            ],
            **self._llm_generation_kwargs(llm_params, task="title")
        )
        return self._clean_title_translation(response.choices[0].message.content)

    def generate_title_translation(self, openai_client:OpenAI, llm_params:dict) -> Optional[str]:
        try:
            title_zh = self._generate_title_translation_with_llm(openai_client, llm_params)
            if not contains_chinese(title_zh):
                title_zh = self._repair_title_translation(openai_client, llm_params, title_zh)
            self.title_zh = title_zh if contains_chinese(title_zh) else TITLE_ZH_FAILURE
            return self.title_zh
        except Exception as e:
            safe_error = self._safe_error_message(e)
            logger.warning(f"Failed to translate title of {self.url}: {safe_error}")
            self.title_zh = f"{TITLE_ZH_FAILURE} Error: {safe_error}"
            return None

    @staticmethod
    def _clean_abstract_translation(text: str) -> str:
        text = (text or "").strip()
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        label_pattern = (
            # 长标签排在短标签前面，避免依赖正则回溯（"中文摘要" 必须先于 "中文" 尝试）。
            r"^(?:Simplified Chinese|Chinese abstract|Translated abstract|Translation|Chinese|"
            r"中文摘要|中文翻译|中文|译文|翻译|摘要)"
            r"\s*[:：]\s*"
        )
        text = re.sub(label_pattern, "", text.strip(), flags=re.IGNORECASE)
        return text.strip()

    def _generate_abstract_translation_with_llm(self, openai_client:OpenAI, llm_params:dict) -> str:
        if not self.abstract or self.abstract_is_placeholder:
            return ""
        if contains_chinese(self.abstract):
            return self.abstract

        prompt = (
            "Translate the following scientific paper abstract into Simplified Chinese.\n"
            "Requirements:\n"
            "- Translate the abstract in full. Do not summarize, condense, or omit sentences.\n"
            "- Return only the Chinese translation: no labels, no markdown, no commentary.\n"
            "- Keep numbers, units, frequency bands, device models, formulas and acronyms exactly as written.\n\n"
            f"{glossary_hint_for(self.title, self.abstract)}"
            f"\nAbstract:\n{self.abstract}"
        )
        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a translator specialized in RF, microwave and integrated circuit papers. "
                        "You translate abstracts into accurate, fluent Simplified Chinese and return nothing else."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            **self._llm_generation_kwargs(llm_params, task="abstract")
        )
        return self._clean_abstract_translation(response.choices[0].message.content)

    def generate_abstract_translation(self, openai_client:OpenAI, llm_params:dict) -> Optional[str]:
        """Translate the abstract into Chinese. Non-fatal: returns ``None`` on failure."""
        try:
            abstract_zh = self._generate_abstract_translation_with_llm(openai_client, llm_params)
        except Exception as e:
            safe_error = self._safe_error_message(e)
            logger.warning(f"Failed to translate abstract of {self.url}: {safe_error}")
            self.abstract_zh = None
            return None

        self.abstract_zh = abstract_zh if contains_chinese(abstract_zh) else None
        return self.abstract_zh
    
    def generate_tldr(self, openai_client:OpenAI,llm_params:dict) -> str:
        try:
            tldr = self._generate_tldr_with_llm(openai_client,llm_params)
            self.tldr = tldr
            return tldr
        except Exception as e:
            safe_error = self._safe_error_message(e)
            logger.warning(f"Failed to generate tldr of {self.url}: {safe_error}")
            if wants_bilingual_tldr(llm_params.get('language', 'English')):
                tldr = (
                    f"English: {self.abstract}\n"
                    f"{ZH_LABEL}: {self._chinese_error_hint(safe_error)} Error: {safe_error}"
                )
            else:
                tldr = self.abstract
            self.tldr = tldr
            return tldr

    def _generate_affiliations_with_llm(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        if self.full_text is not None:
            prompt = f"Given the beginning of a paper, extract the affiliations of the authors in a python list format, which is sorted by the author order. If there is no affiliation found, return an empty list '[]':\n\n{self.full_text}"
            # use gpt-4o tokenizer for estimation
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:2000]  # truncate to 2000 tokens
            prompt = enc.decode(prompt_tokens)
            affiliations = openai_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant who perfectly extracts affiliations of authors from a paper. You should return a python list of affiliations sorted by the author order, like [\"TsingHua University\",\"Peking University\"]. If an affiliation is consisted of multi-level affiliations, like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation 'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations, and do not return any intermediate results.",
                    },
                    {"role": "user", "content": prompt},
                ],
                **self._llm_generation_kwargs(llm_params, task="affiliations")
            )
            affiliations = affiliations.choices[0].message.content

            affiliations = re.search(r'\[.*?\]', affiliations, flags=re.DOTALL).group(0)
            affiliations = json.loads(affiliations)
            affiliations = list(set(affiliations))
            affiliations = [str(a) for a in affiliations]

            return affiliations
    
    def generate_affiliations(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        try:
            affiliations = self._generate_affiliations_with_llm(openai_client,llm_params)
            self.affiliations = affiliations
            return affiliations
        except Exception as e:
            logger.warning(f"Failed to generate affiliations of {self.url}: {e}")
            self.affiliations = None
            return None
@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]
