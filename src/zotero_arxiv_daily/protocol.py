from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json
RawPaperItem = TypeVar('RawPaperItem')

ZH_LABEL = "\u4e2d\u6587"
DEFAULT_TLDR_MAX_TOKENS = 512
SILICONFLOW_DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def wants_bilingual_tldr(language: str) -> bool:
    language = str(language or "").lower()
    has_english = "english" in language or "\u82f1\u6587" in language or "\u82f1" in language
    has_chinese = "chinese" in language or "\u4e2d\u6587" in language or "\u4e2d" in language or "zh" in language
    return has_english and has_chinese


def contains_chinese(text: str | None) -> bool:
    return bool(re.search("[\\u4e00-\\u9fff]", text or ""))

@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    tldr: Optional[str] = None
    affiliations: Optional[list[str]] = None
    score: Optional[float] = None

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
            **self._llm_generation_kwargs(llm_params)
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
    def _llm_generation_kwargs(llm_params: dict) -> dict:
        kwargs = dict(Paper._config_get(llm_params, "generation_kwargs", {}) or {})
        api_config = Paper._config_get(llm_params, "api", {}) or {}
        base_url = str(Paper._config_get(api_config, "base_url", "") or "").lower()

        if kwargs.get("model") == "gpt-4o-mini" and "siliconflow" in base_url:
            kwargs["model"] = SILICONFLOW_DEFAULT_MODEL

        try:
            max_tokens = int(kwargs.get("max_tokens", DEFAULT_TLDR_MAX_TOKENS))
        except (TypeError, ValueError):
            max_tokens = DEFAULT_TLDR_MAX_TOKENS
        if max_tokens > DEFAULT_TLDR_MAX_TOKENS:
            kwargs["max_tokens"] = DEFAULT_TLDR_MAX_TOKENS
        else:
            kwargs["max_tokens"] = max_tokens

        return kwargs

    @staticmethod
    def _has_bilingual_tldr(tldr: str | None) -> bool:
        return bool(tldr and "English:" in tldr and f"{ZH_LABEL}:" in tldr and contains_chinese(tldr))

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
                **self._llm_generation_kwargs(llm_params)
            )
        except Exception as e:
            logger.warning(f"Failed to repair bilingual tldr of {self.url}: {e}")
            return current_tldr
        repaired_tldr = response.choices[0].message.content
        if self._has_bilingual_tldr(repaired_tldr):
            return repaired_tldr
        return current_tldr
    
    def generate_tldr(self, openai_client:OpenAI,llm_params:dict) -> str:
        try:
            tldr = self._generate_tldr_with_llm(openai_client,llm_params)
            self.tldr = tldr
            return tldr
        except Exception as e:
            logger.warning(f"Failed to generate tldr of {self.url}: {e}")
            if wants_bilingual_tldr(llm_params.get('language', 'English')):
                tldr = (
                    f"English: {self.abstract}\n"
                    f"{ZH_LABEL}: \u4e2d\u6587\u6458\u8981\u751f\u6210\u5931\u8d25\uff0c\u8bf7\u68c0\u67e5 LLM \u914d\u7f6e\u6216\u8fd0\u884c\u65e5\u5fd7\u3002"
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
                **self._llm_generation_kwargs(llm_params)
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
