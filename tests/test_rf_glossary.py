"""Tests for rf_glossary: selective injection of RF terminology into prompts."""

from zotero_arxiv_daily.rf_glossary import RF_GLOSSARY, glossary_hint_for


def test_hint_includes_only_terms_present_in_the_text():
    hint = glossary_hint_for("A Doherty Power Amplifier", "Improves back-off efficiency.")

    assert "多赫蒂" in hint
    assert "功率放大器" in hint
    assert "功率回退" in hint
    # 文中没出现的术语不应被塞进 prompt，否则每篇都要为整张表付 token。
    assert "包络跟踪" not in hint
    assert "史密斯圆图" not in hint


def test_hint_is_empty_for_non_rf_text():
    assert glossary_hint_for("A study of protein folding", "Molecular biology methods.") == ""


def test_hint_is_empty_for_blank_input():
    assert glossary_hint_for("", None) == ""


def test_hint_lists_tokens_to_keep_untranslated():
    hint = glossary_hint_for("A GaN Doherty PA", "AM-AM and AM-PM characteristics at 28 GHz.")

    assert "do not translate" in hint.lower()
    assert "AM-AM" in hint
    assert "AM-PM" in hint
    assert "GHz" in hint


def test_hint_is_case_insensitive():
    assert "多赫蒂" in glossary_hint_for("a doherty amplifier")
    assert "多赫蒂" in glossary_hint_for("A DOHERTY AMPLIFIER")


def test_glossary_has_no_empty_entries():
    assert all(term.strip() and zh.strip() for term, zh in RF_GLOSSARY.items())
