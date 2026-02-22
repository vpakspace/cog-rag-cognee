"""Tests for i18n translations."""


def test_english_translator():
    from ui.i18n import get_translator

    t = get_translator("en")
    assert t("app_title") == "Cog-RAG Cognee"
    assert t("tab_upload") == "Upload"
    assert t("tab_search") == "Search & Q&A"


def test_russian_translator():
    from ui.i18n import get_translator

    t = get_translator("ru")
    assert t("app_title") == "Cog-RAG Cognee"
    assert t("tab_upload") == "Загрузка"
    assert t("tab_search") == "Поиск и Q&A"


def test_missing_key_fallback():
    from ui.i18n import get_translator

    t = get_translator("en")
    assert t("nonexistent_key") == "nonexistent_key"
