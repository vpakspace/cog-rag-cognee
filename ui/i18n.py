"""Internationalization: EN/RU translations."""
from __future__ import annotations

from typing import Callable

TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "app_title": "Cog-RAG Cognee",
        "tab_upload": "Upload",
        "tab_search": "Search & Q&A",
        "tab_graph": "Graph Explorer",
        "tab_settings": "Settings",
        "upload_header": "Upload Documents",
        "upload_drag": "Drag and drop files here",
        "upload_text": "Or paste text below",
        "upload_btn": "Ingest & Cognify",
        "upload_success": "Document ingested successfully!",
        "search_header": "Search Knowledge Graph",
        "search_placeholder": "Ask a question...",
        "search_btn": "Search",
        "search_mode": "Search mode",
        "search_results": "Results",
        "search_answer": "Answer",
        "search_confidence": "Confidence",
        "search_sources": "Sources",
        "graph_header": "Knowledge Graph Explorer",
        "graph_nodes": "Nodes",
        "graph_edges": "Edges",
        "settings_header": "Settings",
        "settings_config": "Current Configuration",
        "settings_clear": "Clear all data",
        "settings_clear_confirm": "Are you sure? This will delete all data.",
        "health_ok": "All services running",
        "health_fail": "Some services unavailable",
        "language": "Language",
    },
    "ru": {
        "app_title": "Cog-RAG Cognee",
        "tab_upload": "Загрузка",
        "tab_search": "Поиск и Q&A",
        "tab_graph": "Граф знаний",
        "tab_settings": "Настройки",
        "upload_header": "Загрузка документов",
        "upload_drag": "Перетащите файлы сюда",
        "upload_text": "Или вставьте текст ниже",
        "upload_btn": "Загрузить и обработать",
        "upload_success": "Документ успешно загружен!",
        "search_header": "Поиск по графу знаний",
        "search_placeholder": "Задайте вопрос...",
        "search_btn": "Найти",
        "search_mode": "Режим поиска",
        "search_results": "Результаты",
        "search_answer": "Ответ",
        "search_confidence": "Уверенность",
        "search_sources": "Источники",
        "graph_header": "Граф знаний",
        "graph_nodes": "Узлы",
        "graph_edges": "Связи",
        "settings_header": "Настройки",
        "settings_config": "Текущая конфигурация",
        "settings_clear": "Очистить все данные",
        "settings_clear_confirm": "Вы уверены? Все данные будут удалены.",
        "health_ok": "Все сервисы работают",
        "health_fail": "Некоторые сервисы недоступны",
        "language": "Язык",
    },
}


def get_translator(lang: str = "en") -> Callable[[str], str]:
    """Return a translation function for the given language."""
    translations = TRANSLATIONS.get(lang, TRANSLATIONS["en"])

    def translate(key: str) -> str:
        return translations.get(key, key)

    return translate
