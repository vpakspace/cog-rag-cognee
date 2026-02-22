"""Streamlit UI for cog-rag-cognee — 4 tabs."""
from __future__ import annotations

import asyncio

import streamlit as st

from cog_rag_cognee.cognee_setup import apply_cognee_env
from cog_rag_cognee.config import get_settings
from cog_rag_cognee.service import PipelineService
from ui.components.graph_viz import render_graph
from ui.i18n import get_translator

# --- Page config ---
st.set_page_config(page_title="Cog-RAG Cognee", page_icon="🧠", layout="wide")

# --- Sidebar ---
lang = st.sidebar.selectbox("Language / Язык", ["en", "ru"], index=0)
t = get_translator(lang)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**{t('app_title')}** v0.1.0")
st.sidebar.markdown("100% local stack")


@st.cache_resource
def init_service() -> PipelineService:
    settings = get_settings()
    apply_cognee_env(settings)
    return PipelineService()


svc = init_service()

# --- Tabs ---
tab_upload, tab_search, tab_graph, tab_settings = st.tabs(
    [t("tab_upload"), t("tab_search"), t("tab_graph"), t("tab_settings")]
)

# --- Tab 1: Upload ---
with tab_upload:
    st.header(t("upload_header"))

    uploaded_file = st.file_uploader(
        t("upload_drag"), type=["pdf", "txt", "md", "docx", "html"]
    )
    text_input = st.text_area(t("upload_text"), height=200)

    if st.button(t("upload_btn")):
        with st.spinner("Processing..."):
            if text_input.strip():
                result = asyncio.run(svc.add_text(text_input))
                cognify_result = asyncio.run(svc.cognify())
                st.success(t("upload_success"))
                st.json({"ingest": result, "cognify": str(cognify_result)})
            elif uploaded_file is not None:
                content = uploaded_file.read().decode("utf-8", errors="ignore")
                result = asyncio.run(svc.add_text(content))
                cognify_result = asyncio.run(svc.cognify())
                st.success(t("upload_success"))
                st.json({"ingest": result, "cognify": str(cognify_result)})
            else:
                st.warning("Please upload a file or enter text.")

# --- Tab 2: Search & Q&A ---
with tab_search:
    st.header(t("search_header"))

    search_mode = st.selectbox(
        t("search_mode"),
        ["GRAPH_COMPLETION", "RAG_COMPLETION", "CHUNKS", "SUMMARIES"],
        index=0,
    )

    query = st.text_input(t("search_placeholder"))

    if st.button(t("search_btn")) and query:
        with st.spinner("Searching..."):
            qa = asyncio.run(svc.query(query, search_type=search_mode))

            st.subheader(t("search_answer"))
            st.text(qa.answer)

            col1, col2 = st.columns(2)
            with col1:
                st.metric(t("search_confidence"), f"{qa.confidence:.0%}")
            with col2:
                st.metric(t("search_results"), len(qa.sources))

            if qa.sources:
                with st.expander(t("search_sources")):
                    for i, src in enumerate(qa.sources, 1):
                        st.markdown(f"**{i}.** [{src.score:.2f}] {src.content[:200]}")

# --- Tab 3: Graph Explorer ---
with tab_graph:
    st.header(t("graph_header"))
    render_graph([], [])

# --- Tab 4: Settings ---
with tab_settings:
    st.header(t("settings_header"))

    settings = get_settings()
    st.subheader(t("settings_config"))
    st.json(
        {
            "llm": f"{settings.llm_provider}/{settings.llm_model}",
            "embeddings": f"{settings.embedding_provider}/{settings.embedding_model}",
            "graph_db": settings.graph_database_provider,
            "vector_db": settings.vector_db_provider,
        }
    )

    st.markdown("---")
    if st.button(t("settings_clear"), type="secondary"):
        if st.checkbox(t("settings_clear_confirm")):
            asyncio.run(svc.reset())
            st.success("Data cleared!")
