"""Streamlit UI for cog-rag-cognee — 4 tabs."""
from __future__ import annotations

import asyncio

import httpx
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
                try:
                    file_bytes = uploaded_file.read()
                    result = asyncio.run(
                        svc.add_bytes(file_bytes, uploaded_file.name)
                    )
                    cognify_result = asyncio.run(svc.cognify())
                    st.success(t("upload_success"))
                    st.json({"ingest": result, "cognify": str(cognify_result)})
                except ImportError:
                    st.error(t("upload_docling_missing"))
                except Exception as exc:
                    st.error(f"{t('upload_error')}: {exc}")
            else:
                st.warning(t("upload_no_input"))

# --- Tab 2: Search & Q&A ---
with tab_search:
    st.header(t("search_header"))

    search_mode = st.selectbox(
        t("search_mode"),
        ["CHUNKS", "GRAPH_COMPLETION", "RAG_COMPLETION", "SUMMARIES"],
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

    settings = get_settings()
    api_base = f"http://{settings.api_host}:{settings.api_port}/api/v1"

    # Fetch stats for sidebar filter
    try:
        stats_resp = httpx.get(f"{api_base}/graph/stats", timeout=5)
        stats_data = stats_resp.json() if stats_resp.status_code == 200 else {}
    except Exception:
        stats_data = {}

    entity_types_available = list(stats_data.get("entity_types", {}).keys())

    # Sidebar filter
    selected_types = st.multiselect(
        t("graph_filter"),
        options=entity_types_available,
        default=entity_types_available,
    )

    # Stats display
    if stats_data.get("nodes", 0) > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.metric(t("graph_nodes"), stats_data.get("nodes", 0))
        with col2:
            st.metric(t("graph_edges"), stats_data.get("edges", 0))

        if stats_data.get("entity_types"):
            with st.expander(t("graph_entity_breakdown")):
                for etype, count in stats_data["entity_types"].items():
                    st.text(f"{etype}: {count}")

    # Load graph data
    if st.button(t("graph_load")) or stats_data.get("nodes", 0) > 0:
        with st.spinner(t("graph_loading")):
            try:
                query: dict[str, str] = {"limit": "200"}
                if selected_types and selected_types != entity_types_available:
                    query["entity_types"] = ",".join(selected_types)
                resp = httpx.get(
                    f"{api_base}/graph/entities", params=query, timeout=10
                )
                if resp.status_code == 200:
                    graph_data = resp.json()
                    nodes = graph_data.get("nodes", [])
                    edges = graph_data.get("edges", [])
                    render_graph(nodes, edges)
                else:
                    st.warning(t("graph_error"))
            except Exception:
                st.warning(t("graph_error"))

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
