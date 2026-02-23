"""Streamlit UI for cog-rag-cognee — 4 tabs, all data via API."""
from __future__ import annotations

import httpx
import streamlit as st

from cog_rag_cognee.config import get_settings
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

# --- API base URL ---
_settings = get_settings()
API_BASE = f"http://{_settings.api_host}:{_settings.api_port}/api/v1"


@st.cache_resource
def _get_http_client() -> httpx.Client:
    """Return a cached httpx.Client with base_url and optional auth header."""
    headers: dict[str, str] = {}
    if _settings.api_key:
        headers["X-API-Key"] = _settings.api_key
    return httpx.Client(base_url=API_BASE, headers=headers)


def _safe_json(resp: httpx.Response) -> dict | list | None:
    """Parse JSON response safely, return None on non-JSON or decode error."""
    content_type = resp.headers.get("content-type", "")
    if "application/json" not in content_type:
        return None
    try:
        return resp.json()
    except Exception:
        return None


# --- Sidebar health status ---
try:
    _health_resp = _get_http_client().get("/health", timeout=3)
    if _health_resp.status_code == 200:
        _health = _safe_json(_health_resp)
        if _health is None:
            st.sidebar.markdown(f"⚠️ {t('health_fail')}")
        else:
            _neo4j_icon = "🟢" if _health.get("neo4j") else "🔴"
            _ollama_icon = "🟢" if _health.get("ollama") else "🔴"
            st.sidebar.markdown(f"{_neo4j_icon} Neo4j  {_ollama_icon} Ollama")
    else:
        st.sidebar.markdown(f"⚠️ {t('health_fail')}")
except Exception:
    st.sidebar.markdown(f"⚠️ {t('health_fail')}")


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
                try:
                    resp = _get_http_client().post(
                        "/ingest",
                        json={"text": text_input},
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        st.success(t("upload_success"))
                        st.json(resp.json())
                    else:
                        st.error(f"{t('upload_error')}: {resp.text}")
                except Exception as exc:
                    st.error(f"{t('upload_error')}: {exc}")
            elif uploaded_file is not None:
                try:
                    file_bytes = uploaded_file.read()
                    resp = _get_http_client().post(
                        "/ingest-file",
                        files={"file": (uploaded_file.name, file_bytes)},
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        st.success(t("upload_success"))
                        st.json(resp.json())
                    else:
                        st.error(f"{t('upload_error')}: {resp.text}")
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
            try:
                resp = _get_http_client().post(
                    "/query",
                    json={"text": query, "mode": search_mode},
                    timeout=60,
                )
                if resp.status_code == 200:
                    qa = resp.json()

                    st.subheader(t("search_answer"))
                    st.text(qa["answer"])

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(t("search_confidence"), f"{qa['confidence']:.0%}")
                    with col2:
                        st.metric(t("search_results"), len(qa.get("sources", [])))

                    if qa.get("sources"):
                        with st.expander(t("search_sources")):
                            for i, src in enumerate(qa["sources"], 1):
                                st.markdown(
                                    f"**{i}.** [{src['score']:.2f}] {src['content'][:200]}"
                                )
                else:
                    st.error(f"Search error: {resp.text}")
            except Exception as exc:
                st.error(f"Search error: {exc}")

# --- Tab 3: Graph Explorer ---
with tab_graph:
    st.header(t("graph_header"))

    # Fetch stats for sidebar filter
    try:
        stats_resp = _get_http_client().get("/graph/stats", timeout=5)
        stats_data = _safe_json(stats_resp) or {} if stats_resp.status_code == 200 else {}
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
                graph_params: dict[str, str] = {"limit": "200"}
                if selected_types and selected_types != entity_types_available:
                    graph_params["entity_types"] = ",".join(selected_types)
                resp = _get_http_client().get(
                    "/graph/entities",
                    params=graph_params,
                    timeout=10,
                )
                if resp.status_code == 200:
                    graph_data = _safe_json(resp)
                    if graph_data is None:
                        st.warning(t("graph_error"))
                    else:
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
    if st.button(t("settings_clear"), type="secondary"):  # noqa: SIM102
        if st.checkbox(t("settings_clear_confirm")):
            try:
                resp = _get_http_client().post(
                    "/reset",
                    json={"confirm": True},
                    timeout=30,
                )
                if resp.status_code == 200:
                    st.success("Data cleared!")
                else:
                    st.error(f"Reset failed: {resp.text}")
            except Exception as exc:
                st.error(f"Reset failed: {exc}")
