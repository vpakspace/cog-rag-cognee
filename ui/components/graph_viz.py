"""PyVis interactive graph visualization."""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st


def render_graph(nodes: list[dict], edges: list[dict]) -> None:
    """Render interactive graph using PyVis."""
    if not nodes:
        st.info("No graph data available. Ingest documents first.")
        return

    try:
        from pyvis.network import Network
    except ImportError:
        st.error("PyVis not installed: pip install pyvis")
        return

    import streamlit.components.v1 as components

    net = Network(height="600px", width="100%", directed=True)
    net.barnes_hut()

    color_map = {
        "Person": "#e74c3c",
        "Organization": "#3498db",
        "Location": "#2ecc71",
        "Date": "#f39c12",
        "Document": "#9b59b6",
        "Chunk": "#95a5a6",
    }

    for node in nodes:
        color = color_map.get(node.get("type", ""), "#bdc3c7")
        net.add_node(
            node["id"],
            label=node.get("label", node["id"]),
            color=color,
            title=f"{node.get('type', 'Unknown')}: {node.get('label', '')}",
        )

    for edge in edges:
        net.add_edge(edge["source"], edge["target"], label=edge.get("type", ""))

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        html_content = Path(f.name).read_text()
        components.html(html_content, height=620)
