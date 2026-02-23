"""PyVis interactive graph visualization."""
from __future__ import annotations

import os
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

    seen_nodes: set[str] = set()
    for node in nodes:
        node_id = node.get("label", str(node.get("id", "")))
        if node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)
        color = color_map.get(node.get("type", ""), "#bdc3c7")
        net.add_node(
            node_id,
            label=node_id,
            color=color,
            title=f"{node.get('type', 'Unknown')}: {node_id}",
        )

    for edge in edges:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        # Auto-add nodes referenced by edges but not in node list
        for nid in (src, tgt):
            if nid and nid not in seen_nodes:
                seen_nodes.add(nid)
                net.add_node(nid, label=nid, color="#bdc3c7")
        if src and tgt:
            net.add_edge(src, tgt, label=edge.get("type", ""))

    fd, tmp_name = tempfile.mkstemp(suffix=".html")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        net.save_graph(str(tmp))
        html_content = tmp.read_text()
        components.html(html_content, height=620)
    finally:
        tmp.unlink(missing_ok=True)
