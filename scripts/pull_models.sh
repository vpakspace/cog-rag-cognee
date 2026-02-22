#!/usr/bin/env bash
# Pull required Ollama models for Cog-RAG Cognee.
set -euo pipefail

echo "Pulling LLM model: llama3.1:8b ..."
ollama pull llama3.1:8b

echo "Pulling embedding model: nomic-embed-text ..."
ollama pull nomic-embed-text

echo "Done. Models ready."
