#!/usr/bin/env python3
"""
Ingest products.txt into Qdrant vector store.
Clears existing vector data before re-ingesting.

Usage:
    python scripts/ingest_products.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import BASE_DIR
from src.inference.embedder import Embedder
from src.memory.vector_store import VectorStore
from src.rag.ingest import ingest_products

PRODUCTS_PATH = BASE_DIR / "products.txt"


def main():
    print("=" * 50)
    print("  Products Ingest — Qdrant Vector Store")
    print("=" * 50)

    if not PRODUCTS_PATH.exists():
        print(f"ERROR: {PRODUCTS_PATH} not found.")
        sys.exit(1)

    print("\n[1/3] Resetting Qdrant collection...")
    vs = VectorStore()
    vs.reset_collection()
    print("      Done.")

    print("\n[2/3] Loading embedding model...")
    embedder = Embedder()
    embedder.load()
    print("      Done.")

    print("\n[3/3] Ingesting products.txt...")
    count = ingest_products(str(PRODUCTS_PATH), embedder, vs)
    print(f"      Done. {count} products ingested.\n")


if __name__ == "__main__":
    main()
