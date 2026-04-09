import time

from src.memory.vector_store import VectorStore
from src.inference.embedder import Embedder
from src.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        limit: int = 3,
        category: str | None = None,
        score_threshold: float = 0.50,
    ) -> list[dict]:
        t0 = time.time()
        query_vector = self.embedder.embed(query)
        embed_ms = round((time.time() - t0) * 1000)

        results = self.vector_store.search(
            query_vector=query_vector,
            limit=limit,
            category=category,
        )
        filtered = [r for r in results if r.get("score", 0) >= score_threshold]

        logger.info(
            "rag_retrieve",
            extra={
                "query": query[:80],
                "category": category,
                "embed_ms": embed_ms,
                "results_total": len(results),
                "results_above_threshold": len(filtered),
                "score_threshold": score_threshold,
            },
        )
        return filtered

    def format_context(self, results: list[dict]) -> str:
        if not results:
            return ""

        lines = ["Relevant information:"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('name', 'Unknown')}: {r.get('description', '')}")
            if "features" in r:
                lines.append(f"   Features: {', '.join(r['features'][:3])}")
        return "\n".join(lines)
