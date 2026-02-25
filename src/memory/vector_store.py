from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from config.settings import QDRANT_PATH, QDRANT_COLLECTION, EMBED_DIMENSION


class VectorStore:
    def __init__(self):
        # Embedded mode - stores data locally, no Docker needed
        QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(QDRANT_PATH))
        self.collection = QDRANT_COLLECTION

    def create_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection not in collections:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=EMBED_DIMENSION,
                    distance=Distance.COSINE,
                ),
            )

    def upsert(self, id: str, vector: list[float], payload: dict):
        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=hash(id) % (10**12),
                    vector=vector,
                    payload={"id": id, **payload},
                )
            ],
        )

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        category: str | None = None,
    ) -> list[dict]:
        filter_condition = None
        if category:
            filter_condition = Filter(
                must=[FieldCondition(key="category", match=MatchValue(value=category))]
            )

        # Use query_points for newer qdrant-client versions
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=limit,
            query_filter=filter_condition,
        )

        return [{"score": r.score, **r.payload} for r in results.points]

    def reset_collection(self):
        """Delete and recreate the collection, clearing all data."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection in collections:
            self.client.delete_collection(self.collection)
        self.create_collection()

    def get_by_id(self, id: str) -> dict | None:
        results = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="id", match=MatchValue(value=id))]
            ),
            limit=1,
        )
        points = results[0]
        return points[0].payload if points else None
