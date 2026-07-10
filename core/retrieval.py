from qdrant_client import models

from core.clients import (
    COLLECTION_NAME,
    DENSE_MODEL,
    get_qdrant_client,
    get_sparse_model,
    get_voyage_client,
)
from core.models import RetrievedChunk


def embed_query_dense(query: str) -> list[float]:
    result = get_voyage_client().embed([query], model=DENSE_MODEL, input_type="query")
    return result.embeddings[0]


def embed_query_sparse(query: str) -> models.SparseVector:
    embedding = next(get_sparse_model().embed([query]))
    return models.SparseVector(
        indices=embedding.indices.tolist(), values=embedding.values.tolist()
    )


def hybrid_search(query: str, top_k: int = 8) -> list[RetrievedChunk]:
    """Run hybrid (dense + sparse) search over the sec_filings collection, fused via RRF."""
    dense_vector = embed_query_dense(query)
    sparse_vector = embed_query_sparse(query)

    results = get_qdrant_client().query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=dense_vector, using="dense", limit=top_k * 2),
            models.Prefetch(query=sparse_vector, using="sparse", limit=top_k * 2),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
    )

    return [
        RetrievedChunk(
            ticker=point.payload["ticker"],
            company_name=point.payload["company_name"],
            filing_type=point.payload["filing_type"],
            section=point.payload["section"],
            fiscal_year=point.payload["fiscal_year"],
            text=point.payload["text"],
            score=point.score,
        )
        for point in results.points
    ]
