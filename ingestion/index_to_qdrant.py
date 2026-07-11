import json
from pathlib import Path

from qdrant_client import models

from core.clients import COLLECTION_NAME, get_qdrant_client, get_sparse_model
from core.embeddings import embed_dense as _embed_dense
from core.models import FilingChunk
from ingestion.fetch_filings import COMPANIES

CHUNKS_DIR = Path("data/chunks")


def ensure_collection() -> None:
    client = get_qdrant_client()
    if client.collection_exists(COLLECTION_NAME):
        return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            # voyage-finance-2 outputs 1024-dim embeddings
            "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(),
        },
    )


def embed_dense(texts: list[str]) -> list[list[float]]:
    return _embed_dense(texts, input_type="document")


def load_chunks(ticker: str) -> list[FilingChunk]:
    path = CHUNKS_DIR / f"{ticker}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [FilingChunk(**item) for item in raw]


def index_all_filings() -> None:
    client = get_qdrant_client()
    ensure_collection()
    sparse_model = get_sparse_model()

    point_id = 0
    for ticker in COMPANIES:
        chunks = load_chunks(ticker)
        texts = [c.text for c in chunks]
        dense_vectors = embed_dense(texts)
        sparse_vectors = list(sparse_model.embed(texts))

        points = []
        for chunk, dense_vec, sparse_vec in zip(chunks, dense_vectors, sparse_vectors):
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vec,
                        "sparse": models.SparseVector(
                            indices=sparse_vec.indices.tolist(),
                            values=sparse_vec.values.tolist(),
                        ),
                    },
                    payload={
                        "ticker": chunk.ticker,
                        "company_name": chunk.company_name,
                        "filing_type": chunk.filing_type,
                        "section": chunk.section,
                        "fiscal_year": chunk.fiscal_year,
                        "text": chunk.text,
                    },
                )
            )
            point_id += 1
        client.upsert(collection_name=COLLECTION_NAME, points=points)


if __name__ == "__main__":
    index_all_filings()
