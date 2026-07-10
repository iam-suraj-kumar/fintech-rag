import os

import voyageai
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
DENSE_MODEL = "voyage-finance-2"
SPARSE_MODEL = "Qdrant/bm25"
COLLECTION_NAME = "sec_filings"

_qdrant_client: QdrantClient | None = None
_voyage_client: voyageai.Client | None = None
_sparse_model: SparseTextEmbedding | None = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL)
    return _qdrant_client


def get_voyage_client() -> voyageai.Client:
    global _voyage_client
    if _voyage_client is None:
        _voyage_client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    return _voyage_client


def get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    return _sparse_model
