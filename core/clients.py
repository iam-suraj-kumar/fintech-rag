import atexit
import os

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_LOCAL_PATH = os.environ.get("QDRANT_LOCAL_PATH") or "data/qdrant_local"
SPARSE_MODEL = "Qdrant/bm25"
COLLECTION_NAME = "sec_filings_advanced"

_qdrant_client: QdrantClient | None = None
_sparse_model: SparseTextEmbedding | None = None


def get_qdrant_client() -> QdrantClient:
    """Return a singleton Qdrant client.

    Runs in local (embedded, on-disk) mode by default -- no server required.
    Set QDRANT_URL to point at a running Qdrant server instead.
    """
    global _qdrant_client
    if _qdrant_client is None:
        if QDRANT_URL:
            _qdrant_client = QdrantClient(url=QDRANT_URL)
        else:
            _qdrant_client = QdrantClient(path=QDRANT_LOCAL_PATH)
            # Local mode releases its file lock via a lazily-imported `portalocker` in
            # close(). Left to __del__ at interpreter shutdown, that import fails
            # (sys.meta_path is already gone) and prints a harmless but noisy
            # "Exception ignored in..." traceback. Closing explicitly at normal exit
            # avoids that.
            atexit.register(_qdrant_client.close)
    return _qdrant_client


def get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    return _sparse_model
