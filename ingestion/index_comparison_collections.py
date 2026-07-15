import logging
from pathlib import Path

from ingestion.index_to_qdrant import index_all_filings

BASIC_COLLECTION = "sec_filings_basic"
ADVANCED_COLLECTION = "sec_filings_advanced"


def index_comparison_collections() -> None:
    index_all_filings(collection_name=BASIC_COLLECTION, chunks_dir=Path("data/chunks_basic"))
    index_all_filings(collection_name=ADVANCED_COLLECTION, chunks_dir=Path("data/chunks_advanced"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    index_comparison_collections()
