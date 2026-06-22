import hashlib
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from config import PDF_PATH, get_vector_store

# How many chunks to embed per request. Free-tier embedding APIs reject large
# bursts, so we send small batches instead of all chunks at once.
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))


def _is_rate_limit(exc):
    message = str(exc).lower()
    return any(
        token in message
        for token in ("429", "quota", "rate limit", "resourceexhausted")
    )


@retry(
    retry=retry_if_exception(_is_rate_limit),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)
def _add_batch(store, documents, ids):
    store.add_documents(documents, ids=ids)


def ingest_pdf():
    if not PDF_PATH or not os.path.exists(PDF_PATH):
        raise RuntimeError(
            f"PDF não encontrado em '{PDF_PATH}'. Configure a variável PDF_PATH no .env."
        )

    print(f"Lendo o PDF: {PDF_PATH}")
    documents = PyPDFLoader(PDF_PATH).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

    if not chunks:
        raise RuntimeError("Nenhum conteúdo foi extraído do PDF.")

    print(f"{len(chunks)} chunks gerados. Criando embeddings e salvando no banco...")

    # Deterministic ids keep the chunk identities stable across runs.
    ids = [
        hashlib.sha256(
            f"{chunk.metadata.get('source', '')}-"
            f"{chunk.metadata.get('page', '')}-"
            f"{chunk.metadata.get('start_index', i)}".encode("utf-8")
        ).hexdigest()
        for i, chunk in enumerate(chunks)
    ]

    # Recreate the collection so the stored vectors always reflect the current
    # PDF (no orphan chunks left over from a previous ingestion).
    store = get_vector_store(pre_delete_collection=True)

    total = len(chunks)
    for start in range(0, total, EMBED_BATCH_SIZE):
        end = min(start + EMBED_BATCH_SIZE, total)
        _add_batch(store, chunks[start:end], ids[start:end])
        print(f"  {end}/{total} chunks salvos.")

    print("Ingestão concluída com sucesso.")


if __name__ == "__main__":
    ingest_pdf()
