import os

from dotenv import load_dotenv

load_dotenv()

VALID_PROVIDERS = ("openai", "google")

# "openai" (default) or "google". Controls both embeddings and the chat LLM.
# The embedding provider MUST be the same for ingestion and search, otherwise
# the question vector and the stored vectors live in different spaces (and have
# different dimensions).
PROVIDER = os.getenv("PROVIDER", "openai").strip().lower()

COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "pdf_documents")
PDF_PATH = os.getenv("PDF_PATH", "./document.pdf")


def _validate_provider():
    if PROVIDER not in VALID_PROVIDERS:
        raise RuntimeError(
            f"PROVIDER inválido: '{PROVIDER}'. "
            f"Valores aceitos: {', '.join(VALID_PROVIDERS)}. "
            "Configure a variável PROVIDER no arquivo .env."
        )


def _require_env(name):
    value = os.getenv(name)
    if not value or not value.strip():
        raise RuntimeError(
            f"A variável {name} não está definida. Configure-a no arquivo .env "
            "(veja o .env.example)."
        )
    return value


def _parse_threshold():
    """Optional max distance for a chunk to be considered relevant.

    Disabled by default (returns None) so the default behaviour stays exactly
    spec-compliant: the 10 nearest chunks are always passed to the LLM and the
    grounding is enforced by the prompt. When set, chunks farther than the
    threshold are dropped before building the context.
    """
    raw = os.getenv("SIMILARITY_THRESHOLD")
    if not raw or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        raise RuntimeError(
            f"SIMILARITY_THRESHOLD inválido: '{raw}'. "
            "Use um número (ex.: 0.5) ou deixe vazio para desabilitar."
        )


SIMILARITY_THRESHOLD = _parse_threshold()


def get_embeddings():
    """Return the embedding model for the configured PROVIDER."""
    _validate_provider()
    if PROVIDER == "google":
        _require_env("GOOGLE_API_KEY")
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
        )

    _require_env("OPENAI_API_KEY")
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )


def get_llm():
    """Return the chat LLM for the configured PROVIDER."""
    _validate_provider()
    if PROVIDER == "google":
        _require_env("GOOGLE_API_KEY")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash-lite"),
            temperature=0,
        )

    _require_env("OPENAI_API_KEY")
    from langchain_openai import ChatOpenAI

    # NOTE: gpt-5-* models only accept the default temperature, so we don't
    # override it here.
    return ChatOpenAI(model=os.getenv("OPENAI_LLM_MODEL", "gpt-5-nano"))


def get_vector_store(embeddings=None, pre_delete_collection=False):
    """Return a PGVector store bound to the configured collection.

    Pass ``pre_delete_collection=True`` (used by the ingestion script) to wipe
    the collection first, guaranteeing the stored vectors always reflect the
    current PDF instead of leaving orphan chunks from a previous run.
    """
    connection = _require_env("DATABASE_URL")

    from langchain_postgres import PGVector

    return PGVector(
        embeddings=embeddings or get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=connection,
        use_jsonb=True,
        pre_delete_collection=pre_delete_collection,
    )
