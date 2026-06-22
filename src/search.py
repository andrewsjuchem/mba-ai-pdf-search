from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from config import SIMILARITY_THRESHOLD, get_llm, get_vector_store

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def _format_chunk(document):
    """Render a chunk with a light source reference (1-based page number)."""
    page = document.metadata.get("page")
    label = f"[página {page + 1}]" if isinstance(page, int) else "[trecho]"
    return f"{label}\n{document.page_content}"


def search_prompt(question=None):
    """Build the RAG chain.

    The returned chain takes a question (string) and:
      1. vectorizes it and fetches the 10 most relevant chunks (k=10);
      2. concatenates them into the CONTEXTO;
      3. fills the prompt and calls the LLM;
      4. returns the answer as plain text.
    """
    store = get_vector_store()
    llm = get_llm()
    prompt = PromptTemplate(
        input_variables=["contexto", "pergunta"],
        template=PROMPT_TEMPLATE,
    )

    def retrieve_context(query: str) -> str:
        try:
            results = store.similarity_search_with_score(query, k=10)
        except Exception as error:  # noqa: BLE001
            if "dimension" in str(error).lower():
                raise RuntimeError(
                    "Incompatibilidade de dimensões entre a pergunta e os vetores "
                    "armazenados. Isso normalmente ocorre quando o PROVIDER (ou o "
                    "modelo de embedding) usado na busca é diferente do usado na "
                    "ingestão. Verifique a variável PROVIDER no .env e use o mesmo "
                    "provedor da ingestão, ou rode novamente 'python src/ingest.py' "
                    "para recriar a coleção com o provedor atual."
                ) from error
            raise

        if SIMILARITY_THRESHOLD is not None:
            # Lower distance = more similar; keep only chunks within the limit.
            results = [
                (doc, score)
                for doc, score in results
                if score <= SIMILARITY_THRESHOLD
            ]

        return "\n\n".join(_format_chunk(document) for document, _score in results)

    chain = (
        {
            "contexto": RunnableLambda(retrieve_context),
            "pergunta": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
