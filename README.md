# MBA AI - PDF Search

Semantic ingestion and search over a PDF using **LangChain** + **PostgreSQL/pgVector**.

The project does two things:

1. **Ingestion** — reads a PDF, splits it into chunks, generates embeddings and stores
   the vectors in PostgreSQL (pgVector extension).
2. **Search** — a CLI chat where the user asks questions and gets answers based **only**
   on the PDF content.

> **Note:** the prompt and the chat messages are tailored for **Portuguese (pt-BR)**.
> Questions and answers are expected to be in Portuguese.

## Project structure

```
├── docker-compose.yml      # PostgreSQL + pgVector
├── requirements.txt        # Dependencies
├── .env.example            # Environment variables template
├── src/
│   ├── config.py           # Provider selection (OpenAI/Gemini) + vector store
│   ├── ingest.py           # PDF ingestion script
│   ├── search.py           # Search chain (similarity_search_with_score, k=10)
│   └── chat.py             # CLI for user interaction
├── document.pdf            # PDF to ingest
└── README.md
```

## Requirements

- Python 3.10+
- Docker & Docker Compose
- An OpenAI API key (default) **or** a Google (Gemini) API key

## Setup

### 1. Environment variables

Copy the template and fill in your keys:

```bash
cp .env.example .env
```

Key variables:

- `PROVIDER` — `openai` (default) or `google`. Controls **both** embeddings and the LLM.
  It must be the same for ingestion and search.
- `OPENAI_API_KEY` / `GOOGLE_API_KEY` — your API key for the chosen provider.
- `DATABASE_URL` — already points to the Docker database
  (`postgresql+psycopg://postgres:postgres@localhost:5432/rag`).
- `PG_VECTOR_COLLECTION_NAME` — collection name (e.g. `pdf_documents`).
- `PDF_PATH` — path to the PDF (`./document.pdf`).
- `SIMILARITY_THRESHOLD` *(optional)* — max distance for a chunk to count as
  relevant (lower = stricter). Leave empty to disable.

> ⚠️ Use the **same `PROVIDER` for ingestion and search**. The OpenAI and Gemini
> embeddings have different dimensions, so searching with a different provider than
> the one used to ingest raises a clear dimension-mismatch error — just re-run
> `python src/ingest.py` with the desired provider.

Models used:

| Provider | Embedding                | LLM                     |
|----------|--------------------------|-------------------------|
| OpenAI   | `text-embedding-3-small` | `gpt-5-nano`            |
| Gemini   | `models/gemini-embedding-001` | `gemini-2.5-flash-lite` |

### 2. Virtual environment & dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Execution order

### 1. Start the database

```bash
docker compose up -d
```

This starts PostgreSQL and automatically enables the `vector` extension.

### 2. Run the PDF ingestion

```bash
python src/ingest.py
```

Splits the PDF into 1000-character chunks with 150 overlap, embeds them and stores the
vectors. Re-running is safe — the collection is recreated on each run, so the stored
vectors always reflect the current PDF (no leftover chunks).

> **No manual cleanup needed.** Ingestion recreates the collection automatically
> (`pre_delete_collection=True`). To re-ingest the same PDF for a different model or
> provider, just change `PROVIDER` in `.env` and run `python src/ingest.py` again — no
> need to drop tables or reset the database. (Different providers produce vectors of
> different dimensions, so switching `PROVIDER` *requires* re-ingesting.)

### 3. Run the chat

```bash
python src/chat.py
```

Example:

```
Faça sua pergunta:

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

Type `sair` (or `exit` / `quit`) to leave the chat.

## How it works

On each question the chat:

1. vectorizes the question;
2. fetches the 10 most relevant chunks via `similarity_search_with_score(query, k=10)`;
3. concatenates them into the prompt's `CONTEXTO`;
4. calls the LLM, which is instructed to answer **only** from the context — otherwise it
   replies *"Não tenho informações necessárias para responder sua pergunta."*

> **The chat is stateless (no conversation memory).** Each question is handled
> independently — no previous questions or answers are sent to the model, and retrieval
> uses only the current question. Follow-ups that rely on earlier turns (e.g. *"and the
> second highest?"*) won't work; ask the full, self-contained question instead. This is
> intentional: it keeps the "answer only from the PDF" grounding clean.
