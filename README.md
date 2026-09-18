# RAG Generator

A small Retrieval-Augmented Generation system: upload documents at runtime
through a web UI, they get chunked/embedded/indexed into ChromaDB, and you
can ask questions that are answered **only** from what's actually in the
documents — with a citation (source file + chunk snippet) for every claim,
and an honest "I couldn't find that" when nothing relevant is indexed.

No document set is hardcoded. Everything about which LLM/embedding provider
to use is a `.env` change, not a code change.

## What it does

- Upload PDF / TXT / MD / DOCX files through the browser, into a named
  collection you pick or create.
- Each file is chunked, embedded, and stored in a persistent ChromaDB
  collection.
- Ask a question against a collection: it's embedded, the closest chunks
  are retrieved, and an LLM answers using **only** those chunks — never
  outside knowledge.
- Every answer comes with citations (source filename, chunk index, and the
  actual snippet used) pulled straight from the vector store's metadata,
  not parsed out of the model's text.
- If retrieval finds nothing relevant, you get "I couldn't find that in the
  provided documents" — and the LLM is never even called for that case.
- Multiple named collections can coexist (e.g. one per document set), and
  you can switch or create one from the UI.

## Architecture

```
                 ┌───────────────────────────┐
                 │      Browser (JS)          │
                 │ templates/index.html       │
                 │ static/app.js               │
                 └─────────────┬─────────────┘
                               │ fetch() — JSON / multipart
                               ▼
                 ┌───────────────────────────┐
                 │       Flask (app.py)        │
                 │ /api/collections             │
                 │ /api/upload                   │
                 │ /api/ask                       │
                 └─────────────┬─────────────┘
                               │
        ┌──────────────────────┼───────────────────────┐
        ▼                      ▼                       ▼
┌────────────────┐   ┌───────────────────┐   ┌────────────────────┐
│ rag/ingest.py  │   │ rag/chunking.py    │   │  rag/qa.py           │
│ extract_text() │──▶│ chunk_text()        │   │ ingest_file()        │
│ pdf/txt/md/    │   │ word-window chunks  │   │ answer_question()    │
│ docx → text    │   └───────────────────┘   └──────────┬───────────┘
└────────────────┘                                      │
                                                         ▼
                            ┌────────────────────────────────────────┐
                            │           rag/providers.py                │
                            │  LLMProvider / EmbeddingProvider (ABCs)  │
                            │  Anthropic · OpenAI-compatible (OpenAI /  │
                            │  Groq / Google AI Studio / Ollama) ·      │
                            │  HuggingFace (local, default embeddings)  │
                            └──────────────────┬─────────────────────┘
                                               │ embeddings / generation
                                               ▼
                            ┌────────────────────────────────────────┐
                            │             rag/store.py                  │
                            │  VectorStore — persistent ChromaDB,       │
                            │  named collections, source/chunk           │
                            │  metadata, dimension-mismatch guard         │
                            └────────────────────────────────────────┘
```

Config (`config.py`) is the single place every `.env` value is read; every
other module receives already-resolved settings as arguments, never calls
`os.getenv()` itself.

## Setup

```bash
git clone https://github.com/DinilKumarasiri/rag_gen_hrc.git
cd rag_gen_hrc
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set LLM_API_KEY to a real Anthropic API key
flask run
```

Open `http://127.0.0.1:5000`. The default embedding provider
(`sentence-transformers`, local) needs no API key at all — the only key you
need to provide is `LLM_API_KEY` for whichever LLM provider you pick.

## Usage (under 5 minutes)

1. Create a collection (e.g. `acme-hr`) in the UI.
2. Upload the four files in `sample_docs/` — one of each supported format,
   themed as a fictional company's internal docs (PTO policy, security
   guidelines, onboarding guide, expense policy).
3. Ask something like *"What is the PTO carryover cap?"* or *"What MFA
   methods are acceptable?"* — you'll get a grounded answer with a citation
   pointing at the exact source file and chunk.
4. Ask something unrelated, e.g. *"What is the capital of France?"* — you'll
   get the "I couldn't find that in the provided documents" fallback.

## Switching providers

Everything below is a `.env` change only — see the commented-out blocks in
`.env.example`.

**LLM** (`LLM_PROVIDER`, `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`):

| Provider | `LLM_PROVIDER` | `LLM_BASE_URL` | Example `LLM_MODEL` |
|---|---|---|---|
| Anthropic (default) | `anthropic` | *(unused)* | `claude-sonnet-4-5` |
| Groq (free tier) | `openai` | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| Google AI Studio (free tier) | `openai` | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-2.0-flash` |
| Ollama (local) | `openai` | `http://localhost:11434/v1` | `llama3.1` |
| OpenAI | `openai` | `https://api.openai.com/v1` | `gpt-4o-mini` |

`OpenAICompatibleLLM` is one class that covers OpenAI, Groq, Google AI
Studio, and Ollama — anything that speaks the OpenAI chat completions API.
Only `LLM_BASE_URL`/`LLM_MODEL`/`LLM_API_KEY` change between them.

**Embeddings** (`EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, plus
`EMBEDDING_API_KEY`/`EMBEDDING_BASE_URL` when needed):

| Provider | `EMBEDDING_PROVIDER` | Needs a key? |
|---|---|---|
| Local Hugging Face model (default) | `huggingface` | No |
| OpenAI-compatible (OpenAI / Google AI Studio / Ollama) | `openai` | Yes (`EMBEDDING_API_KEY`) |

Anthropic has no first-party embeddings API, so `LLM_PROVIDER=anthropic`
must be paired with a separate `EMBEDDING_PROVIDER` (the `huggingface`
default works out of the box).

## Design decisions & tradeoffs

- **Thin provider ABCs.** `LLMProvider.generate()` and
  `EmbeddingProvider.embed()` are the entire interface. Concrete classes
  (`AnthropicLLM`, `OpenAICompatibleLLM`, `HuggingFaceEmbedding`,
  `OpenAICompatibleEmbedding`) and two factory functions
  (`get_llm_provider`, `get_embedding_provider`) are the only places that
  know about a specific vendor. Everything downstream only ever talks to
  the ABC, so swapping vendors is a `.env` change.
- **Explicit embeddings, not Chroma's built-in embedding functions.** We
  call our own `EmbeddingProvider.embed()` and pass vectors straight into
  Chroma's `add()`/`query()`. This is what actually makes the embedding
  provider swappable — if Chroma computed embeddings itself, the provider
  abstraction wouldn't reach that far.
- **No LLM call on empty/irrelevant retrieval.** `answer_question()`
  checks retrieved chunks against a distance threshold *before* touching
  the LLM at all, and returns the fixed "couldn't find" string directly if
  nothing qualifies. This means the "never hallucinate on empty retrieval"
  requirement holds structurally, not just by prompting — no LLM call
  means no chance of the model improvising an answer.
  - The LLM factory is passed to `answer_question()` as a zero-argument
    callable (`get_llm`), not an already-constructed provider. Python
    evaluates call arguments eagerly, so an already-constructed provider
    would be built (and could fail on a missing API key) on *every*
    request, even ones that never needed it — silently breaking the
    guarantee above whenever no LLM key was configured. This was caught by
    manually testing the no-match path with no API key set; a regression
    test (`test_answer_question_*_never_calls_llm`) now asserts the
    factory itself is never invoked on that path.
  - `DISTANCE_THRESHOLD` (`rag/qa.py`) was tuned empirically against the
    default embedding model, `all-MiniLM-L6-v2`: on the sample docs,
    genuinely relevant question/chunk pairs scored 0.43-0.62 cosine
    distance, unrelated ones scored 0.94-1.0. 0.75 sits in the gap. This
    is model-dependent (see Known Limitations).
- **Citations from metadata, not from the model's text.** Each citation
  (`source`, `chunk_index`, `snippet`) is built directly from the chunk
  that was actually retrieved, before the LLM is even called. The model
  can't hallucinate a citation because it never generates one.
- **Word-count chunking, not token-based.** `chunk_text()` splits on
  whitespace into overlapping windows of *words*. Simpler and dependency-free
  compared to running a tokenizer, and "roughly N words per chunk" is good
  enough for retrieval quality — exact token alignment matters for LLM
  context windows, not for embedding similarity.
- **Re-upload replaces, not duplicates.** `VectorStore.add_chunks()`
  deletes any existing chunks for that exact filename (via Chroma's
  metadata filter) before adding the new ones, and the physical file in
  `uploads/<collection>/` is overwritten the same way. Re-indexing a
  document you already uploaded doesn't leave stale duplicate chunks
  behind.
- **Lazy provider construction.** `app.py` builds each provider on first
  use and caches it, rather than at import time — a missing/invalid
  `LLM_API_KEY` doesn't crash the whole server on `flask run`, only the
  specific request that actually needed the LLM.
- **Flat `rag/` package, no service/repository layers.** `ingest.py`,
  `chunking.py`, `store.py`, `qa.py`, `providers.py` — five files, one
  responsibility each, no interfaces beyond the two provider ABCs. Chosen
  over a deeper layered architecture because there's nothing here that
  needs it yet, and every layer is something I'd have to justify.

## Known limitations

- **Changing `EMBEDDING_MODEL` breaks existing collections.** Chroma's
  HNSW index is fixed to the dimensionality of the first vectors it saw.
  If you change `EMBEDDING_MODEL` after a collection has been indexed, the
  new embeddings won't match, and `add_chunks`/`query` raise
  `DimensionMismatchError` with a message telling you to re-index — but
  there's no automatic migration. You have to delete that collection's
  data under `CHROMA_PERSIST_DIR` and re-upload all its documents with the
  new model.
- **`DISTANCE_THRESHOLD` is a fixed constant tuned for the default
  embedding model.** Switching `EMBEDDING_MODEL` may shift what counts as
  "relevant" vs. "irrelevant" similarity, and could need re-tuning (see
  `rag/qa.py`).
- **No OCR.** Scanned/image-only PDFs extract no text and are rejected as
  unparseable — there's no image-to-text step.
- **No auth.** Anyone who can reach the Flask process can create
  collections, upload documents, or query any collection. Fine for a local
  take-home/demo; not something to expose publicly as-is.
- **Dev server only.** `flask run` is the Werkzeug development server —
  fine for review, but a real deployment would want a proper WSGI server
  (gunicorn/uwsgi) and likely async/background ingestion for large uploads,
  since indexing currently happens synchronously within the upload
  request.
- **Anthropic has no embeddings API.** `LLM_PROVIDER=anthropic` always
  needs a different `EMBEDDING_PROVIDER` alongside it (the `huggingface`
  default handles this automatically).

## Testing

```bash
pytest
```

20 tests, all fast (no network calls, no real LLM/embedding API):

- `tests/test_chunking.py` — word-window chunking: empty input, exact
  fits, overlap correctness, the last (short) chunk not being dropped, and
  invalid `chunk_size`/`overlap` combinations.
- `tests/test_retrieval.py` — `VectorStore` against a temp Chroma dir:
  empty-collection queries, `top_k`, re-upload replacing rather than
  duplicating chunks (and not touching other files' chunks), and
  dimension-mismatch detection on both `add` and `query`.
- `tests/test_qa.py` — `qa.py` orchestration with fake embedding/LLM
  providers: the no-match path never invokes the LLM factory (proven with
  a factory that raises if called), a relevant hit does call the LLM and
  returns citations that match the indexed chunk's metadata, and
  `ingest_file()` chunks/embeds/stores correctly end to end.

## Repo structure

```
app.py                  Flask routes
config.py               Reads every .env value once
rag/
  providers.py          LLM/embedding provider ABCs, implementations, factories
  ingest.py              Text extraction (pdf/txt/md/docx)
  chunking.py             Word-window chunking
  store.py                 ChromaDB wrapper (collections, add, query)
  qa.py                     ingest_file() and answer_question() orchestration
templates/index.html    Single-page UI
static/{style.css,app.js} Plain CSS/JS, no build step
sample_docs/             One file per supported format, for a quick demo
tests/                   pytest suite (see Testing above)
.env.example             All config knobs, with commented alternative provider blocks
```
