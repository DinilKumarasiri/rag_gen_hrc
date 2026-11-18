"""Flask app: collection management, document upload/indexing, and Q&A —
thin routes over rag/qa.py, rag/store.py, and rag/providers.py."""
import re
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from config import Config
from rag.ingest import EmptyDocumentError, UnsupportedFileTypeError
from rag.providers import ProviderError, get_embedding_provider, get_llm_provider
from rag.qa import answer_question, ingest_file
from rag.store import DimensionMismatchError, VectorStore

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB per upload

store = VectorStore(persist_dir=Config.CHROMA_PERSIST_DIR)

# Providers are constructed lazily (on first actual use) rather than at
# import time, so a missing/invalid LLM_API_KEY doesn't crash the whole
# server on startup — only requests that need the LLM fail, with a clear
# error message.
_embedder = None
_llm = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = get_embedding_provider(Config)
    return _embedder


def get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm_provider(Config)
    return _llm


# Mirrors Chroma's own collection naming rules (3-63 chars, alnum/-/_
# only, must start and end with an alphanumeric character).
COLLECTION_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{1,61}[a-zA-Z0-9]$")


@app.route("/")
def index():
    return render_template("index.html", collections=store.list_collections())


@app.route("/api/collections", methods=["GET"])
def list_collections():
    return jsonify({"collections": store.list_collections()})


@app.route("/api/collections", methods=["POST"])
def create_collection():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not COLLECTION_NAME_RE.match(name):
        return (
            jsonify(
                {
                    "error": "Collection names must be 3-63 characters: letters, "
                    "numbers, '-' or '_', and must start/end with a letter or number."
                }
            ),
            400,
        )
    store.create_collection(name)
    return jsonify({"collection": name}), 201


@app.route("/api/upload", methods=["POST"])
def upload():
    collection = (request.form.get("collection") or "").strip()
    if not collection:
        return jsonify({"error": "Select or create a collection first."}), 400
    if collection not in store.list_collections():
        return jsonify({"error": f"Collection '{collection}' does not exist."}), 404

    upload_file = request.files.get("file")
    if upload_file is None or not upload_file.filename:
        return jsonify({"error": "No file selected."}), 400

    filename = secure_filename(upload_file.filename)
    if not filename:
        return jsonify({"error": "Invalid filename."}), 400

    file_bytes = upload_file.read()
    if not file_bytes:
        return jsonify({"error": f"'{filename}' is empty."}), 400

    try:
        chunk_count = ingest_file(
            filename,
            file_bytes,
            collection,
            store,
            get_embedder(),
            Config.CHUNK_SIZE,
            Config.CHUNK_OVERLAP,
        )
    except (UnsupportedFileTypeError, EmptyDocumentError) as exc:
        return jsonify({"error": str(exc)}), 400
    except DimensionMismatchError as exc:
        return jsonify({"error": str(exc)}), 409
    except ProviderError as exc:
        return jsonify({"error": str(exc)}), 502

    # Save the original file only after successful indexing, so a failed
    # upload never leaves an orphaned file on disk. Re-uploading the same
    # filename overwrites it here, matching the "replace, don't duplicate"
    # behavior in the vector store.
    upload_dir = Path(Config.UPLOAD_DIR) / collection
    upload_dir.mkdir(parents=True, exist_ok=True)
    (upload_dir / filename).write_bytes(file_bytes)

    return jsonify({"filename": filename, "chunks_indexed": chunk_count}), 201


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    collection = (data.get("collection") or "").strip()
    question = (data.get("question") or "").strip()

    if not collection:
        return jsonify({"error": "Select a collection first."}), 400
    if not question:
        return jsonify({"error": "Enter a question."}), 400
    if collection not in store.list_collections():
        return jsonify({"error": f"Collection '{collection}' does not exist."}), 404

    try:
        result = answer_question(
            question, collection, store, get_embedder(), get_llm, Config.TOP_K
        )
    except DimensionMismatchError as exc:
        return jsonify({"error": str(exc)}), 409
    except ProviderError as exc:
        return jsonify({"error": str(exc)}), 502

    return jsonify(result)


@app.errorhandler(413)
def handle_too_large(_error):
    return jsonify({"error": "File is too large (max 20 MB)."}), 413


@app.errorhandler(500)
def handle_server_error(error):
    app.logger.exception("Unhandled server error", exc_info=error)
    return jsonify({"error": "An unexpected server error occurred."}), 500

def handle_404(_error):
    return jsonify({"error": "Resource not found."}), 404
    
if __name__ == "__main__":
    app.run(debug=True)
