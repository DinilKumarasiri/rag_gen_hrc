const collectionSelect = document.getElementById("collection-select");
const newCollectionInput = document.getElementById("new-collection-name");
const createCollectionBtn = document.getElementById("create-collection-btn");
const fileInput = document.getElementById("file-input");
const uploadBtn = document.getElementById("upload-btn");
const uploadLog = document.getElementById("upload-log");
const questionInput = document.getElementById("question-input");
const askBtn = document.getElementById("ask-btn");
const answerBox = document.getElementById("answer-box");
const answerText = document.getElementById("answer-text");
const citationsDiv = document.getElementById("citations");
const statusMessage = document.getElementById("status-message");

let statusTimer = null;

function showStatus(message, isError) {
  statusMessage.textContent = message;
  statusMessage.className = "status " + (isError ? "status-error" : "status-success");
  statusMessage.classList.remove("hidden");
  clearTimeout(statusTimer);
  statusTimer = setTimeout(() => statusMessage.classList.add("hidden"), 5000);
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  return { ok: res.ok, data };
}

createCollectionBtn.addEventListener("click", async () => {
  const name = newCollectionInput.value.trim();
  if (!name) {
    showStatus("Enter a collection name.", true);
    return;
  }

  try {
    const { ok, data } = await postJson("/api/collections", { name });
    if (!ok) {
      showStatus(data.error || "Could not create collection.", true);
      return;
    }
    const option = document.createElement("option");
    option.value = data.collection;
    option.textContent = data.collection;
    collectionSelect.appendChild(option);
    collectionSelect.value = data.collection;
    newCollectionInput.value = "";
    showStatus(`Collection "${data.collection}" created.`, false);
  } catch (err) {
    showStatus("Network error creating collection.", true);
  }
});

uploadBtn.addEventListener("click", async () => {
  const collection = collectionSelect.value;
  if (!collection) {
    showStatus("Select or create a collection first.", true);
    return;
  }
  const file = fileInput.files[0];
  if (!file) {
    showStatus("Choose a file to upload.", true);
    return;
  }

  const formData = new FormData();
  formData.append("collection", collection);
  formData.append("file", file);

  uploadBtn.disabled = true;
  uploadBtn.textContent = "Uploading...";
  try {
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      showStatus(data.error || "Upload failed.", true);
      return;
    }
    const item = document.createElement("li");
    item.textContent = `${data.filename} — ${data.chunks_indexed} chunk(s) indexed`;
    uploadLog.prepend(item);
    fileInput.value = "";
    showStatus(`Indexed ${data.filename}.`, false);
  } catch (err) {
    showStatus("Network error during upload.", true);
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.textContent = "Upload";
  }
});

function renderCitations(citations) {
  citationsDiv.replaceChildren();
  (citations || []).forEach((citation) => {
    const card = document.createElement("div");
    card.className = "citation-card";

    const sourceLine = document.createElement("div");
    sourceLine.className = "source-line";
    sourceLine.textContent = `${citation.source} (chunk ${citation.chunk_index})`;

    const snippet = document.createElement("p");
    snippet.textContent = citation.snippet;

    card.appendChild(sourceLine);
    card.appendChild(snippet);
    citationsDiv.appendChild(card);
  });
}

askBtn.addEventListener("click", async () => {
  const collection = collectionSelect.value;
  const question = questionInput.value.trim();
  if (!collection) {
    showStatus("Select a collection first.", true);
    return;
  }
  if (!question) {
    showStatus("Enter a question.", true);
    return;
  }

  askBtn.disabled = true;
  askBtn.textContent = "Asking...";
  answerBox.classList.add("hidden");
  try {
    const { ok, data } = await postJson("/api/ask", { collection, question });
    if (!ok) {
      showStatus(data.error || "Could not get an answer.", true);
      return;
    }
    answerText.textContent = data.answer;
    renderCitations(data.citations);
    answerBox.classList.remove("hidden");
  } catch (err) {
    showStatus("Network error while asking.", true);
  } finally {
    askBtn.disabled = false;
    askBtn.textContent = "Ask";
  }
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    askBtn.click();
  }
});
