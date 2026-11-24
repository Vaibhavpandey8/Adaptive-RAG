const docList = document.getElementById("docList");
const fileInput = document.getElementById("fileInput");
const uploadStatus = document.getElementById("uploadStatus");
const routeModeEl = document.getElementById("routeMode");
const retrieverModeEl = document.getElementById("retrieverMode");
const chatMessages = document.getElementById("chatMessages");
const chatForm = document.getElementById("chatForm");
const queryInput = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");
const newChatBtn = document.getElementById("newChatBtn");

let isSending = false;
let conversationHistory = [];
const initialWelcomeHtml = chatMessages.innerHTML;

function escapeHtml(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatAnswer(raw) {
  const safe = escapeHtml(raw)
    .replace(/\[([a-zA-Z0-9._-]+\.(?:md|txt|pdf)[^\]]*)\]/g, '<span class="cite">[$1]</span>')
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

  const lines = safe.split(/\r?\n/);
  let html = "";
  let inList = false;

  for (const line of lines) {
    const t = line.trim();
    if (!t) {
      if (inList) {
        html += "</ul>";
        inList = false;
      }
      continue;
    }
    if (/^[-*]\s+/.test(t)) {
      if (!inList) {
        html += "<ul>";
        inList = true;
      }
      html += `<li>${t.replace(/^[-*]\s+/, "")}</li>`;
    } else {
      if (inList) {
        html += "</ul>";
        inList = false;
      }
      html += `<p>${t}</p>`;
    }
  }
  if (inList) html += "</ul>";
  return html;
}

function bindSuggestions() {
  chatMessages.querySelectorAll(".suggestion").forEach((btn) => {
    btn.addEventListener("click", () => {
      queryInput.value = btn.dataset.q;
      chatForm.requestSubmit();
    });
  });
}

bindSuggestions();

newChatBtn.addEventListener("click", () => {
  conversationHistory = [];
  chatMessages.innerHTML = initialWelcomeHtml;
  bindSuggestions();
  queryInput.focus();
});

async function loadState() {
  try {
    const dRes = await fetch("/api/documents");
    const docsData = await dRes.json();
    renderDocs(docsData.documents);
  } catch {
    docList.innerHTML = `<li class="doc-empty">Server offline.</li>`;
  }
}

function renderDocs(docs) {
  if (!docs || docs.length === 0) {
    docList.innerHTML = `<li class="doc-empty">No documents uploaded yet.</li>`;
    return;
  }
  docList.innerHTML = docs
    .map(
      (d) => `
      <li class="doc-item" title="${escapeHtml(d.filename)} (${d.chunk_count} chunks)">
        <span class="doc-name">${escapeHtml(d.filename)}</span>
        <button class="doc-del" data-file="${escapeHtml(d.filename)}" title="Remove">&times;</button>
      </li>`
    )
    .join("");

  docList.querySelectorAll(".doc-del").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const name = btn.dataset.file;
      uploadStatus.textContent = `Removing ${name}...`;
      const res = await fetch(`/api/documents/${encodeURIComponent(name)}`, { method: "DELETE" });
      if (res.ok) {
        uploadStatus.textContent = `Removed ${name}`;
        loadState();
      }
    });
  });
}

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;

  uploadStatus.textContent = `Uploading ${file.name}...`;
  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch("/api/upload", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed");
    uploadStatus.textContent = `Indexed ${file.name}`;
    loadState();
  } catch (err) {
    uploadStatus.textContent = err.message;
  } finally {
    fileInput.value = "";
  }
});

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (isSending) return;

  const query = queryInput.value.trim();
  if (!query) return;

  const welcome = document.getElementById("welcomeState");
  if (welcome) welcome.remove();

  appendUserMessage(query);
  queryInput.value = "";

  const loader = appendTyping();
  isSending = true;
  sendBtn.disabled = true;

  try {
    const res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        history: conversationHistory.slice(-6),
        force_route: routeModeEl.value,
        retriever_mode: retrieverModeEl ? retrieverModeEl.value : "hybrid",
      }),
    });
    const data = await res.json();
    loader.remove();

    if (!res.ok) {
      appendError(data.detail || "Something went wrong.");
      return;
    }

    conversationHistory.push({ role: "user", content: query });
    conversationHistory.push({ role: "assistant", content: data.answer });
    appendAssistantMessage(data);
  } catch (err) {
    loader.remove();
    appendError(`Connection error: ${err.message}`);
  } finally {
    isSending = false;
    sendBtn.disabled = false;
    queryInput.focus();
  }
});

function appendUserMessage(text) {
  const el = document.createElement("div");
  el.className = "msg user";
  el.textContent = text;
  chatMessages.appendChild(el);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendTyping() {
  const el = document.createElement("div");
  el.className = "msg assistant typing";
  el.textContent = "Thinking...";
  chatMessages.appendChild(el);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return el;
}

function appendError(msg) {
  const el = document.createElement("div");
  el.className = "msg assistant";
  el.style.color = "#dc2626";
  el.textContent = msg;
  chatMessages.appendChild(el);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendAssistantMessage(data) {
  const el = document.createElement("div");
  el.className = "msg assistant";

  const routeNames = {
    A: "Direct LLM",
    B: "Single-Hop RAG",
    C: "Multi-Hop RAG",
  };

  const routeLabel = routeNames[data.route] || data.route;
  const totalSec = (data.metrics.total_ms / 1000).toFixed(2);
  const confPct = Math.round((data.routing.confidence || 0) * 100);

  const rewrittenHtml = data.was_rewritten
    ? `<div class="d-row"><strong>Follow-up rewritten to:</strong> ${escapeHtml(data.standalone_query)}</div>`
    : "";

  const retryHtml = data.crag_retry
    ? `<div class="d-row"><strong>Corrective Retry:</strong> Initial retrieval was weak, automatically reformulated query &amp; retried.</div>`
    : "";

  const subQHtml =
    data.sub_queries && data.sub_queries.length > 0
      ? `<div class="d-row"><strong>Sub-queries:</strong><ol>${data.sub_queries
          .map((q) => `<li>${escapeHtml(q)}</li>`)
          .join("")}</ol></div>`
      : "";

  const chunksHtml =
    data.retrieved_chunks && data.retrieved_chunks.length > 0
      ? `<div class="d-row"><strong>Reranked sources (${data.retrieved_chunks.length}):</strong>${data.retrieved_chunks
          .slice(0, 4)
          .map(
            (c) => `
            <div class="source-item">
              <div class="source-item-head">${escapeHtml(c.source)} &middot; ${escapeHtml(c.section)}${
                c.rerank_score !== undefined ? ` &middot; Relevance ${c.rerank_score}/10` : ""
              }</div>
              <div class="source-item-text">${escapeHtml(c.content)}</div>
            </div>`
          )
          .join("")}</div>`
      : "";

  const m = data.metrics;
  const timingHtml = `<div class="d-row"><strong>Timing:</strong> Router ${m.router_ms}ms &middot; Retrieval ${m.retrieval_ms}ms &middot; Rerank ${m.rerank_ms}ms &middot; LLM ${m.generation_ms}ms</div>`;

  el.innerHTML = `
    <div class="msg-body">${formatAnswer(data.answer)}</div>
    <div class="msg-meta">
      <span class="route-tag route-${escapeHtml(data.route)}">Route ${escapeHtml(data.route)} &middot; ${routeLabel}</span>
      <span>${data.retrieval_steps} retrieval step${data.retrieval_steps === 1 ? "" : "s"}</span>
      ${data.crag_retry ? "<span>CRAG retry</span>" : ""}
      ${data.guardrail_triggered ? "<span>Abstained</span>" : ""}
      <span>${totalSec}s</span>
      <button type="button" class="details-toggle">Details &#9662;</button>
    </div>
    <div class="details-box">
      ${rewrittenHtml}
      <div class="d-row"><strong>Why this route:</strong> ${escapeHtml(data.routing.reason)} (${confPct}% confidence)</div>
      ${retryHtml}
      ${subQHtml}
      ${timingHtml}
      ${chunksHtml}
    </div>
  `;

  const toggleBtn = el.querySelector(".details-toggle");
  const detailsBox = el.querySelector(".details-box");
  toggleBtn.addEventListener("click", () => {
    const isOpen = detailsBox.classList.toggle("open");
    toggleBtn.innerHTML = isOpen ? "Hide details &#9652;" : "Details &#9662;";
  });

  chatMessages.appendChild(el);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

loadState();
