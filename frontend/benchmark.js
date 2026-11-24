const runEvalBtn = document.getElementById("runEvalBtn");
const evalRetrieverMode = document.getElementById("evalRetrieverMode");
const evalLoading = document.getElementById("evalLoading");
const evalResults = document.getElementById("evalResults");

function escapeHtml(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function loadLatest() {
  try {
    const res = await fetch("/api/eval/latest");
    const data = await res.json();
    if (data.has_results) {
      renderReport(data);
    } else {
      evalResults.innerHTML = `
        <div class="empty-card">
          <h3>No benchmark run saved yet</h3>
          <p class="muted">Click <strong>Run Live Benchmark</strong> above to evaluate all 9 questions.</p>
        </div>`;
    }
  } catch (err) {
    evalResults.innerHTML = `<div class="empty-card"><p class="muted">Failed to connect to backend.</p></div>`;
  }
}

runEvalBtn.addEventListener("click", async () => {
  runEvalBtn.disabled = true;
  evalLoading.classList.remove("hidden");
  try {
    const res = await fetch("/api/eval/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ retriever_mode: evalRetrieverMode.value }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Evaluation failed");
    renderReport(data);
  } catch (err) {
    evalResults.innerHTML = `<div class="empty-card"><p style="color:#dc2626">${escapeHtml(err.message)}</p></div>`;
  } finally {
    runEvalBtn.disabled = false;
    evalLoading.classList.add("hidden");
  }
});

function renderReport(report) {
  const cards = report.summary
    .map((row) => {
      const ours = row.method.includes("Adaptive RAG");
      return `
        <div class="metric-card ${ours ? "featured" : ""}">
          <h3>${escapeHtml(row.method)}</h3>
          <div class="metric-row"><span>Accuracy (Keyword Recall)</span><strong>${row.avg_accuracy}%</strong></div>
          <div class="bar-track"><div class="bar-fill" style="width:${row.avg_accuracy}%"></div></div>
          <div class="metric-row" style="margin-top:10px;"><span>Avg Latency</span><strong>${(row.avg_latency_ms / 1000).toFixed(2)}s</strong></div>
          <div class="metric-row"><span>Avg Retrieval Steps</span><strong>${row.avg_retrieval_steps}</strong></div>
          <div class="metric-row">
            <span>Router Accuracy</span>
            <strong>${row.route_accuracy !== null && row.route_accuracy !== undefined ? row.route_accuracy + "%" : "Static"}</strong>
          </div>
        </div>`;
    })
    .join("");

  const rows = report.per_question
    .map((q) => {
      const sA = q.scores["No-RAG (Always A)"] || {};
      const sB = q.scores["Naive RAG (Always B)"] || {};
      const sAd = q.scores["Adaptive RAG (Ours)"] || {};
      return `
        <tr>
          <td><strong>${escapeHtml(q.id)}</strong></td>
          <td>${escapeHtml(q.query)}</td>
          <td><span class="route-tag route-${escapeHtml(q.gold_route)}">${escapeHtml(q.gold_route)}</span></td>
          <td>
            <span class="route-tag route-${escapeHtml(q.predicted_route)}">${escapeHtml(q.predicted_route)}</span>
            <span class="${q.route_correct ? "ok-mark" : "bad-mark"}">${q.route_correct ? "✓" : "✗"}</span>
          </td>
          <td>${sA.accuracy ?? "-"}%</td>
          <td>${sB.accuracy ?? "-"}%</td>
          <td><strong>${sAd.accuracy ?? "-"}%</strong></td>
        </tr>`;
    })
    .join("");

  evalResults.innerHTML = `
    <p class="muted" style="margin-bottom:14px;">
      Last run: ${escapeHtml(report.timestamp)} &middot; Retriever: <strong>${escapeHtml(report.retriever_mode.toUpperCase())}</strong> &middot; Duration: ${report.elapsed_sec}s
    </p>
    <div class="metric-grid">${cards}</div>
    <div class="table-card">
      <table class="clean-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Question</th>
            <th>Expected</th>
            <th>Predicted</th>
            <th>No-RAG</th>
            <th>Naive RAG</th>
            <th>Adaptive RAG</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

loadLatest();
