const $ = (id) => document.getElementById(id);
const qEl = $("q");
const goEl = $("go");
const statusEl = $("status");
const resultEl = $("result");
const chipsEl = $("chips");

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function bandClass(band) {
  return { high: "b-high", medium: "b-medium", low: "b-low" }[band] || "b-unknown";
}

const pct = (x) => `${Math.round((x || 0) * 100)}%`;

// Report bodies come from an LLM and usually carry markdown. marked +
// DOMPurify (loaded from CDN in index.html) render it safely; if either
// library failed to load we fall back to escaped plain text.
if (window.marked) marked.use({ gfm: true, breaks: true });
function rich(text) {
  if (window.marked && window.DOMPurify) {
    return DOMPurify.sanitize(marked.parse(String(text ?? "")));
  }
  return `<div class="plain">${esc(text)}</div>`;
}

function renderSources(sources) {
  if (!sources || !sources.length) return "";
  const items = sources.map((s) => {
    const title = esc(s.title || s.source_id);
    const link = s.url ? `<a href="${esc(s.url)}" target="_blank" rel="noopener">${title}</a>` : title;
    const rel = s.reliability != null ? ` · tin cậy ${pct(s.reliability)}` : "";
    const mem = s.type === "memory" ? ` <span class="badge b-medium">bộ nhớ</span>` : "";
    return `<li>${link} <span class="meta">[${esc(s.source_id)}] ${esc(s.type)}${rel}</span>${mem}</li>`;
  }).join("");
  return `<div class="src"><b>Nguồn (${sources.length})</b><ul>${items}</ul></div>`;
}

function render(data) {
  const r = data.report;
  const parts = [];

  parts.push(`<div class="card result-card">
    <div class="metrics">
      <div class="metric"><b>${pct(r.overall_confidence)}</b><span>Độ tin cậy</span></div>
      <div class="metric"><b>$${(data.cost_usd || 0).toFixed(4)}</b><span>Chi phí</span></div>
      <div class="metric"><b>${data.llm_calls}</b><span>LLM calls</span></div>
      <div class="metric"><b>${data.tool_calls}</b><span>Tool calls</span></div>
      <div class="metric"><b>${(data.elapsed_sec || 0).toFixed(1)}s</b><span>Thời gian</span></div>
    </div>
  </div>`);

  parts.push(`<div class="card result-card">
    <h2>Khuyến nghị</h2>
    <div class="rec md">${rich(r.recommendation)}</div>
  </div>`);

  if (r.sections && r.sections.length) {
    const secs = r.sections.map((s) => `
      <div class="section-head">
        <h3>${esc(s.heading)}</h3>
        <span class="badge ${bandClass(s.confidence_band)}">${esc(s.confidence_band)}</span>
      </div>
      <div class="body-text md">${rich(s.body)}</div>`).join("");
    parts.push(`<div class="card result-card"><h2>Phân tích</h2>${secs}</div>`);
  }

  if (r.uncertainties && r.uncertainties.length) {
    const items = r.uncertainties.map((u) => `<li>${esc(u)}</li>`).join("");
    parts.push(`<div class="card result-card warn-box"><h2>Điểm chưa chắc chắn</h2><ul>${items}</ul></div>`);
  }

  if (r.contradictions && r.contradictions.length) {
    const items = r.contradictions.map((c) => `<li>${esc(c)}</li>`).join("");
    parts.push(`<div class="card result-card contra-box"><h2>Mâu thuẫn giữa các nguồn</h2><ul>${items}</ul></div>`);
  }

  if (r.all_sources && r.all_sources.length) {
    parts.push(`<div class="card result-card">${renderSources(r.all_sources)}</div>`);
  }

  resultEl.innerHTML = parts.join("");
  resultEl.classList.remove("hidden");
}

function setProgress(msg) {
  statusEl.className = "status";
  statusEl.classList.remove("hidden");
  statusEl.innerHTML = `<span class="spinner"></span> ${esc(msg)}`;
}

// --- SSE streaming ---------------------------------------------------------
// /research/stream pushes one `progress` frame per pipeline stage, then a
// single terminal `result` (same payload as /research) or `error` frame.

async function runStream(question) {
  const res = await fetch("/research/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok || !res.body) {
    throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let data = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    let sep;
    while ((sep = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, sep);
      buf = buf.slice(sep + 2);

      let event = "message";
      let payload = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) payload += line.slice(5).trim();
      }
      if (!payload) continue;

      const parsed = JSON.parse(payload);
      if (event === "progress") {
        setProgress(parsed.msg || parsed.step);
      } else if (event === "result") {
        data = parsed;
      } else if (event === "error") {
        throw new Error(parsed.message || "Lỗi không xác định từ server.");
      }
    }
  }

  if (!data) throw new Error("Kết thúc luồng mà không nhận được kết quả.");
  return data;
}

async function runClassic(question) {
  const res = await fetch("/research", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return res.json();
}

async function run() {
  const question = qEl.value.trim();
  if (question.length < 3) {
    statusEl.className = "status error";
    statusEl.textContent = "Câu hỏi cần ít nhất 3 ký tự.";
    statusEl.classList.remove("hidden");
    return;
  }

  goEl.disabled = true;
  resultEl.classList.add("hidden");
  resultEl.innerHTML = "";
  setProgress("Đang lập kế hoạch và thu thập dữ liệu, có thể mất một lúc…");

  try {
    let data;
    try {
      data = await runStream(question);
    } catch (streamErr) {
      // Streaming unavailable (older server) or broke before a result —
      // one classic retry still gets the answer.
      setProgress("Đang thu thập dữ liệu, có thể mất một lúc…");
      data = await runClassic(question);
    }
    statusEl.classList.add("hidden");
    render(data);
  } catch (err) {
    statusEl.className = "status error";
    statusEl.textContent = `Lỗi: ${err.message}`;
  } finally {
    goEl.disabled = false;
  }
}

goEl.addEventListener("click", run);
qEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run();
});
chipsEl.addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  qEl.value = chip.textContent.trim();
  qEl.focus();
});
