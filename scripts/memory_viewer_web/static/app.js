/* Memory Viewer Web — app.js */

// ---- Element refs ----
const metaEl         = document.getElementById("meta");
const refreshBtn     = document.getElementById("refreshBtn");
const statsRow       = document.getElementById("statsRow");
const categoryChart  = document.getElementById("categoryChart");
const sourceChart    = document.getElementById("sourceChart");
const userSelect     = document.getElementById("userSelect");
const categorySelect = document.getElementById("categorySelect");
const searchInput    = document.getElementById("searchInput");
const searchBtn      = document.getElementById("searchBtn");
const filterCount    = document.getElementById("filterCount");
const memoryList     = document.getElementById("memoryList");
const paginationBar  = document.getElementById("paginationBar");

// ---- State ----
let currentPage = 1;
const perPage = 50;

// ---- Label maps ----
const CATEGORY_LABELS = {
  preference: "偏好", fact: "事实", decision: "决策",
  entity: "实体", other: "其他",
};
const SOURCE_LABELS = {
  auto: "自动提取", manual: "主动保存", compaction: "压缩提取",
};

const CAT_COLORS = {
  preference: "var(--accent)",
  fact:       "var(--good)",
  decision:   "var(--warn)",
  entity:     "var(--purple)",
  other:      "var(--muted)",
};

const SRC_COLORS = {
  auto:       "var(--accent)",
  manual:     "var(--good)",
  compaction: "var(--warn)",
};

// ---- Helpers ----
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function importanceColor(v) {
  if (v >= 0.7) return "var(--good)";
  if (v >= 0.4) return "var(--warn)";
  return "var(--muted2)";
}

// ---- Fetch helpers ----
async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

// ---- Overview ----
async function fetchOverview() {
  const data = await fetchJSON("/api/overview");
  metaEl.textContent = data.db_path;
  renderStats(data);
  renderBarChart(categoryChart, data.categories, CATEGORY_LABELS, CAT_COLORS);
  renderBarChart(sourceChart, data.sources, SOURCE_LABELS, SRC_COLORS);
  return data;
}

function renderStats(data) {
  const items = [
    { value: data.total_memories, label: "Total Memories" },
    { value: data.total_users, label: "Users" },
    { value: Object.keys(data.categories).length, label: "Categories" },
    { value: Object.keys(data.sources).length, label: "Sources" },
  ];
  statsRow.innerHTML = "";
  items.forEach((item, i) => {
    const card = el("div", "stat-card");
    card.style.animationDelay = `${i * 60}ms`;
    const v = el("div", "stat-value", String(item.value));
    const l = el("div", "stat-label", item.label);
    card.appendChild(v);
    card.appendChild(l);
    statsRow.appendChild(card);
  });
}

// ---- D3 Bar Charts ----
function renderBarChart(container, dataObj, labels, colors) {
  container.innerHTML = "";
  const entries = Object.entries(dataObj).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) {
    container.textContent = "No data";
    return;
  }

  const maxVal = d3.max(entries, d => d[1]) || 1;
  const barH = 22;
  const gap = 6;
  const labelW = 72;
  const barMaxW = 160;
  const valueW = 40;
  const totalW = labelW + barMaxW + valueW + 12;
  const totalH = entries.length * (barH + gap) + gap;

  const barScale = d3.scaleLinear().domain([0, maxVal]).range([0, barMaxW]);

  const svg = d3.select(container).append("svg")
    .attr("width", totalW)
    .attr("height", totalH);

  const g = svg.selectAll("g")
    .data(entries)
    .enter().append("g")
    .attr("transform", (d, i) => `translate(0, ${i * (barH + gap) + gap})`);

  // Label
  g.append("text")
    .attr("x", labelW - 6)
    .attr("y", barH / 2)
    .attr("dominant-baseline", "central")
    .attr("text-anchor", "end")
    .attr("class", "chart-bar-label")
    .text(d => labels[d[0]] || d[0]);

  // Bar background
  g.append("rect")
    .attr("x", labelW)
    .attr("y", 2)
    .attr("width", barMaxW)
    .attr("height", barH - 4)
    .attr("rx", 4)
    .attr("fill", "rgba(255,255,255,0.03)");

  // Bar fill
  g.append("rect")
    .attr("x", labelW)
    .attr("y", 2)
    .attr("width", 0)
    .attr("height", barH - 4)
    .attr("rx", 4)
    .attr("fill", d => colors[d[0]] || "var(--accent)")
    .attr("opacity", 0.55)
    .transition()
    .duration(500)
    .delay((d, i) => i * 80)
    .attr("width", d => barScale(d[1]));

  // Value
  g.append("text")
    .attr("x", labelW + barMaxW + 8)
    .attr("y", barH / 2)
    .attr("dominant-baseline", "central")
    .attr("class", "chart-bar-value")
    .text(d => d[1]);
}

// ---- Users ----
async function fetchUsers() {
  const data = await fetchJSON("/api/users");
  // Preserve current selection
  const current = userSelect.value;
  // Remove all options except the first "All Users"
  while (userSelect.options.length > 1) {
    userSelect.remove(1);
  }
  data.users.forEach(u => {
    const opt = document.createElement("option");
    opt.value = u.user_id;
    opt.textContent = `${u.short_id} (${u.count})`;
    userSelect.appendChild(opt);
  });
  // Restore selection if still valid
  if (current) userSelect.value = current;
}

// ---- Memories ----
async function fetchMemories(page) {
  currentPage = page;
  const params = new URLSearchParams();
  if (userSelect.value) params.set("user_id", userSelect.value);
  if (categorySelect.value) params.set("category", categorySelect.value);
  const q = searchInput.value.trim();
  if (q) params.set("search", q);
  params.set("page", page);
  params.set("per_page", perPage);

  const data = await fetchJSON(`/api/memories?${params}`);
  renderMemories(data.memories);
  renderPagination(data.pagination);
  filterCount.textContent = `${data.pagination.total} results`;
}

function renderMemories(memories) {
  memoryList.innerHTML = "";
  if (memories.length === 0) {
    const empty = el("div", "empty-state");
    const icon = el("div", "empty-icon", "\u2205");
    const text = el("div", "", "No memories found");
    empty.appendChild(icon);
    empty.appendChild(text);
    memoryList.appendChild(empty);
    return;
  }

  memories.forEach((m, i) => {
    const card = el("div", "memory-card");
    card.style.opacity = "0";
    card.style.transform = "translateY(8px)";
    card.style.transition = `opacity 0.25s ease ${i * 30}ms, transform 0.25s ease ${i * 30}ms`;

    // Content
    const content = el("div", "memory-content", m.content);
    card.appendChild(content);

    // Expand/collapse button (inserted after content, before meta)
    const expandBtn = el("button", "expand-btn");
    expandBtn.style.display = "none";
    card.appendChild(expandBtn);

    // Meta row
    const meta = el("div", "memory-meta");

    // Category badge
    const catBadge = el("span", `cat-badge ${m.category}`, m.category_label);
    meta.appendChild(catBadge);

    // Importance bar
    const impWrap = el("span", "importance-wrap");
    const track = el("span", "importance-track");
    const fill = el("span", "importance-fill");
    fill.style.width = `${(m.importance * 100).toFixed(0)}%`;
    fill.style.background = importanceColor(m.importance);
    track.appendChild(fill);
    impWrap.appendChild(track);
    const impVal = el("span", "importance-val", m.importance.toFixed(1));
    impWrap.appendChild(impVal);
    meta.appendChild(impWrap);

    // Source tag
    const srcTag = el("span", "source-tag", m.source_label);
    meta.appendChild(srcTag);

    // Separator
    meta.appendChild(el("span", "sep", "|"));

    // Timestamps
    const ts = el("span", "ts-text", `${m.updated_at_str}`);
    meta.appendChild(ts);

    // Access count
    if (m.access_count > 0) {
      meta.appendChild(el("span", "sep", "|"));
      meta.appendChild(el("span", "ts-text", `${m.access_count}x accessed`));
    }

    // User ID chip (useful when "All Users" is selected)
    if (!userSelect.value) {
      const uid = m.user_id;
      const short = uid.length > 8 ? uid.slice(-8) : uid;
      const chip = el("span", "user-id-chip", short);
      chip.title = uid;
      meta.appendChild(chip);
    }

    // Delete button
    const delBtn = el("button", "delete-btn", "Delete");
    delBtn.addEventListener("click", () => deleteMemory(m.id));
    meta.appendChild(delBtn);

    card.appendChild(meta);
    memoryList.appendChild(card);

    // Trigger entrance animation + detect overflow
    requestAnimationFrame(() => {
      card.style.opacity = "1";
      card.style.transform = "translateY(0)";

      // Check if content overflows
      if (content.scrollHeight > content.clientHeight + 2) {
        content.classList.add("clamped");
        expandBtn.textContent = "Show more \u25BE";
        expandBtn.style.display = "";
        expandBtn.addEventListener("click", () => {
          const expanded = content.classList.toggle("expanded");
          content.classList.toggle("clamped", !expanded);
          expandBtn.textContent = expanded ? "Show less \u25B4" : "Show more \u25BE";
        });
      }
    });
  });
}

function renderPagination(pg) {
  paginationBar.innerHTML = "";
  if (pg.total_pages <= 1) return;

  // Prev
  const prev = el("button", "", "\u2190");
  prev.disabled = pg.page <= 1;
  prev.addEventListener("click", () => fetchMemories(pg.page - 1));
  paginationBar.appendChild(prev);

  // Page numbers — show max 7 pages
  const maxShow = 7;
  let start = Math.max(1, pg.page - Math.floor(maxShow / 2));
  let end = Math.min(pg.total_pages, start + maxShow - 1);
  if (end - start < maxShow - 1) start = Math.max(1, end - maxShow + 1);

  for (let i = start; i <= end; i++) {
    const btn = el("button", i === pg.page ? "active" : "", String(i));
    btn.addEventListener("click", () => fetchMemories(i));
    paginationBar.appendChild(btn);
  }

  // Next
  const next = el("button", "", "\u2192");
  next.disabled = pg.page >= pg.total_pages;
  next.addEventListener("click", () => fetchMemories(pg.page + 1));
  paginationBar.appendChild(next);
}

// ---- Delete ----
async function deleteMemory(id) {
  if (!confirm(`Delete memory ${id}?`)) return;
  const res = await fetch(`/api/memories/${id}`, { method: "DELETE" });
  const data = await res.json();
  if (data.ok) {
    fetchMemories(currentPage);
    fetchOverview();
  } else {
    alert(`Delete failed: ${data.error || "unknown error"}`);
  }
}

// ---- Event listeners ----
refreshBtn.addEventListener("click", () => bootstrap());

userSelect.addEventListener("change", () => {
  currentPage = 1;
  fetchMemories(1);
});

categorySelect.addEventListener("change", () => {
  currentPage = 1;
  fetchMemories(1);
});

searchBtn.addEventListener("click", () => {
  currentPage = 1;
  fetchMemories(1);
});

searchInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    currentPage = 1;
    fetchMemories(1);
  }
});

// ---- Bootstrap ----
async function bootstrap() {
  await Promise.all([
    fetchOverview(),
    fetchUsers(),
    fetchMemories(1),
  ]);
}

bootstrap().catch(err => {
  memoryList.innerHTML = `<div class="empty-state">Error: ${err.message}</div>`;
});
