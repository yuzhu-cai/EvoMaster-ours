const titleEl = document.getElementById("title");
const metaEl = document.getElementById("meta");
const statsRow = document.getElementById("statsRow");
const chartRow = document.getElementById("chartRow");
const dynamicFilters = document.getElementById("dynamicFilters");
const searchInput = document.getElementById("searchInput");
const searchBtn = document.getElementById("searchBtn");
const refreshBtn = document.getElementById("refreshBtn");
const filterCount = document.getElementById("filterCount");
const recordList = document.getElementById("recordList");
const paginationBar = document.getElementById("paginationBar");
const sourceSelect = document.getElementById("sourceSelect");
const viewSelect = document.getElementById("viewSelect");
const onlyFailedWrap = document.getElementById("onlyFailedWrap");
const onlyFailedInput = document.getElementById("onlyFailedInput");

let currentPage = 1;
let currentConfig = null;
let currentFacets = null;
let currentSource = "memory";
let currentView = "records";

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  return response.json();
}

function makeUrl(path, extraParams = {}) {
  const params = new URLSearchParams({ source: currentSource, ...extraParams });
  return `${path}?${params}`;
}

function resetLoadingState() {
  statsRow.innerHTML = Array.from({ length: 4 }, () => '<div class="stat-card shimmer"></div>').join("");
  chartRow.innerHTML = Array.from({ length: 2 }, () => '<div class="chart-card"><div class="shimmer" style="height:150px"></div></div>').join("");
  recordList.innerHTML = Array.from({ length: 3 }, () => '<div class="shimmer" style="height:92px"></div>').join("");
}

function renderStats(cards) {
  statsRow.innerHTML = "";
  cards.forEach((item, index) => {
    const card = el("div", "stat-card");
    card.style.animationDelay = `${index * 60}ms`;
    card.appendChild(el("div", "stat-value", String(item.value)));
    card.appendChild(el("div", "stat-label", item.label));
    statsRow.appendChild(card);
  });
}

function renderBarChart(container, chart, index) {
  const entries = Object.entries(chart.data || {}).sort((a, b) => b[1] - a[1]);
  const card = el("div", "chart-card");
  card.appendChild(el("div", "chart-title", chart.title));
  if (!entries.length) {
    card.appendChild(el("div", "empty-state", "No data"));
    container.appendChild(card);
    return;
  }
  const mount = el("div", "chart-mount");
  card.appendChild(mount);
  container.appendChild(card);
  const maxVal = d3.max(entries, d => d[1]) || 1;
  const barH = 22;
  const gap = 6;
  const labelW = 104;
  const barMaxW = 180;
  const valueW = 46;
  const totalW = labelW + barMaxW + valueW + 16;
  const totalH = entries.length * (barH + gap) + gap;
  const palette = ["#66e3ff", "#b8a0ff", "#26e3a1", "#ffd166", "#ff7b9c", "#81c784"];
  const scale = d3.scaleLinear().domain([0, maxVal]).range([0, barMaxW]);
  const svg = d3.select(mount).append("svg").attr("width", totalW).attr("height", totalH);
  const rows = svg.selectAll("g").data(entries).enter().append("g").attr("transform", (d, i) => `translate(0, ${i * (barH + gap) + gap})`);
  rows.append("text").attr("x", labelW - 6).attr("y", barH / 2).attr("dominant-baseline", "central").attr("text-anchor", "end").attr("class", "chart-bar-label").text(d => (chart.labels || {})[d[0]] || d[0]);
  rows.append("rect").attr("x", labelW).attr("y", 2).attr("width", barMaxW).attr("height", barH - 4).attr("rx", 4).attr("fill", "rgba(255,255,255,0.03)");
  rows.append("rect").attr("x", labelW).attr("y", 2).attr("width", 0).attr("height", barH - 4).attr("rx", 4).attr("fill", palette[index % palette.length]).attr("opacity", 0.65).transition().duration(450).delay((d, i) => i * 70).attr("width", d => scale(d[1]));
  rows.append("text").attr("x", labelW + barMaxW + 8).attr("y", barH / 2).attr("dominant-baseline", "central").attr("class", "chart-bar-value").text(d => d[1]);
}

function renderCharts(charts) {
  chartRow.innerHTML = "";
  charts.forEach((chart, index) => renderBarChart(chartRow, chart, index));
}

function renderSourceSelector(config) {
  sourceSelect.innerHTML = "";
  (config.sources || []).forEach(source => {
    const option = document.createElement("option");
    option.value = source.value;
    option.textContent = source.available ? source.label : `${source.label} (missing)`;
    option.disabled = !source.available;
    sourceSelect.appendChild(option);
  });
  sourceSelect.value = currentSource;
}

function renderViewSelector(facets) {
  viewSelect.innerHTML = "";
  const views = facets.views || [];
  if (!views.length) {
    viewSelect.classList.add("hidden");
    currentView = "records";
    return;
  }
  viewSelect.classList.remove("hidden");
  views.forEach(view => {
    const option = document.createElement("option");
    option.value = view.value;
    option.textContent = view.label;
    viewSelect.appendChild(option);
  });
  const validValues = new Set(views.map(view => view.value));
  if (!validValues.has(currentView)) currentView = views[0].value;
  viewSelect.value = currentView;
}

function currentFiltersForView() {
  if (!currentFacets) return [];
  if (currentFacets.filters_by_view) return currentFacets.filters_by_view[currentView] || [];
  return currentFacets.filters || [];
}

function renderFilters() {
  dynamicFilters.innerHTML = "";
  currentFiltersForView().forEach(filter => {
    const select = document.createElement("select");
    select.dataset.key = filter.key;
    const allOption = document.createElement("option");
    allOption.value = "";
    allOption.textContent = filter.all_label;
    select.appendChild(allOption);
    filter.options.forEach(opt => {
      const option = document.createElement("option");
      option.value = opt.value;
      option.textContent = opt.label;
      select.appendChild(option);
    });
    select.addEventListener("change", () => fetchRecords(1).catch(showError));
    dynamicFilters.appendChild(select);
  });
  onlyFailedWrap.classList.toggle("hidden", !(currentSource === "scheduler" && currentView === "runs"));
  searchInput.placeholder = currentFacets?.search_placeholder || "Search...";
}

function collectParams(page) {
  const params = new URLSearchParams({ source: currentSource, page, per_page: 50 });
  const query = searchInput.value.trim();
  if (query) params.set("search", query);
  if (currentView) params.set("view", currentView);
  dynamicFilters.querySelectorAll("select").forEach(select => {
    if (select.value) params.set(select.dataset.key, select.value);
  });
  if (currentSource === "scheduler" && currentView === "runs" && onlyFailedInput.checked) params.set("only_failed", "1");
  return params;
}

function renderMetaList(metaItems) {
  const wrap = el("div", "record-meta");
  metaItems.forEach((item, index) => {
    wrap.appendChild(el("span", "meta-label", `${item.label}:`));
    wrap.appendChild(el("span", "meta-value", item.value || "-"));
    if (index !== metaItems.length - 1) wrap.appendChild(el("span", "sep", "|"));
  });
  return wrap;
}

function renderBadges(badges) {
  const wrap = el("div", "badge-row");
  badges.forEach(badge => wrap.appendChild(el("span", `badge ${badge.kind}`, badge.text)));
  return wrap;
}

function buildDeleteButton(item) {
  if (currentSource !== "memory") return null;
  const button = el("button", "delete-btn", "Delete");
  button.addEventListener("click", () => deleteRecord(item.id));
  return button;
}

function renderRecords(items) {
  recordList.innerHTML = "";
  if (!items.length) {
    recordList.appendChild(el("div", "empty-state", "No records found"));
    return;
  }
  items.forEach((item, index) => {
    const card = el("div", "record-card");
    card.style.opacity = "0";
    card.style.transform = "translateY(8px)";
    card.style.transition = `opacity 0.25s ease ${index * 25}ms, transform 0.25s ease ${index * 25}ms`;
    const header = el("div", "record-header");
    header.appendChild(el("div", "record-primary", item.primary_text || "-"));
    const deleteBtn = buildDeleteButton(item);
    if (deleteBtn) header.appendChild(deleteBtn);
    card.appendChild(header);
    if (item.secondary_text) card.appendChild(el("div", "record-secondary", item.secondary_text));
    if (item.badges?.length) card.appendChild(renderBadges(item.badges));
    if (item.meta?.length) card.appendChild(renderMetaList(item.meta));
    recordList.appendChild(card);
    requestAnimationFrame(() => {
      card.style.opacity = "1";
      card.style.transform = "translateY(0)";
    });
  });
}

function renderPagination(pg) {
  paginationBar.innerHTML = "";
  if (!pg || pg.total_pages <= 1) return;
  const prev = el("button", "", "←");
  prev.disabled = pg.page <= 1;
  prev.addEventListener("click", () => fetchRecords(pg.page - 1).catch(showError));
  paginationBar.appendChild(prev);
  const maxShow = 7;
  let start = Math.max(1, pg.page - Math.floor(maxShow / 2));
  let end = Math.min(pg.total_pages, start + maxShow - 1);
  if (end - start < maxShow - 1) start = Math.max(1, end - maxShow + 1);
  for (let page = start; page <= end; page += 1) {
    const button = el("button", page === pg.page ? "active" : "", String(page));
    button.addEventListener("click", () => fetchRecords(page).catch(showError));
    paginationBar.appendChild(button);
  }
  const next = el("button", "", "→");
  next.disabled = pg.page >= pg.total_pages;
  next.addEventListener("click", () => fetchRecords(pg.page + 1).catch(showError));
  paginationBar.appendChild(next);
}

async function fetchConfig() {
  currentConfig = await fetchJSON(makeUrl("/api/config"));
  titleEl.textContent = currentConfig.title;
  metaEl.textContent = `${currentSource} · ${currentConfig.db_path}`;
  renderSourceSelector(currentConfig);
}

async function fetchOverview() {
  const data = await fetchJSON(makeUrl("/api/overview"));
  metaEl.textContent = `${currentSource} · ${data.db_path}`;
  renderStats(data.cards || []);
  renderCharts(data.charts || []);
}

async function fetchFacets() {
  currentFacets = await fetchJSON(makeUrl("/api/facets"));
  renderViewSelector(currentFacets);
  renderFilters();
}

async function fetchRecords(page = 1) {
  currentPage = page;
  const data = await fetchJSON(`/api/records?${collectParams(page)}`);
  renderRecords(data.items || []);
  renderPagination(data.pagination);
  filterCount.textContent = `${data.pagination?.total || 0} ${currentFacets?.item_label || "records"}`;
}

async function deleteRecord(id) {
  if (!confirm(`Delete ${id}?`)) return;
  const response = await fetchJSON(`/api/records/${id}?source=${currentSource}`, { method: "DELETE" });
  if (response.ok) await Promise.all([fetchOverview(), fetchRecords(currentPage)]);
}

async function bootstrap(resetView = true) {
  resetLoadingState();
  if (resetView) currentView = currentSource === "scheduler" ? "jobs" : "records";
  await fetchConfig();
  await fetchFacets();
  await Promise.all([fetchOverview(), fetchRecords(1)]);
}

function showError(error) {
  recordList.innerHTML = "";
  recordList.appendChild(el("div", "empty-state", `Error: ${error.message}`));
}

refreshBtn.addEventListener("click", () => bootstrap(false).catch(showError));
searchBtn.addEventListener("click", () => fetchRecords(1).catch(showError));
searchInput.addEventListener("keydown", event => {
  if (event.key === "Enter") fetchRecords(1).catch(showError);
});
sourceSelect.addEventListener("change", () => {
  currentSource = sourceSelect.value;
  onlyFailedInput.checked = false;
  bootstrap(true).catch(showError);
});
viewSelect.addEventListener("change", () => {
  currentView = viewSelect.value;
  onlyFailedInput.checked = false;
  renderFilters();
  fetchRecords(1).catch(showError);
});
onlyFailedInput.addEventListener("change", () => fetchRecords(1).catch(showError));

bootstrap(true).catch(showError);
