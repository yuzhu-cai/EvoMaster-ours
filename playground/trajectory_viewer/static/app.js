const $ = (id) => document.getElementById(id);

const el = {
  experimentSelect: $("experimentSelect"),
  trajectorySelect: $("trajectorySelect"),
  refreshBtn: $("refreshBtn"),
  metricStrip: $("metricStrip"),
  runCount: $("runCount"),
  experimentFilter: $("experimentFilter"),
  experimentList: $("experimentList"),
  entryCount: $("entryCount"),
  entryFilter: $("entryFilter"),
  lastEntryBtn: $("lastEntryBtn"),
  timelineList: $("timelineList"),
  entryTitle: $("entryTitle"),
  entryMeta: $("entryMeta"),
  messageFilter: $("messageFilter"),
  messageList: $("messageList"),
  collapseBtn: $("collapseBtn"),
  rawBtn: $("rawBtn"),
  viewTabs: $("viewTabs"),
  viewStats: $("viewStats"),
  comparePane: $("comparePane"),
  rawModal: $("rawModal"),
  rawJson: $("rawJson"),
  closeRawBtn: $("closeRawBtn"),
};

const state = {
  runsRoot: "",
  experiments: [],
  currentExperiment: "",
  trajectories: [],
  currentTrajectory: "",
  trajectorySummary: null,
  entries: [],
  currentEntryIndex: null,
  currentEntry: null,
  compact: false,
  messageView: "step",
  roles: new Set(["system", "user", "assistant", "tool"]),
  loadingToken: 0,
};

function withParams(path, params = {}) {
  const url = new URL(path, window.location.origin);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null) url.searchParams.set(key, value);
  });
  return url.toString();
}

async function getJson(path, params) {
  const response = await fetch(withParams(path, params));
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function text(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.textContent = value ?? "";
  return node;
}

function button(className, label) {
  const node = document.createElement("button");
  node.className = className;
  node.type = "button";
  node.textContent = label;
  return node;
}

function formatNumber(value) {
  if (value === null || value === undefined || value === "") return "-";
  const number = Number(value);
  if (!Number.isFinite(number)) return String(value);
  if (Math.abs(number) >= 1000000) return number.toLocaleString(undefined, { maximumFractionDigits: 1 });
  if (Math.abs(number) >= 1000) return number.toLocaleString();
  if (Number.isInteger(number)) return String(number);
  return number.toPrecision(6);
}

function formatDate(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return value;
  return date.toLocaleString(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function compactText(value, length = 150) {
  if (value === null || value === undefined) return "";
  const textValue = typeof value === "string" ? value : JSON.stringify(value);
  const oneLine = textValue.replace(/\s+/g, " ").trim();
  return oneLine.length > length ? `${oneLine.slice(0, length)}...` : oneLine;
}

function makePill(label) {
  return text("span", "pill", label);
}

function callInfo(entry) {
  const taskId = entry.task_id || "";
  const turnMatch = taskId.match(/_step_(\d+)$/);
  const continueMatch = taskId.match(/(?:^|_)(continue_\d+)_/);
  const agentMatch = taskId.match(/_([^_]+)_step_\d+$/);
  const stepTools = (entry.step_tools || []).map(([name, count]) => `${name}${count > 1 ? ` x${count}` : ""}`);
  const topTools = (entry.top_tools || []).map(([name, count]) => `${name}${count > 1 ? ` x${count}` : ""}`);
  return {
    callNo: entry.index + 1,
    round: continueMatch ? continueMatch[1].replace("_", " ") : "initial",
    turn: entry.steps || turnMatch?.[1] || "-",
    agent: agentMatch?.[1] || entry.agent_name || "agent",
    primaryTools: stepTools.length ? stepTools : topTools.slice(0, 2),
    taskId,
  };
}

function timelineLine(label, value) {
  const node = document.createElement("div");
  node.className = "timeline-line";
  node.appendChild(text("span", "timeline-line-label", label));
  node.appendChild(text("span", "timeline-line-value", value));
  return node;
}

function setLoading(message) {
  clear(el.messageList);
  el.messageList.appendChild(text("div", "empty", message));
}

function setError(target, error) {
  clear(target);
  target.appendChild(text("div", "error", error.message || String(error)));
}

async function boot(preferredExperiment = state.currentExperiment, preferredTrajectory = state.currentTrajectory) {
  const token = ++state.loadingToken;
  setLoading("Opening the run archive...");
  try {
    const payload = await getJson("/api/experiments");
    if (token !== state.loadingToken) return;
    state.runsRoot = payload.runs_root || "";
    state.experiments = payload.experiments || [];
    const nextExperiment = pickExisting(preferredExperiment, state.experiments.map((x) => x.id));
    state.currentExperiment = nextExperiment || "";
    renderExperiments();
    renderExperimentSelect();
    if (state.currentExperiment) {
      await loadExperiment(state.currentExperiment, preferredTrajectory);
    } else {
      renderMetrics();
      clear(el.timelineList);
      clear(el.messageList);
      el.messageList.appendChild(text("div", "empty", `No experiments found in ${state.runsRoot || "runs"}.`));
    }
  } catch (error) {
    setError(el.messageList, error);
  }
}

function pickExisting(preferred, values) {
  if (preferred && values.includes(preferred)) return preferred;
  return values[0] || "";
}

async function loadExperiment(experimentId, preferredTrajectory = "") {
  state.currentExperiment = experimentId;
  state.trajectories = [];
  state.currentTrajectory = "";
  state.entries = [];
  state.currentEntryIndex = null;
  state.currentEntry = null;
  renderExperiments();
  renderExperimentSelect();
  renderMetrics();
  clear(el.timelineList);
  clear(el.messageList);
  el.timelineList.appendChild(text("div", "empty", "Reading trajectories..."));
  el.messageList.appendChild(text("div", "empty", "Choose a timeline entry after the trajectory loads."));
  try {
    const payload = await getJson("/api/trajectories", { experiment: experimentId });
    state.trajectories = payload.trajectories || [];
    const nextTrajectory = pickExisting(preferredTrajectory, state.trajectories.map((x) => x.id));
    renderTrajectorySelect();
    if (nextTrajectory) {
      await loadTrajectory(nextTrajectory);
    } else {
      clear(el.timelineList);
      el.timelineList.appendChild(text("div", "empty", "This experiment has no trajectory.json files."));
      renderMetrics();
    }
  } catch (error) {
    setError(el.timelineList, error);
  }
}

async function loadTrajectory(trajectoryId) {
  state.currentTrajectory = trajectoryId;
  state.currentEntryIndex = null;
  state.currentEntry = null;
  renderTrajectorySelect();
  clear(el.timelineList);
  clear(el.messageList);
  el.timelineList.appendChild(text("div", "empty", "Composing the timeline..."));
  el.messageList.appendChild(text("div", "empty", "Loading transcript..."));
  try {
    const payload = await getJson("/api/trajectory", {
      experiment: state.currentExperiment,
      trajectory: trajectoryId,
    });
    state.trajectorySummary = payload;
    state.entries = payload.entries || [];
    renderTimeline();
    renderMetrics();
    const latest = state.entries.length ? state.entries[state.entries.length - 1].index : null;
    if (latest !== null) await loadEntry(latest);
  } catch (error) {
    setError(el.timelineList, error);
    setError(el.messageList, error);
  }
}

async function loadEntry(index) {
  state.currentEntryIndex = index;
  renderTimeline();
  clear(el.messageList);
  clear(el.viewStats);
  clear(el.comparePane);
  el.comparePane.hidden = true;
  el.messageList.appendChild(text("div", "empty", `Loading entry ${index + 1}...`));
  try {
    const payload = await getJson("/api/entry", {
      experiment: state.currentExperiment,
      trajectory: state.currentTrajectory,
      index,
      view: state.messageView,
    });
    state.currentEntry = payload;
    renderEntry();
    renderMetrics();
  } catch (error) {
    setError(el.messageList, error);
  }
}

function renderExperimentSelect() {
  clear(el.experimentSelect);
  if (!state.experiments.length) {
    const option = new Option("No experiments", "");
    el.experimentSelect.appendChild(option);
    return;
  }
  state.experiments.forEach((experiment) => {
    const option = new Option(experiment.id, experiment.id);
    option.selected = experiment.id === state.currentExperiment;
    el.experimentSelect.appendChild(option);
  });
}

function renderTrajectorySelect() {
  clear(el.trajectorySelect);
  if (!state.trajectories.length) {
    el.trajectorySelect.appendChild(new Option("No trajectories", ""));
    return;
  }
  state.trajectories.forEach((trajectory) => {
    const label = `${trajectory.name} - ${trajectory.entry_count || 0} entries`;
    const option = new Option(label, trajectory.id);
    option.selected = trajectory.id === state.currentTrajectory;
    el.trajectorySelect.appendChild(option);
  });
}

function renderExperiments() {
  el.runCount.textContent = String(state.experiments.length);
  const filter = el.experimentFilter.value.trim().toLowerCase();
  clear(el.experimentList);
  const visible = state.experiments.filter((experiment) => {
    const haystack = `${experiment.id} ${JSON.stringify(experiment.best_scores || [])}`.toLowerCase();
    return !filter || haystack.includes(filter);
  });
  if (!visible.length) {
    el.experimentList.appendChild(text("div", "empty", "No experiment matches the filter."));
    return;
  }
  visible.forEach((experiment) => {
    const card = button(`experiment-card${experiment.id === state.currentExperiment ? " active" : ""}`, "");
    card.appendChild(text("div", "card-title", experiment.id));
    card.appendChild(text("div", "card-sub", `${experiment.trajectory_count} trajectories - modified ${formatDate(experiment.modified_at)}`));
    const row = document.createElement("div");
    row.className = "score-row";
    (experiment.best_scores || []).slice(0, 3).forEach((score) => {
      row.appendChild(makePill(`${score.trajectory}: ${score.metric || "metric"} ${formatNumber(score.score)}`));
    });
    if (!row.childNodes.length) row.appendChild(makePill("no best_meta"));
    card.appendChild(row);
    card.addEventListener("click", () => loadExperiment(experiment.id));
    el.experimentList.appendChild(card);
  });
}

function renderMetrics() {
  clear(el.metricStrip);
  const experiment = state.experiments.find((item) => item.id === state.currentExperiment);
  const trajectory = state.trajectorySummary;
  const bestMeta = trajectory?.best_meta || {};
  const entry = state.currentEntry?.summary || {};
  const metrics = [
    ["Experiments", state.experiments.length, state.runsRoot || "runs"],
    ["Trajectories", experiment?.trajectory_count ?? state.trajectories.length, state.currentExperiment || "none"],
    ["Entries", trajectory?.entry_count ?? state.entries.length, state.currentTrajectory || "none"],
    ["Messages", entry.message_count ?? trajectory?.last_message_count ?? "-", `selected #${state.currentEntryIndex !== null ? state.currentEntryIndex + 1 : "-"}`],
    ["Best score", bestMeta.score !== undefined ? formatNumber(bestMeta.score) : "-", bestMeta.metric ? `${bestMeta.metric} / ${bestMeta.direction || ""}` : "no best_meta"],
  ];
  metrics.forEach(([label, value, note]) => {
    const card = document.createElement("div");
    card.className = "metric-card";
    card.appendChild(text("div", "metric-label", label));
    card.appendChild(text("div", "metric-value", String(value)));
    card.appendChild(text("div", "metric-note", note));
    el.metricStrip.appendChild(card);
  });
}

function renderTimeline() {
  el.entryCount.textContent = String(state.entries.length);
  clear(el.timelineList);
  const filter = el.entryFilter.value.trim().toLowerCase();
  const visible = state.entries.filter((entry) => {
    const info = callInfo(entry);
    const tools = (entry.top_tools || []).map((item) => item[0]).join(" ");
    const stepTools = (entry.step_tools || []).map((item) => item[0]).join(" ");
    const haystack = `call ${info.callNo} ${info.round} turn ${info.turn} ${entry.task_id} ${entry.status} ${entry.excerpt} ${tools} ${stepTools}`.toLowerCase();
    return !filter || haystack.includes(filter);
  });
  if (!visible.length) {
    el.timelineList.appendChild(text("div", "empty", "No timeline entries match."));
    return;
  }
  visible.forEach((entry) => {
    const info = callInfo(entry);
    const item = button(`timeline-item${entry.index === state.currentEntryIndex ? " active" : ""}`, "");
    item.dataset.index = String(info.callNo);
    item.setAttribute("aria-label", `Open call ${info.callNo}`);

    const head = document.createElement("div");
    head.className = "timeline-call-head";
    head.appendChild(text("div", "timeline-number", `#${info.callNo}`));
    const main = document.createElement("div");
    main.className = "timeline-main";
    main.appendChild(text("div", "timeline-title", info.primaryTools.length ? info.primaryTools.join(" + ") : "LLM response"));
    main.appendChild(text("div", "timeline-sub", `${info.round} · turn ${info.turn} · ${entry.status || "running"}`));
    head.appendChild(main);
    item.appendChild(head);

    const details = document.createElement("div");
    details.className = "timeline-details";
    details.appendChild(timelineLine("prompt", `${entry.message_count || 0} msgs · ${formatNumber(entry.cumulative_tokens)} tok`));
    details.appendChild(timelineLine("step", `${entry.step_message_count || 0} msgs`));
    if (entry.compacted) details.appendChild(timelineLine("context", `compressed ${entry.compaction_strategy || ""}`.trim()));
    item.appendChild(details);

    const mini = document.createElement("div");
    mini.className = "mini-row";
    (entry.step_tools || []).slice(0, 3).forEach(([name, count]) => mini.appendChild(makePill(`${name} x${count}`)));
    if (entry.summary_count) mini.appendChild(makePill(`summary x${entry.summary_count}`));
    if (entry.latest_tokens) mini.appendChild(makePill(`llm ${formatNumber(entry.latest_tokens)} tok`));
    item.appendChild(mini);
    item.appendChild(text("div", "timeline-excerpt", compactText(entry.excerpt, 180)));
    item.appendChild(text("div", "timeline-code", compactText(info.taskId, 90)));
    item.addEventListener("click", () => loadEntry(entry.index));
    el.timelineList.appendChild(item);
  });
  const active = el.timelineList.querySelector(".timeline-item.active");
  if (active) requestAnimationFrame(() => active.scrollIntoView({ block: "nearest" }));
}

function renderEntry() {
  const payload = state.currentEntry;
  if (!payload) return;
  const summary = payload.summary || {};
  const info = callInfo({ ...summary, index: payload.index });
  el.entryTitle.textContent = `Call #${info.callNo} · ${info.round} · turn ${info.turn}`;
  renderViewTabs();
  renderEntryMeta(payload);
  renderViewStats(payload);
  if (payload.view === "compare") {
    renderCompare(payload.compare);
    renderMessages([]);
  } else {
    el.comparePane.hidden = true;
    renderMessages(payload.messages || []);
  }
}

function renderEntryMeta(payload) {
  clear(el.entryMeta);
  const summary = payload.summary || {};
  const meta = payload.trajectory_meta || {};
  const pills = [
    `entry ${payload.index + 1}/${payload.entry_count}`,
    `status ${summary.status || "unknown"}`,
    `turn ${summary.steps || "-"}`,
    `agent ${summary.agent_name || meta.agent_name || "-"}`,
    `messages ${summary.message_count || 0}`,
    `tokens ${formatNumber(summary.cumulative_tokens)}`,
  ];
  const compression = payload.compression || {};
  if (compression.truncated) pills.push(`compressed ${compression.strategy || "summary"}`);
  if (compression.pruned) pills.push("tool-pruned");
  if (compression.message_delta > 0) pills.push(`hidden msgs +${compression.message_delta}`);
  if (summary.task_id) pills.push(`id ${compactText(summary.task_id, 56)}`);
  pills.forEach((label) => el.entryMeta.appendChild(makePill(label)));
}

function renderViewTabs() {
  document.querySelectorAll(".view-tab[data-view]").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.view === state.messageView);
  });
}

function viewCard(title, stats, description = "") {
  const card = document.createElement("div");
  card.className = "view-stat-card";
  card.appendChild(text("div", "view-stat-title", title));
  const values = document.createElement("div");
  values.className = "view-stat-values";
  values.appendChild(makePill(`${formatNumber(stats?.message_count)} msgs`));
  values.appendChild(makePill(`${formatNumber(stats?.estimated_tokens)} est tok`));
  values.appendChild(makePill(`A ${stats?.assistant_count || 0}`));
  values.appendChild(makePill(`T ${stats?.tool_count || 0}`));
  values.appendChild(makePill(`U ${stats?.user_count || 0}`));
  if (stats?.summary_count) values.appendChild(makePill(`summary ${stats.summary_count}`));
  if (stats?.pruned_tool_count) values.appendChild(makePill(`pruned ${stats.pruned_tool_count}`));
  card.appendChild(values);
  if (description) card.appendChild(text("div", "view-stat-desc", description));
  return card;
}

function renderViewStats(payload) {
  clear(el.viewStats);
  const views = payload.views || {};
  const compression = payload.compression || {};
  const current = views[payload.view === "compare" ? "compressed" : payload.view] || {};
  el.viewStats.appendChild(viewCard(current.label || state.messageView, current, current.description));
  if (payload.view === "compare") {
    el.viewStats.appendChild(viewCard(views.full?.label || "压缩前重建", views.full || {}));
    el.viewStats.appendChild(viewCard(views.compressed?.label || "压缩后轨迹", views.compressed || {}));
  }
  const delta = document.createElement("div");
  delta.className = "view-stat-card delta";
  delta.appendChild(text("div", "view-stat-title", "压缩差异"));
  const values = document.createElement("div");
  values.className = "view-stat-values";
  const deltaMsgs = compression.message_delta === null || compression.message_delta === undefined
    ? "load full"
    : formatNumber(compression.message_delta);
  const deltaTokens = compression.token_delta_estimate === null || compression.token_delta_estimate === undefined
    ? "load full"
    : formatNumber(compression.token_delta_estimate);
  values.appendChild(makePill(`msgs ${deltaMsgs}`));
  values.appendChild(makePill(`est tok ${deltaTokens}`));
  values.appendChild(makePill(compression.truncated ? `strategy ${compression.strategy || "summary"}` : "not truncated"));
  delta.appendChild(values);
  el.viewStats.appendChild(delta);
}

function renderCompare(compare) {
  clear(el.comparePane);
  el.comparePane.hidden = false;
  if (!compare) {
    el.comparePane.appendChild(text("div", "empty", "No compare data."));
    return;
  }
  const columns = [
    ["before", compare.before],
    ["after", compare.after],
  ];
  columns.forEach(([key, data]) => {
    const col = document.createElement("section");
    col.className = `compare-column ${key}`;
    const head = document.createElement("div");
    head.className = "compare-head";
    head.appendChild(text("div", "compare-title", data?.label || key));
    head.appendChild(text("div", "compare-sub", data?.description || ""));
    col.appendChild(head);
    const list = document.createElement("div");
    list.className = "compare-messages";
    const messages = filterMessages(data?.messages || []);
    if (!messages.length) {
      list.appendChild(text("div", "empty", "No messages visible with the current filters."));
    } else {
      messages.forEach((message) => list.appendChild(messageCard(message)));
    }
    col.appendChild(list);
    el.comparePane.appendChild(col);
  });
}

function renderMessages(messages) {
  clear(el.messageList);
  el.messageList.classList.toggle("compact-mode", state.compact);
  if (state.currentEntry?.view === "compare") {
    el.messageList.hidden = true;
    return;
  }
  el.messageList.hidden = false;
  const visible = filterMessages(messages);
  if (!visible.length) {
    el.messageList.appendChild(text("div", "empty", "No messages visible with the current filters."));
    return;
  }
  visible.forEach((message) => el.messageList.appendChild(messageCard(message)));
}

function filterMessages(messages) {
  const query = el.messageFilter.value.trim().toLowerCase();
  return messages.filter((message) => {
    if (!state.roles.has(message.role)) return false;
    if (!query) return true;
    const haystack = `${message.role} ${message.name || ""} ${message.content || ""} ${message.reasoning_content || ""} ${JSON.stringify(message.tool_calls || [])}`.toLowerCase();
    return haystack.includes(query);
  });
}

function messageCard(message) {
  const card = document.createElement("article");
  card.className = `message-card ${message.role || "unknown"}`;
  if (message.is_summary) card.classList.add("summary-message");
  if (message.is_pruned_tool) card.classList.add("pruned-message");
  const top = document.createElement("div");
  top.className = "message-top";
  const role = document.createElement("div");
  role.className = "role-badge";
  role.appendChild(text("span", "role-dot", ""));
  role.appendChild(text("span", "", message.name ? `${message.role} / ${message.name}` : message.role));
  top.appendChild(role);
  top.appendChild(text("div", "message-index", `${message.source || "view"} #${message.index + 1}`));
  card.appendChild(top);

  const metaRow = document.createElement("div");
  metaRow.className = "mini-row";
  const meta = message.meta || {};
  if (meta.model) metaRow.appendChild(makePill(meta.model));
  if (meta.finish_reason) metaRow.appendChild(makePill(`finish ${meta.finish_reason}`));
  if (meta.usage?.total_tokens) metaRow.appendChild(makePill(`${formatNumber(meta.usage.total_tokens)} tok`));
  if (message.token_estimate) metaRow.appendChild(makePill(`~${formatNumber(message.token_estimate)} tok`));
  if (message.is_summary) metaRow.appendChild(makePill("summary"));
  if (message.is_pruned_tool) metaRow.appendChild(makePill("old tool output cleared"));
  if (message.tool_call_id) metaRow.appendChild(makePill(`call ${compactText(message.tool_call_id, 24)}`));
  if (metaRow.childNodes.length) card.appendChild(metaRow);

  const bodyText = bodyForMessage(message);
  if (bodyText) {
    card.appendChild(text("div", "content", bodyText));
  }

  (message.tool_calls || []).forEach((call) => card.appendChild(toolCall(call)));
  return card;
}

function bodyForMessage(message) {
  if (typeof message.content === "string" && message.content.trim()) return message.content;
  if (message.reasoning_content) return `Reasoning:\n${message.reasoning_content}`;
  if ((message.tool_calls || []).length) return "Assistant requested tool execution.";
  if (message.content === null) return "<empty content>";
  if (message.content !== undefined) return JSON.stringify(message.content, null, 2);
  return "";
}

function toolCall(call) {
  const node = document.createElement("div");
  node.className = "tool-call";
  node.appendChild(text("div", "tool-call-title", call.name || "tool_call"));
  const value = call.parsed_arguments ?? call.arguments ?? "";
  const printable = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  node.appendChild(text("pre", "tool-args", printable));
  return node;
}

function openRaw() {
  if (!state.currentEntry) return;
  el.rawJson.textContent = JSON.stringify(state.currentEntry, null, 2);
  el.rawModal.hidden = false;
}

function closeRaw() {
  el.rawModal.hidden = true;
}

el.refreshBtn.addEventListener("click", () => boot());
el.experimentSelect.addEventListener("change", (event) => {
  if (event.target.value) loadExperiment(event.target.value);
});
el.trajectorySelect.addEventListener("change", (event) => {
  if (event.target.value) loadTrajectory(event.target.value);
});
el.experimentFilter.addEventListener("input", renderExperiments);
el.entryFilter.addEventListener("input", renderTimeline);
el.messageFilter.addEventListener("input", () => renderEntry());
el.lastEntryBtn.addEventListener("click", () => {
  if (state.entries.length) loadEntry(state.entries[state.entries.length - 1].index);
});
el.collapseBtn.addEventListener("click", () => {
  state.compact = !state.compact;
  el.collapseBtn.textContent = state.compact ? "Expand" : "Compact";
  renderEntry();
});
el.rawBtn.addEventListener("click", openRaw);
el.closeRawBtn.addEventListener("click", closeRaw);
el.rawModal.addEventListener("click", (event) => {
  if (event.target === el.rawModal) closeRaw();
});
document.querySelectorAll(".view-tab[data-view]").forEach((tab) => {
  tab.addEventListener("click", () => {
    const view = tab.dataset.view || "step";
    if (view === state.messageView) return;
    state.messageView = view;
    renderViewTabs();
    if (state.currentEntryIndex !== null) loadEntry(state.currentEntryIndex);
  });
});

document.querySelectorAll(".role-toggles input[data-role]").forEach((checkbox) => {
  checkbox.addEventListener("change", () => {
    const role = checkbox.dataset.role;
    if (checkbox.checked) state.roles.add(role);
    else state.roles.delete(role);
    renderEntry();
  });
});

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeRaw();
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    el.messageFilter.focus();
  }
});

boot();
