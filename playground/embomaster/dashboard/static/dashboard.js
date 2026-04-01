(function () {
  var state = {
    runs: [],
    runId: "",
    auto: true,
    timer: null,
    overview: null,
    rounds: [],
    route: { nodes: [], edges: [] },
    selectedRound: null,
    roundDetail: null,
    streamEntries: [],
    selectedEntryIndex: null,
    pods: [],
    podCache: null,
    selectedPodKey: "",
    routeFilter: "all",
    routeSearch: "",
    podRoundFilter: "all",
    podSourceFilter: "all",
    podStatusFilter: "all",
    podSearch: "",
    artifacts: null,
    selectedArtifactPath: ""
  };

  var el = {
    runSelect: document.getElementById("runSelect"),
    reloadRunsBtn: document.getElementById("reloadRunsBtn"),
    refreshBtn: document.getElementById("refreshBtn"),
    autoBtn: document.getElementById("autoBtn"),
    runMeta: document.getElementById("runMeta"),
    runWarn: document.getElementById("runWarn"),
    statRounds: document.getElementById("statRounds"),
    statBestMetric: document.getElementById("statBestMetric"),
    statBestRound: document.getElementById("statBestRound"),
    statDebugTests: document.getElementById("statDebugTests"),
    statPods: document.getElementById("statPods"),
    statStatus: document.getElementById("statStatus"),
    routeFilterSelect: document.getElementById("routeFilterSelect"),
    routeSearchInput: document.getElementById("routeSearchInput"),
    routeSvg: document.getElementById("routeSvg"),
    routeEmpty: document.getElementById("routeEmpty"),
    roundList: document.getElementById("roundList"),
    selectedRoundTag: document.getElementById("selectedRoundTag"),
    roundDetail: document.getElementById("roundDetail"),
    streamList: document.getElementById("streamList"),
    entryDetail: document.getElementById("entryDetail"),
    podTailInput: document.getElementById("podTailInput"),
    podRoundFilter: document.getElementById("podRoundFilter"),
    podSourceFilter: document.getElementById("podSourceFilter"),
    podStatusFilter: document.getElementById("podStatusFilter"),
    podSearchInput: document.getElementById("podSearchInput"),
    podRefreshBtn: document.getElementById("podRefreshBtn"),
    podCacheMeta: document.getElementById("podCacheMeta"),
    podTable: document.getElementById("podTable"),
    podLog: document.getElementById("podLog"),
    artifactList: document.getElementById("artifactList"),
    artifactPreview: document.getElementById("artifactPreview")
  };

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function fetchJSON(path) {
    return fetch(path, { cache: "no-store" }).then(function (res) {
      if (!res.ok) {
        throw new Error("HTTP " + res.status + " " + path);
      }
      return res.json();
    });
  }

  function fmtMetric(value) {
    if (value == null || value === "") {
      return "-";
    }
    var num = Number(value);
    if (!isFinite(num)) {
      return String(value);
    }
    return num.toFixed(num >= 100 ? 1 : 4).replace(/\.?0+$/, "");
  }

  function fmtCount(value) {
    return String(value == null ? 0 : value);
  }

  function badgeClass(status) {
    var s = String(status || "unknown").toLowerCase();
    if (["completed", "succeeded", "success", "ready", "true"].indexOf(s) >= 0) return "good";
    if (["failed", "error", "timeout", "invalid_artifacts", "false"].indexOf(s) >= 0) return "bad";
    if (["best"].indexOf(s) >= 0) return "best";
    return "warn";
  }

  function badge(label, cls) {
    return '<span class="badge ' + escapeHtml(cls || badgeClass(label)) + '">' + escapeHtml(label) + "</span>";
  }

  function normalizePodSource(source) {
    var value = String(source || "").toLowerCase();
    if (value.indexOf("debug") >= 0) return "debug";
    if (value === "job_pod") return "job_pod";
    if (value === "job") return "job";
    return value || "other";
  }

  function currentSelectedPodItem() {
    return (state.pods || []).find(function (item) {
      var key = [item.source || "", item.round_index || "", item.pod_name || "", item.job_name || ""].join("|");
      return key === state.selectedPodKey;
    }) || null;
  }

  function loadRuns(force) {
    return fetchJSON("/api/runs" + (force ? "?refresh=1" : "")).then(function (data) {
      state.runs = Array.isArray(data.runs) ? data.runs : [];
      if (!state.runId || !state.runs.some(function (item) { return item.run_id === state.runId; })) {
        state.runId = data.default_run_id || (state.runs[0] ? state.runs[0].run_id : "");
      }
      renderRunSelector();
    });
  }

  function renderRunSelector() {
    if (!el.runSelect) return;
    el.runSelect.innerHTML = "";
    state.runs.forEach(function (run) {
      var option = document.createElement("option");
      option.value = run.run_id;
      option.textContent = run.label;
      el.runSelect.appendChild(option);
    });
    if (state.runId) {
      el.runSelect.value = state.runId;
    }
  }

  function resetRoundPanels() {
    state.roundDetail = null;
    state.streamEntries = [];
    state.selectedEntryIndex = null;
    state.artifacts = null;
    state.selectedArtifactPath = "";
    el.roundDetail.textContent = "请选择左侧轮次";
    el.streamList.textContent = "请选择轮次后加载";
    el.entryDetail.textContent = "点击 step 查看细节";
    el.artifactList.textContent = "请选择轮次后加载";
    el.artifactPreview.textContent = "点击文本文件预览内容";
    el.selectedRoundTag.textContent = "未选择";
  }

  function loadRun() {
    if (!state.runId) {
      return Promise.resolve();
    }
    return Promise.all([
      fetchJSON("/api/overview?run_id=" + encodeURIComponent(state.runId)),
      fetchJSON("/api/rounds?run_id=" + encodeURIComponent(state.runId)),
      fetchJSON("/api/route?run_id=" + encodeURIComponent(state.runId)),
      fetchJSON(
        "/api/pods?run_id=" +
        encodeURIComponent(state.runId) +
        "&tail=" + encodeURIComponent(Number(el.podTailInput.value || 300))
      )
    ]).then(function (results) {
      state.overview = results[0];
      state.rounds = Array.isArray(results[1].rounds) ? results[1].rounds : [];
      state.route = results[2] || { nodes: [], edges: [] };
      state.podCache = results[3] && results[3].cache ? results[3].cache : null;
      state.pods = Array.isArray(results[3].items) ? results[3].items : [];
      renderOverview();
      renderRoute();
      renderRounds();
      renderPods();
      var roundIds = state.rounds.map(function (item) { return Number(item.round_index); });
      if (roundIds.indexOf(Number(state.selectedRound)) < 0) {
        var fallback = state.overview.best_round_index || (state.rounds.length ? state.rounds[state.rounds.length - 1].round_index : null);
        state.selectedRound = fallback || null;
      }
      if (state.selectedRound != null) {
        return selectRound(state.selectedRound, true).then(function () {
          var selectedPod = currentSelectedPodItem();
          if (selectedPod) {
            return loadPodLogs(selectedPod, { silent: true, refresh: false });
          }
          return null;
        });
      }
      resetRoundPanels();
      return null;
    }).catch(function (err) {
      setWarning(String(err && err.message ? err.message : err));
    });
  }

  function renderOverview() {
    var meta = state.overview || {};
    var run = meta.run || {};
    if (el.runMeta) {
      el.runMeta.textContent =
        (run.label || "") +
        (run.path ? " | " + run.path : "") +
        (meta.monitor_mode ? " | mode: " + meta.monitor_mode : "") +
        (meta.updated_at ? " | updated: " + meta.updated_at : "");
    }
    if (el.statRounds) el.statRounds.textContent = fmtCount(meta.total_rounds);
    if (el.statBestMetric) el.statBestMetric.textContent = fmtMetric(meta.best_metric);
    if (el.statBestRound) el.statBestRound.textContent = meta.best_round_index == null ? "-" : String(meta.best_round_index);
    if (el.statDebugTests) el.statDebugTests.textContent = fmtCount(meta.total_debug_tests);
    if (el.statPods) el.statPods.textContent = fmtCount(meta.pod_count) + " / " + fmtCount(meta.job_count);
    if (el.statStatus) el.statStatus.innerHTML = badge(meta.status || "-", badgeClass(meta.status));
    setWarning(meta.warning || "");
  }

  function setWarning(text) {
    if (el.runWarn) {
      el.runWarn.textContent = text ? String(text) : "";
    }
  }

  function renderRounds() {
    if (!el.roundList) return;
    if (!state.rounds.length) {
      el.roundList.textContent = "暂无 round 数据";
      return;
    }
    el.roundList.innerHTML = state.rounds.map(function (item) {
      var roundIndex = Number(item.round_index);
      var isActive = Number(state.selectedRound) === roundIndex;
      var metric = fmtMetric(item.metric_value);
      var ratio = 0;
      if (state.rounds.length > 1) {
        ratio = roundIndex / state.rounds.length;
      }
      return (
        '<div class="round-card' + (isActive ? " active" : "") + '" data-round-index="' + roundIndex + '">' +
          '<div class="round-title">' +
            "<strong>Round " + roundIndex + "</strong>" +
            (item.is_best_round ? badge("best", "best") : badge(item.k8s_status || item.status || "unknown")) +
          "</div>" +
          '<div class="round-meta">' +
            "<span>metric: " + escapeHtml(metric) + "</span>" +
            "<span>entries: " + escapeHtml(fmtCount(item.entry_count)) + "</span>" +
          "</div>" +
          '<div class="meta-chip-row">' +
            badge(item.result_valid ? "valid" : "invalid", item.result_valid ? "good" : "bad") +
            badge("debug " + fmtCount(item.debug_test_count), "warn") +
            badge((item.parent_choice_used || "parent: n/a"), "warn") +
          "</div>" +
          '<div class="round-bar"><span style="width:' + Math.max(12, Math.round(ratio * 100)) + '%"></span></div>' +
        "</div>"
      );
    }).join("");
    Array.prototype.forEach.call(el.roundList.querySelectorAll(".round-card"), function (node) {
      node.addEventListener("click", function () {
        var roundIndex = Number(node.getAttribute("data-round-index"));
        selectRound(roundIndex, false);
      });
    });
  }

  function svgEl(name, attrs) {
    var node = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.keys(attrs || {}).forEach(function (key) {
      node.setAttribute(key, attrs[key]);
    });
    return node;
  }

  function routeNodeVisible(node) {
    var mode = String(state.routeFilter || "all");
    if (mode === "best" && !node.is_best_round) return false;
    if (mode === "valid" && !node.result_valid) return false;
    if (mode === "invalid" && node.result_valid) return false;
    if (mode === "active") {
      var s = String(node.k8s_status || node.status || "").toLowerCase();
      if (["running", "pending", "unknown", "skipped", "submitted"].indexOf(s) < 0) return false;
    }
    var text = String(state.routeSearch || "").trim().toLowerCase();
    if (!text) return true;
    var haystack = [
      node.round_index,
      node.workspace_id,
      node.parent_workspace_id,
      node.parent_choice_used,
      node.k8s_status,
      node.status
    ].join(" ").toLowerCase();
    return haystack.indexOf(text) >= 0;
  }

  function renderRoute() {
    var svg = el.routeSvg;
    if (!svg) return;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var allNodes = state.route && Array.isArray(state.route.nodes) ? state.route.nodes.slice() : [];
    var allEdges = state.route && Array.isArray(state.route.edges) ? state.route.edges.slice() : [];
    var nodes = allNodes.filter(routeNodeVisible);
    var visibleRounds = {};
    nodes.forEach(function (node) {
      visibleRounds[node.round_index] = true;
    });
    var edges = allEdges.filter(function (edge) {
      return visibleRounds[edge.source] && visibleRounds[edge.target];
    });
    if (!nodes.length) {
      if (el.routeEmpty) {
        el.routeEmpty.style.display = "block";
        el.routeEmpty.textContent = allNodes.length ? "当前筛选条件下没有 round" : "暂无 round 路线数据";
      }
      return;
    }
    if (el.routeEmpty) el.routeEmpty.style.display = "none";
    nodes.sort(function (a, b) { return Number(a.round_index) - Number(b.round_index); });
    var width = 900;
    var height = 260;
    var marginX = 70;
    var marginY = 34;
    var metricValues = nodes
      .map(function (node) { return node.metric_value == null ? null : Number(node.metric_value); })
      .filter(function (value) { return value != null && isFinite(value); });
    var minMetric = metricValues.length ? Math.min.apply(null, metricValues) : 0;
    var maxMetric = metricValues.length ? Math.max.apply(null, metricValues) : 1;

    function yFor(node, idx) {
      if (node.metric_value != null && isFinite(Number(node.metric_value)) && metricValues.length) {
        if (maxMetric === minMetric) return height / 2;
        var ratio = (Number(node.metric_value) - minMetric) / (maxMetric - minMetric);
        return height - marginY - ratio * (height - marginY * 2);
      }
      return height / 2 + ((idx % 2 === 0 ? -1 : 1) * 38);
    }

    var positions = {};
    nodes.forEach(function (node, idx) {
      var x = marginX + (nodes.length === 1 ? (width - marginX * 2) / 2 : idx * ((width - marginX * 2) / (nodes.length - 1)));
      positions[node.round_index] = { x: x, y: yFor(node, idx) };
    });

    svg.appendChild(svgEl("line", {
      x1: String(marginX - 20),
      y1: String(height - marginY),
      x2: String(width - marginX + 20),
      y2: String(height - marginY),
      stroke: "rgba(24,36,31,0.12)",
      "stroke-width": "1"
    }));

    edges.forEach(function (edge) {
      var from = positions[edge.source];
      var to = positions[edge.target];
      if (!from || !to) return;
      var path = "M " + from.x + " " + from.y + " C " + (from.x + 40) + " " + from.y + ", " + (to.x - 40) + " " + to.y + ", " + to.x + " " + to.y;
      svg.appendChild(svgEl("path", {
        d: path,
        fill: "none",
        stroke: "rgba(24,36,31,0.24)",
        "stroke-width": edge.type === "best" ? "3" : "2"
      }));
    });

    nodes.forEach(function (node) {
      var pos = positions[node.round_index];
      var tone = badgeClass(node.result_valid ? "good" : (node.k8s_status || node.status));
      var fillMap = {
        good: "#0f7a5a",
        bad: "#b53c2f",
        warn: "#a86a00",
        best: "#0057b8"
      };
      var group = svgEl("g", {
        "data-round-index": String(node.round_index),
        style: "cursor:pointer"
      });
      if (node.is_best_round) {
        group.appendChild(svgEl("circle", {
          cx: String(pos.x),
          cy: String(pos.y),
          r: "22",
          fill: "rgba(0,87,184,0.08)",
          stroke: "rgba(0,87,184,0.25)",
          "stroke-width": "2"
        }));
      }
      group.appendChild(svgEl("circle", {
        cx: String(pos.x),
        cy: String(pos.y),
        r: String(Number(state.selectedRound) === Number(node.round_index) ? 18 : 16),
        fill: fillMap[tone] || fillMap.warn
      }));
      var label = svgEl("text", {
        x: String(pos.x),
        y: String(pos.y + 4),
        "text-anchor": "middle",
        "font-size": "11",
        "font-family": "IBM Plex Sans, sans-serif",
        fill: "#fff"
      });
      label.textContent = "R" + node.round_index;
      group.appendChild(label);

      var caption = svgEl("text", {
        x: String(pos.x),
        y: String(pos.y + 34),
        "text-anchor": "middle",
        "font-size": "11",
        "font-family": "IBM Plex Sans, sans-serif",
        fill: "#4f5f58"
      });
      caption.textContent = node.metric_value == null ? "metric -" : "m=" + fmtMetric(node.metric_value);
      group.appendChild(caption);
      group.addEventListener("click", function () {
        selectRound(Number(node.round_index), false);
      });
      svg.appendChild(group);
    });
  }

  function selectRound(roundIndex, silent) {
    state.selectedRound = Number(roundIndex);
    renderRounds();
    if (!silent) {
      resetRoundPanels();
    }
    return Promise.all([
      fetchJSON("/api/round_detail?run_id=" + encodeURIComponent(state.runId) + "&round_index=" + encodeURIComponent(state.selectedRound)),
      fetchJSON("/api/stream?run_id=" + encodeURIComponent(state.runId) + "&round_index=" + encodeURIComponent(state.selectedRound) + "&limit=400"),
      fetchJSON("/api/artifacts?run_id=" + encodeURIComponent(state.runId) + "&round_index=" + encodeURIComponent(state.selectedRound))
    ]).then(function (results) {
      state.roundDetail = results[0].round || null;
      state.streamEntries = Array.isArray(results[1].entries) ? results[1].entries : [];
      state.artifacts = results[2] || null;
      state.selectedEntryIndex = state.streamEntries.length ? state.streamEntries[0].index : null;
      renderRoundDetail();
      renderStream();
      renderArtifacts();
      renderPods();
      if (state.selectedEntryIndex != null) {
        renderEntryDetail(state.streamEntries[0]);
      } else {
        el.entryDetail.textContent = "该轮暂无 step 详情";
      }
    }).catch(function (err) {
      setWarning(String(err && err.message ? err.message : err));
    });
  }

  function renderRoundDetail() {
    if (!state.roundDetail) {
      el.roundDetail.textContent = "未找到该 round 的详情";
      return;
    }
    var r = state.roundDetail;
    el.selectedRoundTag.textContent = "Round " + r.round_index;
    var html = "";
    html += '<div class="detail-card"><p class="detail-block-title">Summary</p>';
    html += '<div class="meta-grid">';
    html += "<div><strong>workspace</strong><span class=\"mono\">" + escapeHtml(r.workspace_id || "-") + "</span></div>";
    html += "<div><strong>parent</strong><span class=\"mono\">" + escapeHtml(r.parent_workspace_id || "-") + "</span></div>";
    html += "<div><strong>metric</strong><span>" + escapeHtml(fmtMetric(r.metric_value)) + "</span></div>";
    html += "<div><strong>k8s</strong><span>" + badge(r.k8s_status || "unknown") + "</span></div>";
    html += "<div><strong>valid</strong><span>" + badge(r.result_valid ? "true" : "false", r.result_valid ? "good" : "bad") + "</span></div>";
    html += "<div><strong>steps</strong><span>" + escapeHtml(fmtCount(r.steps)) + "</span></div>";
    html += "</div></div>";

    html += '<div class="detail-card"><p class="detail-block-title">Workspace</p>';
    html += '<div class="meta-grid">';
    html += "<div><strong>codebase</strong><span class=\"mono\">" + escapeHtml(r.workspace_codebase_path || "-") + "</span></div>";
    html += "<div><strong>source</strong><span>" + escapeHtml(r.workspace_source_type || "-") + "</span></div>";
    html += "<div><strong>large dirs</strong><span>" + escapeHtml(fmtCount(r.workspace_large_dirs_count)) + "</span></div>";
    html += "<div><strong>parent choice</strong><span>" + escapeHtml(r.parent_choice_used || "-") + "</span></div>";
    html += "</div></div>";

    html += '<div class="detail-card"><p class="detail-block-title">Artifact Validation</p>';
    html += '<div class="meta-chip-row">';
    html += badge((r.result_valid ? "valid" : "invalid"), r.result_valid ? "good" : "bad");
    (Array.isArray(r.validation_errors) ? r.validation_errors : []).forEach(function (item) {
      html += badge(item, "bad");
    });
    html += "</div></div>";

    if (r.k8s && (r.k8s.job_name || r.k8s.manifest_path)) {
      html += '<div class="detail-card"><p class="detail-block-title">K8S</p>';
      html += '<div class="meta-grid">';
      html += "<div><strong>job</strong><span class=\"mono\">" + escapeHtml(r.k8s.job_name || "-") + "</span></div>";
      html += "<div><strong>namespace</strong><span>" + escapeHtml(r.k8s.namespace || "-") + "</span></div>";
      html += "<div><strong>manifest</strong><span class=\"mono\">" + escapeHtml(r.k8s.manifest_path || "-") + "</span></div>";
      html += "<div><strong>pods</strong><span>" + escapeHtml(fmtCount(Array.isArray(r.k8s.pods) ? r.k8s.pods.length : 0)) + "</span></div>";
      html += "</div>";
      if (r.k8s_logs_preview) {
        html += '<p class="detail-block-title">Stored K8S Log Tail</p><pre class="mono">' + escapeHtml(r.k8s_logs_preview) + "</pre>";
      }
      if (r.manifest_text) {
        html += '<details class="fold-card"><summary class="fold-summary">Manifest</summary><div class="fold-body"><pre class="mono manifest-box">' + escapeHtml(r.manifest_text) + "</pre></div></details>";
      }
      html += "</div>";
    }

    if (r.coding_result) {
      html += '<div class="detail-card"><p class="detail-block-title">Coding Summary</p><pre class="mono">' + escapeHtml(r.coding_result) + "</pre></div>";
    }
    if (r.feedback) {
      html += '<div class="detail-card"><p class="detail-block-title">Feedback</p><pre class="mono">' + escapeHtml(r.feedback) + "</pre></div>";
    }
    el.roundDetail.innerHTML = html;
  }

  function renderStream() {
    if (!Array.isArray(state.streamEntries) || !state.streamEntries.length) {
      el.streamList.textContent = "该轮没有可展示的 step";
      el.entryDetail.textContent = "点击 step 查看细节";
      return;
    }
    el.streamList.innerHTML = state.streamEntries.map(function (entry) {
      var active = Number(state.selectedEntryIndex) === Number(entry.index);
      var pills = [];
      if (Array.isArray(entry.tool_names)) {
        pills = entry.tool_names.slice(0, 6).map(function (name) { return '<span class="pill">' + escapeHtml(name) + "</span>"; });
      }
      if (entry.debug_test_count) {
        pills.push('<span class="pill">debug ' + escapeHtml(fmtCount(entry.debug_test_count)) + "</span>");
      }
      return (
        '<div class="entry-card' + (active ? " active" : "") + '" data-entry-index="' + entry.index + '">' +
          '<div class="stream-card-head">' +
            "<strong>Step " + escapeHtml(fmtCount(entry.step)) + "</strong>" +
            badge(entry.agent_name || entry.status || "unknown") +
          "</div>" +
          "<p>" + escapeHtml(entry.assistant_preview || entry.prompt_user_preview || "-") + "</p>" +
          '<div class="tool-pills">' + pills.join("") + "</div>" +
        "</div>"
      );
    }).join("");
    Array.prototype.forEach.call(el.streamList.querySelectorAll(".entry-card"), function (node) {
      node.addEventListener("click", function () {
        var idx = Number(node.getAttribute("data-entry-index"));
        state.selectedEntryIndex = idx;
        renderStream();
        var entry = state.streamEntries.find(function (item) { return Number(item.index) === idx; });
        if (entry) {
          renderEntryDetail(entry);
        }
      });
    });
  }

  function renderEntryDetail(entry) {
    if (!entry) {
      el.entryDetail.textContent = "点击 step 查看细节";
      return;
    }
    var html = "";
    html += '<div class="detail-card">';
    html += '<p class="detail-block-title">Entry Meta</p>';
    html += '<div class="meta-grid">';
    html += "<div><strong>agent</strong><span>" + escapeHtml(entry.agent_name || "-") + "</span></div>";
    html += "<div><strong>round</strong><span>" + escapeHtml(fmtCount(entry.exp_index)) + "</span></div>";
    html += "<div><strong>workspace</strong><span class=\"mono\">" + escapeHtml(entry.workspace_id || "-") + "</span></div>";
    html += "<div><strong>task type</strong><span>" + escapeHtml(entry.task_type || "-") + "</span></div>";
    html += "</div></div>";
    html += '<div class="detail-card"><p class="detail-block-title">Assistant</p><pre class="mono">' + escapeHtml(entry.assistant_text || "-") + "</pre></div>";
    html += '<div class="detail-card"><p class="detail-block-title">Prompt Preview</p><pre class="mono">' + escapeHtml(entry.prompt_user_preview || "-") + "</pre></div>";
    if (Array.isArray(entry.tool_responses) && entry.tool_responses.length) {
      html += '<div class="detail-card"><p class="detail-block-title">Tool Responses</p>';
      entry.tool_responses.forEach(function (item) {
        html += '<div class="artifact-group">';
        html += "<strong>" + escapeHtml(item.name || "tool") + "</strong>";
        html += "<pre class=\"mono\">" + escapeHtml(item.content || "") + "</pre>";
        html += "</div>";
      });
      html += "</div>";
    }
    el.entryDetail.innerHTML = html;
  }

  function renderPodCacheMeta() {
    if (!el.podCacheMeta) return;
    var cache = state.podCache || {};
    if (!cache || !cache.enabled) {
      el.podCacheMeta.textContent = "后台轮询缓存未就绪";
      return;
    }
    var parts = ["后台轮询 " + fmtCount(cache.poll_interval_sec) + "s"];
    if (cache.last_refresh && cache.last_refresh !== "-") parts.push("最近刷新: " + cache.last_refresh);
    if (cache.age_sec != null) parts.push("age: " + cache.age_sec + "s");
    if (cache.inflight) parts.push("刷新中");
    if (cache.log_tail) parts.push("log tail: " + cache.log_tail);
    if (cache.error) parts.push("error: " + cache.error);
    el.podCacheMeta.textContent = parts.join(" | ");
  }

  function podItemVisible(item) {
    if (!item) return false;
    if (state.podRoundFilter === "selected" && Number(item.round_index || 0) !== Number(state.selectedRound || 0)) {
      return false;
    }
    if (state.podSourceFilter !== "all" && normalizePodSource(item.source) !== state.podSourceFilter) {
      return false;
    }
    if (state.podStatusFilter !== "all") {
      var status = String(item.status || "").toLowerCase();
      if (state.podStatusFilter === "ready") {
        var readyParts = String(item.ready_summary || "").split("/");
        if (readyParts.length !== 2 || readyParts[0] !== readyParts[1]) return false;
      } else if (status !== state.podStatusFilter) {
        return false;
      }
    }
    var text = String(state.podSearch || "").trim().toLowerCase();
    if (!text) return true;
    var haystack = [
      item.pod_name,
      item.resolved_pod_name,
      item.job_name,
      item.node_name,
      item.namespace,
      item.source,
      item.status,
      item.round_index
    ].join(" ").toLowerCase();
    return haystack.indexOf(text) >= 0;
  }

  function renderPods() {
    renderPodCacheMeta();
    if (!Array.isArray(state.pods) || !state.pods.length) {
      el.podTable.textContent = "暂无 pod / job 数据";
      return;
    }
    var selectedRound = Number(state.selectedRound);
    var items = state.pods.filter(podItemVisible).sort(function (a, b) {
      var aRound = Number(a.round_index || 0);
      var bRound = Number(b.round_index || 0);
      if (aRound === selectedRound && bRound !== selectedRound) return -1;
      if (bRound === selectedRound && aRound !== selectedRound) return 1;
      return bRound - aRound;
    });
    if (!items.length) {
      el.podTable.textContent = "当前筛选条件下没有 pod / job";
      return;
    }
    el.podTable.innerHTML = items.map(function (item, idx) {
      var key = [item.source || "", item.round_index || "", item.pod_name || "", item.job_name || ""].join("|");
      var active = state.selectedPodKey === key;
      var displayName = item.resolved_pod_name || item.pod_name || item.job_name || "-";
      var summary = ["round " + fmtCount(item.round_index), item.namespace || "default"];
      if (item.ready_summary && item.ready_summary !== "-") summary.push("ready " + item.ready_summary);
      return (
        '<div class="pod-card' + (active ? " active" : "") + '" data-pod-key="' + escapeHtml(key) + '" data-pod-index="' + idx + '">' +
          '<div class="pod-row">' +
            "<strong>" + escapeHtml(displayName) + "</strong>" +
            badge(item.status || item.source || "unknown") +
          "</div>" +
          '<div class="pod-row">' +
            "<span>" + escapeHtml(summary.join(" | ")) + "</span>" +
            "<span>" + escapeHtml(item.cache_updated_at || item.log_updated_at || "-") + "</span>" +
          "</div>" +
          '<div class="meta-chip-row">' +
            badge(item.source || "pod") +
            (item.job_name ? badge(item.job_name, "warn") : "") +
            (item.node_name ? badge(item.node_name, "warn") : "") +
            (item.has_live_logs ? badge("logs cached", "good") : "") +
            (item.cache_error ? badge("cache error", "bad") : "") +
          "</div>" +
        "</div>"
      );
    }).join("");
    Array.prototype.forEach.call(el.podTable.querySelectorAll(".pod-card"), function (node) {
      node.addEventListener("click", function () {
        var index = Number(node.getAttribute("data-pod-index"));
        var item = items[index];
        state.selectedPodKey = node.getAttribute("data-pod-key") || "";
        renderPods();
        loadPodLogs(item, { silent: false, refresh: false });
      });
    });
  }

  function loadPodLogs(item, options) {
    if (!item) return Promise.resolve();
    options = options || {};
    var tail = Number(el.podTailInput.value || 300);
    var query = [
      "run_id=" + encodeURIComponent(state.runId),
      "pod_name=" + encodeURIComponent(item.pod_name || ""),
      "job_name=" + encodeURIComponent(item.job_name || ""),
      "namespace=" + encodeURIComponent(item.namespace || "default"),
      "tail=" + encodeURIComponent(tail)
    ];
    if (item.round_index != null) {
      query.push("round_index=" + encodeURIComponent(item.round_index));
    }
    if (options.refresh) {
      query.push("refresh=1");
    }
    if (!options.silent) {
      el.podLog.textContent = "加载日志中...";
    }
    return fetchJSON("/api/pod_logs?" + query.join("&")).then(function (data) {
      var meta = [];
      if (data.source) meta.push("source=" + data.source);
      if (data.updated_at) meta.push("updated=" + data.updated_at);
      if (data.age_seconds != null) meta.push("age=" + data.age_seconds + "s");
      var prefix = meta.length ? "[" + meta.join(" | ") + "]\n\n" : "";
      el.podLog.textContent = prefix + (data.logs || data.error || "没有日志");
    }).catch(function (err) {
      el.podLog.textContent = String(err && err.message ? err.message : err);
    });
  }

  function renderArtifacts() {
    var payload = state.artifacts;
    if (!payload || !Array.isArray(payload.categories) || !payload.categories.length) {
      el.artifactList.textContent = "该轮没有可枚举的输出目录";
      return;
    }
    el.artifactList.innerHTML = payload.categories.map(function (group) {
      var filesHtml = (group.files || []).map(function (file) {
        return (
          '<div class="artifact-file" data-artifact-path="' + escapeHtml(file.absolute_path || "") + '">' +
            "<strong>" + escapeHtml(file.relative_path || file.name || "-") + "</strong>" +
            "<span>" + escapeHtml(fmtCount(file.size_bytes)) + " B | " + escapeHtml(file.mtime || "-") + "</span>" +
          "</div>"
        );
      }).join("");
      return (
        '<div class="artifact-group">' +
          "<strong>" + escapeHtml(group.label || group.relative_root || "-") + "</strong>" +
          "<p class=\"mono\">" + escapeHtml(group.path || "-") + "</p>" +
          '<div class="artifact-files">' + filesHtml + "</div>" +
        "</div>"
      );
    }).join("");
    Array.prototype.forEach.call(el.artifactList.querySelectorAll(".artifact-file"), function (node) {
      node.addEventListener("click", function () {
        var absolutePath = node.getAttribute("data-artifact-path") || "";
        state.selectedArtifactPath = absolutePath;
        loadArtifactPreview(absolutePath);
      });
    });
  }

  function loadArtifactPreview(absolutePath) {
    if (!absolutePath || state.selectedRound == null) return;
    el.artifactPreview.textContent = "加载文件中...";
    fetchJSON(
      "/api/artifact_preview?run_id=" +
      encodeURIComponent(state.runId) +
      "&round_index=" + encodeURIComponent(state.selectedRound) +
      "&absolute_path=" + encodeURIComponent(absolutePath)
    ).then(function (data) {
      el.artifactPreview.textContent = data.ok ? data.content : (data.error || "无法预览该文件");
    }).catch(function (err) {
      el.artifactPreview.textContent = String(err && err.message ? err.message : err);
    });
  }

  function scheduleAutoRefresh() {
    if (state.timer) {
      clearTimeout(state.timer);
      state.timer = null;
    }
    if (!state.auto) return;
    state.timer = setTimeout(function () {
      loadRun().finally(scheduleAutoRefresh);
    }, 8000);
  }

  function bindEvents() {
    el.runSelect.addEventListener("change", function () {
      state.runId = el.runSelect.value;
      resetRoundPanels();
      loadRun().finally(scheduleAutoRefresh);
    });
    el.reloadRunsBtn.addEventListener("click", function () {
      loadRuns(true).then(loadRun).finally(scheduleAutoRefresh);
    });
    el.refreshBtn.addEventListener("click", function () {
      loadRun().finally(scheduleAutoRefresh);
    });
    el.autoBtn.addEventListener("click", function () {
      state.auto = !state.auto;
      el.autoBtn.textContent = "自动刷新: " + (state.auto ? "开" : "关");
      if (state.auto) {
        el.autoBtn.classList.add("primary");
      } else {
        el.autoBtn.classList.remove("primary");
      }
      scheduleAutoRefresh();
    });
    if (el.routeFilterSelect) {
      el.routeFilterSelect.addEventListener("change", function () {
        state.routeFilter = el.routeFilterSelect.value || "all";
        renderRoute();
      });
    }
    if (el.routeSearchInput) {
      el.routeSearchInput.addEventListener("input", function () {
        state.routeSearch = el.routeSearchInput.value || "";
        renderRoute();
      });
    }
    if (el.podRoundFilter) {
      el.podRoundFilter.addEventListener("change", function () {
        state.podRoundFilter = el.podRoundFilter.value || "all";
        renderPods();
      });
    }
    if (el.podSourceFilter) {
      el.podSourceFilter.addEventListener("change", function () {
        state.podSourceFilter = el.podSourceFilter.value || "all";
        renderPods();
      });
    }
    if (el.podStatusFilter) {
      el.podStatusFilter.addEventListener("change", function () {
        state.podStatusFilter = el.podStatusFilter.value || "all";
        renderPods();
      });
    }
    if (el.podSearchInput) {
      el.podSearchInput.addEventListener("input", function () {
        state.podSearch = el.podSearchInput.value || "";
        renderPods();
      });
    }
    if (el.podTailInput) {
      el.podTailInput.addEventListener("change", function () {
        var item = currentSelectedPodItem();
        if (item) {
          loadPodLogs(item, { silent: false, refresh: true });
        }
      });
    }
    if (el.podRefreshBtn) {
      el.podRefreshBtn.addEventListener("click", function () {
        fetchJSON(
          "/api/pods?run_id=" +
          encodeURIComponent(state.runId) +
          "&refresh=1&tail=" + encodeURIComponent(Number(el.podTailInput.value || 300))
        ).then(function (data) {
          state.podCache = data && data.cache ? data.cache : null;
          state.pods = Array.isArray(data.items) ? data.items : [];
          renderPods();
          var item = currentSelectedPodItem();
          if (item) {
            return loadPodLogs(item, { silent: false, refresh: true });
          }
          return null;
        }).catch(function (err) {
          setWarning(String(err && err.message ? err.message : err));
        });
      });
    }
  }

  bindEvents();
  loadRuns(true).then(loadRun).finally(scheduleAutoRefresh);
})();
