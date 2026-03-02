(function () {
  var state = {
    runs: [],
    runId: "",
    cursor: 0,
    entries: [],
    selectedEntryIndex: null,
    selectedExp: null,
    auto: true,
    timer: null,
    latestByExp: []
  };

  var el = {
    meta: document.getElementById("meta"),
    warn: document.getElementById("warn"),
    runSel: document.getElementById("runSel"),
    runRefresh: document.getElementById("runRefresh"),
    pullBtn: document.getElementById("pullBtn"),
    autoBtn: document.getElementById("autoBtn"),
    totalEntries: document.getElementById("totalEntries"),
    totalExps: document.getElementById("totalExps"),
    maxStep: document.getElementById("maxStep"),
    debugCount: document.getElementById("debugCount"),
    parseErrors: document.getElementById("parseErrors"),
    expList: document.getElementById("expList"),
    expContent: document.getElementById("expContent"),
    rows: document.getElementById("rows"),
    entryDetail: document.getElementById("entryDetail"),
    debugExpSel: document.getElementById("debugExpSel"),
    debugRefresh: document.getElementById("debugRefresh"),
    debugBox: document.getElementById("debugBox"),
    podSel: document.getElementById("podSel"),
    podTail: document.getElementById("podTail"),
    podFetch: document.getElementById("podFetch"),
    podBox: document.getElementById("podBox"),
    runResult: document.getElementById("runResult")
  };

  function setWarn(msg) {
    if (el.warn) {
      el.warn.textContent = msg ? String(msg) : "";
    }
  }

  window.onerror = function (_msg, _url, _line, _col, err) {
    setWarn("frontend error: " + (err && err.message ? err.message : _msg || "unknown script error"));
  };

  if (window.addEventListener) {
    window.addEventListener("unhandledrejection", function (evt) {
      var reason = evt && evt.reason ? String(evt.reason) : "unknown promise rejection";
      setWarn("frontend rejection: " + reason);
    });
  }

  function indexOfValue(arr, v) {
    if (!arr || !arr.length) return -1;
    for (var i = 0; i < arr.length; i += 1) {
      if (arr[i] === v) return i;
    }
    return -1;
  }

  function badgeClass(status) {
    var s = String(status || "unknown").toLowerCase();
    if (indexOfValue(["completed", "succeeded", "success", "ready"], s) >= 0) return "badge ok";
    if (indexOfValue(["failed", "error", "timeout"], s) >= 0) return "badge bad";
    return "badge run";
  }

  function escapeHtml(value) {
    var s = String(value == null ? "" : value);
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function badgeHtml(status) {
    var text = String(status || "unknown");
    return '<span class="' + badgeClass(text) + '">' + escapeHtml(text) + '</span>';
  }

  function preHtml(text) {
    return '<pre class="log-view mono">' + escapeHtml(text || "") + "</pre>";
  }

  function fetchJSON(path, onOk, onErr) {
    var ok = typeof onOk === "function" ? onOk : function () {};
    var fail = typeof onErr === "function" ? onErr : function () {};

    if (window.fetch) {
      fetch(path, { cache: "no-store" })
        .then(function (res) {
          if (!res.ok) {
            throw new Error("HTTP " + res.status + " " + path);
          }
          return res.json();
        })
        .then(ok)
        .catch(fail);
      return;
    }

    var xhr = new XMLHttpRequest();
    xhr.open("GET", path, true);
    xhr.setRequestHeader("Cache-Control", "no-store");
    xhr.onreadystatechange = function () {
      if (xhr.readyState !== 4) return;
      if (xhr.status < 200 || xhr.status >= 300) {
        fail(new Error("HTTP " + xhr.status + " " + path));
        return;
      }
      try {
        ok(JSON.parse(xhr.responseText));
      } catch (e) {
        fail(e);
      }
    };
    xhr.onerror = function () {
      fail(new Error("Network error " + path));
    };
    xhr.send(null);
  }

  function loadRuns(force, done) {
    fetchJSON(
      "/api/runs" + (force ? "?refresh=1" : ""),
      function (data) {
        var i;
        var old = state.runId;
        state.runs = Array.isArray(data.runs) ? data.runs : [];

        if (!el.runSel) {
          done && done(null);
          return;
        }

        el.runSel.innerHTML = "";
        for (i = 0; i < state.runs.length; i += 1) {
          var run = state.runs[i];
          var op = document.createElement("option");
          op.value = run.run_id;
          op.textContent = run.label;
          el.runSel.appendChild(op);
        }

        var pick = old;
        var found = false;
        for (i = 0; i < state.runs.length; i += 1) {
          if (state.runs[i].run_id === pick) {
            found = true;
            break;
          }
        }
        if (!pick || !found) {
          pick = data.default_run_id || (state.runs[0] ? state.runs[0].run_id : "");
        }

        state.runId = pick || "";
        if (state.runId) {
          el.runSel.value = state.runId;
        } else if (el.meta) {
          el.meta.textContent = "No trajectory files found.";
        }
        done && done(null);
      },
      function (err) {
        done && done(err);
      }
    );
  }

  function resetRunState() {
    state.cursor = 0;
    state.entries = [];
    state.selectedEntryIndex = null;
    state.selectedExp = null;
    state.latestByExp = [];

    if (el.expContent) el.expContent.textContent = "Select an exp card above.";
    if (el.entryDetail) el.entryDetail.textContent = "Select a stream row.";
    if (el.debugBox) el.debugBox.textContent = "Click 'Load Debug Logs' to fetch structured debug_test cards.";
    if (el.podBox) el.podBox.textContent = "Select pod/job then fetch logs.";
    if (el.runResult) el.runResult.innerHTML = "";
    if (el.rows) el.rows.innerHTML = "";
    if (el.expList) el.expList.innerHTML = "";
    if (el.debugExpSel) el.debugExpSel.innerHTML = '<option value="all">all</option>';
    if (el.podSel) el.podSel.innerHTML = "";
  }

  function switchRun(runId, done) {
    state.runId = runId || "";
    resetRunState();
    if (!state.runId) {
      done && done(null);
      return;
    }

    pullUpdates(function (err) {
      if (err) {
        done && done(err);
        return;
      }
      loadPods(function (err2) {
        if (err2) {
          done && done(err2);
          return;
        }
        loadRunResults(function (err3) {
          done && done(err3 || null);
        });
      });
    });
  }

  function refreshExpFilter() {
    if (!el.debugExpSel) return;

    var keep = el.debugExpSel.value;
    var vals = { all: true };
    var i;
    for (i = 0; i < state.latestByExp.length; i += 1) {
      vals[String(state.latestByExp[i].exp_index)] = true;
    }

    var sorted = [];
    for (var k in vals) {
      if (Object.prototype.hasOwnProperty.call(vals, k)) sorted.push(k);
    }

    sorted.sort(function (a, b) {
      if (a === "all") return -1;
      if (b === "all") return 1;
      return Number(a) - Number(b);
    });

    el.debugExpSel.innerHTML = "";
    for (i = 0; i < sorted.length; i += 1) {
      var op = document.createElement("option");
      op.value = sorted[i];
      op.textContent = sorted[i];
      el.debugExpSel.appendChild(op);
    }

    if (indexOfValue(sorted, keep) >= 0) {
      el.debugExpSel.value = keep;
    }
  }

  function pullUpdates(done) {
    if (!state.runId) {
      done && done(null);
      return;
    }

    fetchJSON(
      "/api/updates?run_id=" + encodeURIComponent(state.runId) + "&cursor=" + state.cursor + "&limit=500",
      function (data) {
        if (data.cursor_reset) {
          state.cursor = 0;
          state.entries = [];
        }

        if (Array.isArray(data.new_entries) && data.new_entries.length) {
          state.entries = state.entries.concat(data.new_entries);
          if (state.entries.length > 1500) {
            state.entries = state.entries.slice(state.entries.length - 1500);
          }
        }

        state.cursor = Number(data.cursor_next || state.cursor);
        state.latestByExp = Array.isArray(data.exp_latest) ? data.exp_latest : [];

        var run = data.run || {};
        var file = data.file || {};
        if (el.meta) {
          el.meta.textContent =
            (run.label || state.runId) +
            " | " +
            (file.path || "") +
            " | " +
            (data.format || "unknown") +
            " | " +
            (data.mode || "") +
            " | updated " +
            (file.mtime || "-");
        }
        setWarn(data.warning || "");

        var s = data.summary || {};
        if (el.totalEntries) el.totalEntries.textContent = String(s.total_entries == null ? 0 : s.total_entries);
        if (el.totalExps) el.totalExps.textContent = String(s.total_exps == null ? 0 : s.total_exps);
        if (el.maxStep) el.maxStep.textContent = String(s.max_step == null ? 0 : s.max_step);
        if (el.debugCount) el.debugCount.textContent = String(s.debug_test_count == null ? 0 : s.debug_test_count);
        if (el.parseErrors) el.parseErrors.textContent = String(s.parse_errors == null ? 0 : s.parse_errors);

        renderExpList();
        renderRows();
        refreshExpFilter();

        if (state.selectedEntryIndex == null && state.entries.length) {
          renderEntryDetail(state.entries[state.entries.length - 1]);
        }
        done && done(null);
      },
      function (err) {
        done && done(err);
      }
    );
  }

  function renderExpList() {
    if (!el.expList) return;

    var maxStep = 1;
    var i;
    for (i = 0; i < state.latestByExp.length; i += 1) {
      var ms = Number(state.latestByExp[i].step || 0);
      if (ms > maxStep) maxStep = ms;
    }

    el.expList.innerHTML = "";

    for (i = 0; i < state.latestByExp.length; i += 1) {
      (function (item) {
        var div = document.createElement("div");
        div.className = "exp-item" + (state.selectedExp === Number(item.exp_index) ? " pick" : "");
        div.onclick = function () {
          state.selectedExp = Number(item.exp_index);
          loadExpEntries(function (err) {
            if (err) {
              setWarn(String(err));
              return;
            }
            loadDebug(function (err2) {
              if (err2) setWarn(String(err2));
              renderExpList();
            });
          });
        };

        var row = document.createElement("div");
        row.className = "row";

        var l = document.createElement("div");
        l.textContent = "exp " + item.exp_index + " | step " + item.step;

        var r = document.createElement("span");
        r.className = badgeClass(item.status);
        r.textContent = item.status || "running";

        row.appendChild(l);
        row.appendChild(r);

        var bar = document.createElement("div");
        bar.className = "bar";
        var fill = document.createElement("span");
        fill.style.width = Math.max(2, Math.round((Number(item.step || 0) / maxStep) * 100)) + "%";
        bar.appendChild(fill);

        div.appendChild(row);
        div.appendChild(bar);
        el.expList.appendChild(div);
      })(state.latestByExp[i]);
    }
  }

  function renderRows() {
    if (!el.rows) return;

    var reversed = state.entries.slice().reverse();
    var rows = [];
    var i;
    for (i = 0; i < reversed.length; i += 1) {
      if (state.selectedExp == null || Number(reversed[i].exp_index) === Number(state.selectedExp)) {
        rows.push(reversed[i]);
      }
      if (rows.length >= 500) break;
    }

    el.rows.innerHTML = "";

    function addTd(tr, v) {
      var td = document.createElement("td");
      td.textContent = String(v == null ? "" : v);
      tr.appendChild(td);
      return td;
    }

    for (i = 0; i < rows.length; i += 1) {
      (function (item) {
        var tr = document.createElement("tr");
        if (state.selectedEntryIndex === item.index) tr.className = "pick";

        tr.onclick = function () {
          state.selectedEntryIndex = item.index;
          fetchEntry(item.index, function (err) {
            if (err) setWarn(String(err));
            renderRows();
          });
        };

        addTd(tr, item.index);
        addTd(tr, item.exp_index);
        addTd(tr, item.step);

        var st = document.createElement("td");
        var b = document.createElement("span");
        b.className = badgeClass(item.status);
        b.textContent = item.status || "running";
        st.appendChild(b);
        tr.appendChild(st);

        addTd(tr, item.message_count);
        addTd(tr, item.debug_test_count || 0);
        addTd(tr, item.last_assistant || "");

        el.rows.appendChild(tr);
      })(rows[i]);
    }
  }

  function renderEntryDetail(item) {
    if (!item || !el.entryDetail) return;

    state.selectedEntryIndex = item.index;

    var lines = [];
    lines.push("#" + item.index + " exp=" + item.exp_index + " step=" + item.step + " status=" + item.status);
    lines.push("task=" + (item.task_id || "-") + " agent=" + (item.agent_name || "-"));
    lines.push("roles=" + JSON.stringify(item.role_counts || {}));
    lines.push("");

    var msgs = Array.isArray(item.recent_messages) ? item.recent_messages : [];
    for (var i = 0; i < msgs.length; i += 1) {
      var m = msgs[i];
      lines.push("[" + (m.role || "") + "]");
      lines.push(m.content || "");
      lines.push("");
    }

    el.entryDetail.textContent = lines.join("\n");
  }

  function fetchEntry(index, done) {
    if (!state.runId) {
      done && done(null);
      return;
    }

    for (var i = 0; i < state.entries.length; i += 1) {
      if (Number(state.entries[i].index) === Number(index)) {
        renderEntryDetail(state.entries[i]);
        break;
      }
    }

    fetchJSON(
      "/api/entry?run_id=" + encodeURIComponent(state.runId) + "&index=" + Number(index),
      function (data) {
        if (data.entry) renderEntryDetail(data.entry);
        done && done(null);
      },
      function (err) {
        done && done(err);
      }
    );
  }

  function loadExpEntries(done) {
    if (!state.runId || state.selectedExp == null) {
      if (el.expContent) el.expContent.textContent = "Select an exp card above.";
      done && done(null);
      return;
    }

    fetchJSON(
      "/api/exp_entries?run_id=" + encodeURIComponent(state.runId) + "&exp_index=" + state.selectedExp + "&limit=300",
      function (data) {
        var list = Array.isArray(data.entries) ? data.entries : [];
        if (!list.length) {
          if (el.expContent) el.expContent.textContent = "No entries for exp " + state.selectedExp + ".";
          done && done(null);
          return;
        }

        var out = [];
        out.push("exp " + state.selectedExp + " | entries=" + list.length);
        out.push("");

        for (var i = 0; i < list.length; i += 1) {
          var e = list[i];
          out.push(
            "#" +
              e.index +
              " step=" +
              e.step +
              " status=" +
              e.status +
              " messages=" +
              e.message_count +
              " debug=" +
              (e.debug_test_count || 0)
          );
          if (e.last_assistant) out.push("assistant: " + e.last_assistant);
          out.push("---");
        }

        if (el.expContent) el.expContent.textContent = out.join("\n");
        done && done(null);
      },
      function (err) {
        done && done(err);
      }
    );
  }

  function loadDebug(done) {
    if (!state.runId) {
      done && done(null);
      return;
    }

    var pick = el.debugExpSel ? el.debugExpSel.value || "all" : "all";
    var expQuery = pick !== "all" ? "&exp_index=" + encodeURIComponent(pick) : "";

    fetchJSON(
      "/api/debug_tests?run_id=" + encodeURIComponent(state.runId) + expQuery + "&limit=80",
      function (data) {
        var cards = Array.isArray(data.cards) ? data.cards : [];
        if (!cards.length) {
          if (el.debugBox) el.debugBox.textContent = "No debug_test records.";
          done && done(null);
          return;
        }

        var html = [];

        for (var ci = 0; ci < cards.length; ci += 1) {
          var card = cards[ci];
          var calls = Array.isArray(card.calls) ? card.calls : [];
          var summary =
            "Exp " +
            card.exp_index +
            " | calls " +
            (card.total_calls || calls.length) +
            " | success " +
            (card.success_count || 0) +
            " | failed " +
            (card.failed_count || 0);

          html.push('<details class="round-item" ' + (ci === 0 ? "open" : "") + ">");
          html.push("<summary>" + escapeHtml(summary) + "</summary>");

          var bound = calls.length < 30 ? calls.length : 30;
          for (var i = 0; i < bound; i += 1) {
            var c = calls[i];
            var callHead =
              "#" +
              (i + 1) +
              " step " +
              c.step +
              " mode=" +
              (c.mode || "-") +
              " exit=" +
              (c.exit_code == null ? "-" : c.exit_code);

            html.push('<details class="round-item">');
            html.push("<summary>" + escapeHtml(callHead) + " " + badgeHtml(c.status || "unknown") + "</summary>");
            html.push(
              '<div class="meta">agent=' +
                escapeHtml(c.agent_name || "-") +
                " pod=" +
                escapeHtml(c.pod_name || "-") +
                " ns=" +
                escapeHtml(c.namespace || "default") +
                "</div>"
            );
            html.push('<div class="meta">working_dir=' + escapeHtml(c.working_dir || "-") + "</div>");
            html.push('<div class="label" style="margin-top:6px;">Command</div>');
            html.push(preHtml(c.command || ""));

            if (c.full_command && c.full_command !== c.command) {
              html.push('<div class="label">Full Command</div>');
              html.push(preHtml(c.full_command || ""));
            }

            html.push('<div class="label">Stdout</div>');
            html.push(preHtml(c.stdout || ""));
            html.push('<div class="label">Stderr</div>');
            html.push(preHtml(c.stderr || ""));
            html.push("</details>");
          }

          if (calls.length > 30) {
            html.push('<div class="warn">only first 30 calls rendered (total ' + calls.length + ")</div>");
          }

          html.push("</details>");
        }

        if (el.debugBox) el.debugBox.innerHTML = html.join("");
        done && done(null);
      },
      function (err) {
        done && done(err);
      }
    );
  }

  function loadPods(done) {
    if (!state.runId) {
      done && done(null);
      return;
    }

    fetchJSON(
      "/api/pods?run_id=" + encodeURIComponent(state.runId),
      function (data) {
        var pods = Array.isArray(data.items) ? data.items : [];
        if (!el.podSel) {
          done && done(null);
          return;
        }

        el.podSel.innerHTML = "";
        for (var i = 0; i < pods.length; i += 1) {
          var p = pods[i];
          var op = document.createElement("option");
          op.value = JSON.stringify({
            pod_name: p.pod_name || "",
            namespace: p.namespace || "default",
            job_name: p.job_name || ""
          });
          var tag = p.source ? "[" + p.source + "]" : "";
          var podPart = p.pod_name ? p.pod_name : "(job:" + (p.job_name || "unknown") + ")";
          op.textContent = tag + " exp=" + (p.exp_index || "-") + " " + podPart;
          el.podSel.appendChild(op);
        }

        if (!pods.length) {
          var op2 = document.createElement("option");
          op2.value = "";
          op2.textContent = "No pod/job found";
          el.podSel.appendChild(op2);
        }
        done && done(null);
      },
      function (err) {
        done && done(err);
      }
    );
  }

  function fetchPodLogs(done) {
    if (!state.runId) {
      done && done(null);
      return;
    }
    if (!el.podSel) {
      done && done(null);
      return;
    }

    var selected = el.podSel.value;
    if (!selected) {
      done && done(null);
      return;
    }

    var obj;
    try {
      obj = JSON.parse(selected);
    } catch (_err) {
      done && done(null);
      return;
    }

    var tail = Math.max(20, Math.min(5000, Number((el.podTail && el.podTail.value) || 200)));

    var q = [];
    q.push("run_id=" + encodeURIComponent(state.runId));
    q.push("tail=" + encodeURIComponent(String(tail)));
    q.push("namespace=" + encodeURIComponent(obj.namespace || "default"));
    if (obj.pod_name) q.push("pod_name=" + encodeURIComponent(obj.pod_name));
    if (obj.job_name) q.push("job_name=" + encodeURIComponent(obj.job_name));

    fetchJSON(
      "/api/pod_logs?" + q.join("&"),
      function (data) {
        var head =
          "pod=" +
          (data.pod_name || "-") +
          " job=" +
          (data.job_name || "-") +
          " ns=" +
          (data.namespace || "default") +
          " source=" +
          (data.source || "-") +
          " ok=" +
          data.ok;

        if (el.podBox) {
          el.podBox.textContent = head + "\n\n" + (data.logs || data.error || "");
        }
        done && done(null);
      },
      function (err) {
        done && done(err);
      }
    );
  }

  function loadRunResults(done) {
    if (!state.runId) {
      done && done(null);
      return;
    }

    fetchJSON(
      "/api/run_results?run_id=" + encodeURIComponent(state.runId),
      function (data) {
        var final = data.final || {};
        var rounds = Array.isArray(data.rounds) ? data.rounds : [];

        var html = [];
        html.push(
          "<div><strong>Final</strong> | status " +
            badgeHtml(final.status || "unknown") +
            " | rounds " +
            escapeHtml(final.rounds == null ? "-" : final.rounds) +
            " | source " +
            escapeHtml(final.source || "unknown") +
            "</div>"
        );
        html.push('<div style="height:8px;"></div>');

        if (!rounds.length) {
          html.push("<div>No per-round result found.</div>");
        } else {
          for (var i = 0; i < rounds.length; i += 1) {
            var r = rounds[i];
            var title = "Round " + r.round_index + " | metric " + (r.metric || "None") + " | job " + (r.job_name || "-");

            html.push('<details class="round-item" ' + (i === rounds.length - 1 ? "open" : "") + ">");
            html.push("<summary>" + escapeHtml(title) + " " + badgeHtml(r.k8s_status || "unknown") + "</summary>");
            html.push(
              '<div class="meta">namespace=' +
                escapeHtml(r.namespace || "default") +
                " | manifest_path=" +
                escapeHtml(r.manifest_path || "-") +
                "</div>"
            );

            if (r.submit_script) {
              html.push('<details class="round-item">');
              html.push("<summary>Submission Script (Manifest YAML)</summary>");
              html.push(preHtml(r.submit_script));
              if (r.submit_script_truncated) {
                html.push('<div class="warn">manifest is truncated in UI</div>');
              }
              html.push("</details>");
            } else {
              html.push('<div class="meta">No submission script captured for this round.</div>');
            }

            if (r.k8s_log_tail) {
              html.push('<details class="round-item">');
              html.push("<summary>K8S Log</summary>");
              html.push(preHtml(r.k8s_log_tail));
              html.push("</details>");
            } else {
              html.push('<div class="meta">No K8S log captured for this round.</div>');
            }

            html.push("</details>");
          }
        }

        if (el.runResult) el.runResult.innerHTML = html.join("");
        done && done(null);
      },
      function (err) {
        done && done(err);
      }
    );
  }

  function setAuto(flag) {
    state.auto = !!flag;

    if (el.autoBtn) {
      el.autoBtn.textContent = state.auto ? "Auto ON" : "Auto OFF";
      el.autoBtn.className = state.auto ? "primary" : "";
    }

    if (state.timer) {
      clearInterval(state.timer);
      state.timer = null;
    }

    if (state.auto) {
      state.timer = setInterval(function () {
        pullUpdates(function (err) {
          if (err) setWarn(String(err));
        });
      }, 2200);
    }
  }

  function wireEvents() {
    if (el.runRefresh) {
      el.runRefresh.onclick = function () {
        loadRuns(true, function (err) {
          if (err) {
            setWarn(String(err));
            return;
          }
          switchRun(state.runId, function (err2) {
            if (err2) setWarn(String(err2));
          });
        });
      };
    }

    if (el.runSel) {
      el.runSel.onchange = function () {
        switchRun(el.runSel.value, function (err) {
          if (err) setWarn(String(err));
        });
      };
    }

    if (el.pullBtn) {
      el.pullBtn.onclick = function () {
        pullUpdates(function (err) {
          if (err) {
            setWarn(String(err));
            return;
          }
          loadRunResults(function (err2) {
            if (err2) setWarn(String(err2));
          });
        });
      };
    }

    if (el.autoBtn) {
      el.autoBtn.onclick = function () {
        setAuto(!state.auto);
      };
    }

    if (el.debugRefresh) {
      el.debugRefresh.onclick = function () {
        loadDebug(function (err) {
          if (err) setWarn(String(err));
        });
      };
    }

    if (el.debugExpSel) {
      el.debugExpSel.onchange = function () {
        loadDebug(function (err) {
          if (err) setWarn(String(err));
        });
      };
    }

    if (el.podFetch) {
      el.podFetch.onclick = function () {
        fetchPodLogs(function (err) {
          if (err) setWarn(String(err));
        });
      };
    }
  }

  function init() {
    if (el.meta) el.meta.textContent = "bootstrapping monitor...";

    wireEvents();

    loadRuns(false, function (err) {
      if (err) {
        setWarn(String(err));
        return;
      }
      switchRun(state.runId, function (err2) {
        if (err2) {
          setWarn(String(err2));
          return;
        }
        setAuto(true);
      });
    });
  }

  init();
})();
