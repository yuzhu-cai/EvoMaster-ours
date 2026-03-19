/**
 * MagiClaw Web Chat — Client Application
 *
 * SocketIO-based chat client with markdown rendering, progress tracking,
 * question handling, and background task updates.
 */
(function () {
  "use strict";

  // ───── State ─────
  let sessionId = crypto.randomUUID();
  let socket = null;
  let messageCards = {}; // message_id -> DOM element
  let isUserScrolled = false;

  // ───── DOM References ─────
  const sidebar = document.getElementById("sidebar");
  const messageList = document.getElementById("message-list");
  const messageInput = document.getElementById("message-input");
  const btnSend = document.getElementById("btn-send");
  const btnNewSession = document.getElementById("btn-new-session");
  const btnListAgents = document.getElementById("btn-list-agents");
  const agentInfo = document.getElementById("agent-info");
  const sessionStatusText = document.getElementById("session-status-text");
  const sessionIdDisplay = document.getElementById("session-id-display");
  const sidebarToggleOpen = document.getElementById("sidebar-toggle-open");
  const sidebarToggleClose = document.getElementById("sidebar-toggle-close");
  const welcomeHero = document.getElementById("welcome-hero");

  // ───── Markdown Setup ─────
  marked.setOptions({
    breaks: true,
    gfm: true,
  });

  // Use marked extension for code highlighting (marked v12+ API)
  var renderer = new marked.Renderer();
  renderer.code = function (token) {
    var code = token.text || "";
    var lang = (token.lang || "").trim();
    var highlighted;
    if (lang && hljs.getLanguage(lang)) {
      try {
        highlighted = hljs.highlight(code, { language: lang }).value;
      } catch (_) {
        highlighted = hljs.highlightAuto(code).value;
      }
    } else {
      highlighted = hljs.highlightAuto(code).value;
    }
    var langAttr = lang ? ' data-lang="' + lang + '"' : '';
    return '<pre' + langAttr + '><code class="hljs">' + highlighted + "</code></pre>";
  };
  marked.use({ renderer: renderer });

  // ───── Helpers ─────

  function renderMarkdown(text) {
    if (!text) return "";
    return marked.parse(text);
  }

  function escapeHtml(text) {
    var div = document.createElement("div");
    div.appendChild(document.createTextNode(text));
    return div.innerHTML;
  }

  function formatTime() {
    return new Date().toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
  }

  function scrollToBottom() {
    if (!isUserScrolled) {
      messageList.scrollTop = messageList.scrollHeight;
    }
  }

  function autoResizeTextarea() {
    messageInput.style.height = "auto";
    messageInput.style.height =
      Math.min(messageInput.scrollHeight, 200) + "px";
  }

  function hideWelcomeHero() {
    if (welcomeHero && !welcomeHero.classList.contains("hidden")) {
      welcomeHero.classList.add("hidden");
    }
  }

  // ───── Typing Indicator ─────
  var typingEl = null;

  function showTypingIndicator() {
    if (typingEl) return;
    typingEl = document.createElement("div");
    typingEl.className = "typing-indicator";
    typingEl.innerHTML =
      '<span class="typing-indicator-label">Agent is thinking</span>' +
      '<div class="typing-dots"><span></span><span></span><span></span></div>';
    messageList.appendChild(typingEl);
    scrollToBottom();
  }

  function hideTypingIndicator() {
    if (typingEl) {
      typingEl.remove();
      typingEl = null;
    }
  }

  // ───── Sidebar (mobile) ─────

  var overlay = document.createElement("div");
  overlay.className = "sidebar-overlay";
  document.body.appendChild(overlay);

  function openSidebar() {
    sidebar.classList.add("open");
    overlay.classList.add("visible");
  }

  function closeSidebar() {
    sidebar.classList.remove("open");
    overlay.classList.remove("visible");
  }

  sidebarToggleOpen.addEventListener("click", openSidebar);
  sidebarToggleClose.addEventListener("click", closeSidebar);
  overlay.addEventListener("click", closeSidebar);

  // ───── Scroll Detection ─────

  messageList.addEventListener("scroll", function () {
    var threshold = 60;
    var distanceFromBottom =
      messageList.scrollHeight -
      messageList.scrollTop -
      messageList.clientHeight;
    isUserScrolled = distanceFromBottom > threshold;
  });

  // ───── Message Creation ─────

  function addUserMessage(text) {
    hideWelcomeHero();
    var wrapper = document.createElement("div");
    wrapper.className = "message message-user";

    var bubble = document.createElement("div");
    bubble.className = "message-bubble";

    var content = document.createElement("div");
    content.textContent = text;
    bubble.appendChild(content);

    var time = document.createElement("div");
    time.className = "message-time";
    time.textContent = formatTime();
    bubble.appendChild(time);

    wrapper.appendChild(bubble);
    messageList.appendChild(wrapper);
    scrollToBottom();
  }

  function addSystemMessage(text) {
    hideWelcomeHero();
    var wrapper = document.createElement("div");
    wrapper.className = "message message-system";

    var bubble = document.createElement("div");
    bubble.className = "message-bubble";

    var content = document.createElement("div");
    content.className = "markdown-content";
    content.innerHTML = renderMarkdown(text);
    bubble.appendChild(content);

    wrapper.appendChild(bubble);
    messageList.appendChild(wrapper);
    scrollToBottom();
  }

  function getOrCreateAgentCard(messageId) {
    if (messageCards[messageId]) {
      return messageCards[messageId];
    }

    var card = document.createElement("div");
    card.className = "agent-card";
    card.setAttribute("data-status", "loading");
    card.setAttribute("data-message-id", messageId);

    card.innerHTML =
      '<div class="agent-card-header">' +
      '  <span class="agent-card-status-icon">&#9679;</span>' +
      '  <span class="agent-card-label">Processing...</span>' +
      "</div>" +
      '<div class="progress-section">' +
      '  <div class="progress-bar-track">' +
      '    <div class="progress-bar-fill"></div>' +
      "  </div>" +
      '  <div class="progress-info">' +
      '    <span class="progress-step-text"></span>' +
      '    <span class="progress-tool-name"></span>' +
      "  </div>" +
      '  <div class="progress-todo"></div>' +
      "</div>" +
      '<div class="agent-card-body">' +
      '  <div class="markdown-content"></div>' +
      "</div>" +
      '<div class="question-section">' +
      '  <div class="question-text markdown-content"></div>' +
      '  <div class="option-buttons"></div>' +
      '  <div class="freeform-answer">' +
      '    <input type="text" class="freeform-input" placeholder="Type your answer...">' +
      '    <button class="freeform-submit">Send</button>' +
      "  </div>" +
      "</div>" +
      '<div class="action-buttons"></div>';

    messageList.appendChild(card);
    messageCards[messageId] = card;
    scrollToBottom();
    return card;
  }

  // ───── Card State Helpers ─────

  function setCardStatus(card, status) {
    card.setAttribute("data-status", status);
    var icon = card.querySelector(".agent-card-status-icon");
    var label = card.querySelector(".agent-card-label");

    switch (status) {
      case "loading":
        icon.innerHTML = "&#9679;";
        label.textContent = "Processing...";
        break;
      case "progress":
        icon.innerHTML = "&#9679;";
        label.textContent = "Working...";
        break;
      case "completed":
        icon.innerHTML = "&#10003;";
        label.textContent = "Completed";
        break;
      case "failed":
        icon.innerHTML = "&#10007;";
        label.textContent = "Failed";
        break;
      case "question":
        icon.innerHTML = "?";
        label.textContent = "Waiting for input";
        break;
    }
  }

  function showCardBody(card, html) {
    var body = card.querySelector(".agent-card-body");
    body.querySelector(".markdown-content").innerHTML = html;
    body.classList.add("visible");
  }

  function hideProgress(card) {
    var section = card.querySelector(".progress-section");
    section.style.display = "none";
  }

  // ───── SocketIO Connection ─────

  function connect() {
    socket = io({ transports: ["websocket", "polling"] });

    socket.on("connect", function () {
      updateConnectionStatus(true);
      socket.emit("join", { session_id: sessionId });
    });

    socket.on("disconnect", function () {
      updateConnectionStatus(false);
    });

    socket.on("welcome", function (data) {
      addSystemMessage(data.message || "Connected to MagiClaw.");
    });

    socket.on("session_info", function (data) {
      if (data.agent) {
        agentInfo.innerHTML =
          '<span class="agent-name">' + escapeHtml(data.agent) + "</span>";
      }
      if (data.session_id) {
        sessionIdDisplay.textContent = data.session_id;
      }
    });

    socket.on("message_ack", function (data) {
      var msgId = data.message_id;
      if (msgId) {
        hideTypingIndicator();
        getOrCreateAgentCard(msgId);
      }
    });

    socket.on("step_progress", function (data) {
      var msgId = data.message_id;
      if (!msgId) return;
      var card = getOrCreateAgentCard(msgId);
      setCardStatus(card, "progress");

      var step = data.step || 0;
      var maxSteps = data.max_steps || 1;
      var pct = Math.min(Math.round((step / maxSteps) * 100), 100);

      var fill = card.querySelector(".progress-bar-fill");
      fill.style.width = pct + "%";

      var stepText = card.querySelector(".progress-step-text");
      stepText.textContent = data.step_text || "Step " + step + "/" + maxSteps;

      var toolName = card.querySelector(".progress-tool-name");
      if (data.tool_name) {
        toolName.textContent = data.tool_name;
      } else {
        toolName.textContent = "";
      }

      var todoEl = card.querySelector(".progress-todo");
      if (data.todo) {
        todoEl.textContent = data.todo;
      } else {
        todoEl.textContent = "";
      }

      scrollToBottom();
    });

    socket.on("agent_response", function (data) {
      var msgId = data.message_id;
      if (!msgId) return;
      hideTypingIndicator();
      var card = getOrCreateAgentCard(msgId);

      hideProgress(card);

      var status = data.status === "error" ? "failed" : "completed";
      setCardStatus(card, status);

      var html = renderMarkdown(data.answer || data.text || "");
      showCardBody(card, html);

      // Action buttons (confirm/cancel)
      if (data.actions && data.actions.length > 0) {
        renderActionButtons(card, msgId, data.actions);
      }

      scrollToBottom();
    });

    socket.on("question", function (data) {
      var msgId = data.message_id;
      if (!msgId) return;
      var card = getOrCreateAgentCard(msgId);

      hideProgress(card);
      setCardStatus(card, "question");

      // Store session metadata for sending back with the answer
      card.dataset.sessionKey = data.session_key || "";
      card.dataset.agentName = data.agent_name || "";

      var section = card.querySelector(".question-section");
      var textEl = section.querySelector(".question-text");
      textEl.innerHTML = renderMarkdown(data.question || data.text || "");
      section.classList.add("visible");

      // Render option buttons if provided
      var optContainer = section.querySelector(".option-buttons");
      optContainer.innerHTML = "";
      if (data.options && data.options.length > 0) {
        data.options.forEach(function (opt) {
          var isString = typeof opt === "string";
          var label = isString ? opt : opt.label || opt.text || String(opt);
          var value = isString ? opt : opt.value || opt.label || String(opt);
          var btn = document.createElement("button");
          btn.className = "option-btn";
          btn.textContent = label;
          btn.addEventListener("click", function () {
            handleOptionClick(card, msgId, btn, value);
          });
          optContainer.appendChild(btn);
        });
      }

      // Free-form input
      var freeformInput = section.querySelector(".freeform-input");
      var freeformSubmit = section.querySelector(".freeform-submit");
      freeformInput.value = "";
      freeformInput.disabled = false;
      freeformSubmit.disabled = false;

      freeformSubmit.onclick = function () {
        var val = freeformInput.value.trim();
        if (!val) return;
        submitAnswer(card, msgId, val);
      };

      freeformInput.onkeydown = function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          var val = freeformInput.value.trim();
          if (!val) return;
          submitAnswer(card, msgId, val);
        }
      };

      scrollToBottom();
    });

    socket.on("delegation_status", function (data) {
      var text = data.message || data.text || "Task delegated.";
      addSystemMessage(text);
    });

    socket.on("background_task_update", function (data) {
      handleBackgroundTaskUpdate(data);
    });

    socket.on("agent_list", function (data) {
      var agents = data.agents || [];
      if (agents.length === 0) {
        addSystemMessage("No agents available.");
        return;
      }
      var lines = ["**Available Agents:**\n"];
      agents.forEach(function (a) {
        var name = typeof a === "string" ? a : a.name || String(a);
        var desc = typeof a === "object" && a.description ? " -- " + a.description : "";
        lines.push("- `" + name + "`" + desc);
      });
      addSystemMessage(lines.join("\n"));
    });

    socket.on("error", function (data) {
      var msg = data.message || data.error || "An error occurred.";
      addSystemMessage("**Error:** " + msg);
    });
  }

  // ───── Question / Action Handlers ─────

  function disableQuestionInputs(card) {
    card.querySelectorAll(".option-btn").forEach(function (b) {
      b.disabled = true;
    });
    var freeformInput = card.querySelector(".freeform-input");
    var freeformSubmit = card.querySelector(".freeform-submit");
    if (freeformInput) freeformInput.disabled = true;
    if (freeformSubmit) freeformSubmit.disabled = true;
  }

  function handleOptionClick(card, messageId, clickedBtn, value) {
    disableQuestionInputs(card);
    clickedBtn.classList.add("selected");
    socket.emit("answer_question", {
      session_id: sessionId,
      message_id: messageId,
      answer: value,
      session_key: card.dataset.sessionKey || "",
      agent_name: card.dataset.agentName || "",
    });
  }

  function submitAnswer(card, messageId, value) {
    disableQuestionInputs(card);
    socket.emit("answer_question", {
      session_id: sessionId,
      message_id: messageId,
      answer: value,
      session_key: card.dataset.sessionKey || "",
      agent_name: card.dataset.agentName || "",
    });
  }

  function renderActionButtons(card, messageId, actions) {
    var container = card.querySelector(".action-buttons");
    container.innerHTML = "";
    container.classList.add("visible");

    actions.forEach(function (action) {
      var btn = document.createElement("button");
      btn.className = "action-btn";
      if (action.type === "confirm" || action.action === "confirm") {
        btn.classList.add("action-btn-confirm");
        btn.textContent = action.label || "Confirm";
      } else {
        btn.classList.add("action-btn-cancel");
        btn.textContent = action.label || "Cancel";
      }

      btn.addEventListener("click", function () {
        var allActionBtns = container.querySelectorAll(".action-btn");
        allActionBtns.forEach(function (b) {
          b.disabled = true;
        });
        socket.emit("card_action", {
          session_id: sessionId,
          message_id: messageId,
          action: action.action || action.type || action.label,
          session_key: action.session_key || "",
          agent_name: action.agent_name || "",
          original_answer: action.original_answer || "",
        });
      });

      container.appendChild(btn);
    });
  }

  // ───── Background Tasks ─────

  var backgroundTasks = {}; // task_id -> DOM element

  function handleBackgroundTaskUpdate(data) {
    var taskId = data.task_id;
    if (!taskId) return;

    var card = backgroundTasks[taskId];
    if (!card) {
      card = document.createElement("div");
      card.className = "background-task-card";
      card.innerHTML =
        '<div class="task-header">' +
        '  <span>&#9881;</span>' +
        '  <span class="task-status"></span>' +
        "</div>" +
        '<div class="task-body"></div>';
      messageList.appendChild(card);
      backgroundTasks[taskId] = card;
    }

    var statusEl = card.querySelector(".task-status");
    statusEl.textContent = data.status || "running";

    var bodyEl = card.querySelector(".task-body");
    if (data.message || data.text) {
      bodyEl.innerHTML = renderMarkdown(data.message || data.text);
    }

    scrollToBottom();
  }

  // ───── Connection Status ─────

  function updateConnectionStatus(connected) {
    var indicator = document.querySelector(".status-indicator");
    if (connected) {
      indicator.className = "status-indicator status-connected";
      sessionStatusText.textContent = "Connected";
    } else {
      indicator.className = "status-indicator status-disconnected";
      sessionStatusText.textContent = "Disconnected";
    }
  }

  // ───── Send Message ─────

  function sendMessage() {
    var text = messageInput.value.trim();
    if (!text || !socket || !socket.connected) return;

    addUserMessage(text);
    showTypingIndicator();
    socket.emit("send_message", {
      session_id: sessionId,
      text: text,
    });

    messageInput.value = "";
    autoResizeTextarea();
    messageInput.focus();
  }

  // ───── UI Event Bindings ─────

  btnSend.addEventListener("click", sendMessage);

  messageInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  messageInput.addEventListener("input", autoResizeTextarea);

  // ───── Session ID Copy ─────
  sessionIdDisplay.addEventListener("click", function () {
    var text = sessionIdDisplay.textContent;
    if (!text || text === "--") return;
    navigator.clipboard.writeText(text).then(function () {
      sessionIdDisplay.classList.add("copied");
      setTimeout(function () {
        sessionIdDisplay.classList.remove("copied");
      }, 1500);
    });
  });

  btnNewSession.addEventListener("click", function () {
    if (!confirm("Start a new session? Current conversation will be cleared.")) {
      return;
    }
    sessionId = crypto.randomUUID();
    messageList.innerHTML = "";
    messageCards = {};
    backgroundTasks = {};
    isUserScrolled = false;
    sessionIdDisplay.textContent = sessionId;

    // Restore welcome hero
    var hero = document.createElement("div");
    hero.id = "welcome-hero";
    hero.className = "welcome-hero";
    hero.innerHTML =
      '<div class="welcome-icon">' +
      '  <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor"' +
      '       stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">' +
      '    <path d="M12 2L9 9l-7 2 5 5-1 7 6-3 6 3-1-7 5-5-7-2z"/>' +
      '  </svg>' +
      '</div>' +
      '<h2 class="welcome-title">MagiClaw</h2>' +
      '<p class="welcome-subtitle">Start a new session and ask anything</p>';
    messageList.appendChild(hero);

    if (socket && socket.connected) {
      socket.emit("new_session", { session_id: sessionId });
    }
  });

  btnListAgents.addEventListener("click", function () {
    if (!socket || !socket.connected) return;
    socket.emit("send_message", {
      session_id: sessionId,
      text: "/list",
    });
  });

  // ───── Init ─────

  sessionIdDisplay.textContent = sessionId;
  connect();
})();
