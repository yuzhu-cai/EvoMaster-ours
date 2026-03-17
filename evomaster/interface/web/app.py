"""EvoMaster Web Chat Interface

Flask + SocketIO server that provides a web-based chat interface
for interacting with EvoMaster agents.
"""
from __future__ import annotations

import logging
import re
import threading
import uuid
from pathlib import Path

from flask import Flask, request, send_from_directory
from flask_socketio import SocketIO, emit, join_room

logger = logging.getLogger(__name__)

# Pre-compiled /agent <name> <task> command regex
_AGENT_CMD_RE = re.compile(r"^/agent\s+(\S+)(?:\s+(.+))?$", re.DOTALL)


class WebApp:
    """Web chat application server.

    Manages Flask + SocketIO server lifecycle and routes
    incoming WebSocket events to the WebTaskDispatcher.
    """

    def __init__(self, config, project_root: Path):
        """
        Args:
            config: WebAppConfig instance
            project_root: Project root directory
        """
        self._config = config
        self._project_root = project_root

        # Flask app
        static_dir = Path(__file__).parent / "static"
        self._flask = Flask(__name__, static_folder=str(static_dir))

        # SocketIO
        self._socketio = SocketIO(
            self._flask,
            async_mode="threading",
            cors_allowed_origins="*",
        )

        # Room tracking: session_id -> room name
        self._rooms: dict[str, str] = {}
        self._rooms_lock = threading.Lock()

        # Container pool (optional)
        container_pool = None
        if config.container_pool and config.container_pool.get("enabled"):
            from evomaster.env.container_pool import ContainerPool, ContainerPoolConfig

            pool_cfg = ContainerPoolConfig(
                **{k: v for k, v in config.container_pool.items() if k != "enabled"}
            )
            container_pool = ContainerPool(pool_cfg)
            container_pool.start()
        self._container_pool = container_pool

        # Dispatcher
        from evomaster.interface.web.dispatcher import WebTaskDispatcher

        self._dispatcher = WebTaskDispatcher(
            project_root=project_root,
            socketio=self._socketio,
            default_agent=config.default_agent,
            default_config_path=config.default_config_path,
            max_workers=config.max_concurrent_tasks,
            task_timeout=config.task_timeout,
            max_sessions=config.max_sessions,
            available_agents=config.available_agents,
            container_pool=container_pool,
        )

        # Register routes and events
        self._register_routes()
        self._register_events()

    def _get_room(self, session_id: str) -> str:
        """Look up the room name for a session, falling back to the default."""
        return self._rooms.get(session_id, f"session:{session_id}")

    def _register_routes(self):
        """Register HTTP routes."""

        @self._flask.route("/")
        def index():
            return send_from_directory(self._flask.static_folder, "index.html")

        @self._flask.route("/api/health")
        def health():
            return {"status": "ok"}

    def _register_events(self):
        """Register SocketIO event handlers."""

        @self._socketio.on("connect")
        def on_connect():
            sid = getattr(request, "sid", "unknown")
            logger.info("Client connected: %s", sid)

        @self._socketio.on("disconnect")
        def on_disconnect():
            logger.info("Client disconnected")

        @self._socketio.on("join")
        def on_join(data):
            """Join or create a session.

            data: {session_id: str, agent?: str}
            """
            try:
                session_id = data.get("session_id")
                if not session_id:
                    emit("error", {"error": "session_id is required"})
                    return

                room = f"session:{session_id}"
                join_room(room)

                with self._rooms_lock:
                    self._rooms[session_id] = room

                agent = data.get("agent", self._config.default_agent)
                session = self._dispatcher._session_manager.get(session_id)

                emit("session_info", {
                    "session_id": session_id,
                    "agent": agent,
                    "message_count": session.message_count if session else 0,
                    "initialized": session.initialized if session else False,
                })

                if not session:
                    self._dispatcher._send_welcome(room)
            except Exception:
                logger.exception("Error handling join event")
                emit("error", {"error": "Failed to join session"})

        @self._socketio.on("send_message")
        def on_send_message(data):
            """Send a chat message.

            data: {session_id: str, text: str, agent?: str}
            """
            try:
                session_id = data.get("session_id")
                text = data.get("text", "").strip()
                if not session_id or not text:
                    emit("error", {"error": "session_id and text are required"})
                    return

                room = self._get_room(session_id)
                message_id = str(uuid.uuid4())

                # Parse /agent command
                agent_name = data.get("agent")
                agent_match = _AGENT_CMD_RE.match(text)
                if agent_match:
                    agent_name = agent_match.group(1)
                    task_text = agent_match.group(2) or ""
                else:
                    task_text = text

                # Slash commands (/new, /help, /list) are handled synchronously
                # by the dispatcher — no Processing card needed.
                # Normal messages get their message_ack from the step reporter
                # inside the dispatcher, so we don't emit one here either.

                # Dispatch
                self._dispatcher.dispatch(
                    session_id=session_id,
                    message_id=message_id,
                    task_text=task_text,
                    agent_name=agent_name,
                    room=room,
                )
            except Exception:
                logger.exception("Error handling send_message event")
                emit("error", {"error": "Failed to process message"})

        @self._socketio.on("answer_question")
        def on_answer_question(data):
            """Answer an ask_user question.

            data: {session_id, session_key, agent_name, answer_text}
            """
            try:
                session_id = data.get("session_id")
                room = self._get_room(session_id)

                self._dispatcher.dispatch_card_action(
                    session_id=session_id,
                    action="answer_question",
                    session_key=data.get("session_key", ""),
                    agent_name=data.get("agent_name", ""),
                    task_text=data.get("answer", ""),
                    room=room,
                )
            except Exception:
                logger.exception("Error handling answer_question event")
                emit("error", {"error": "Failed to process answer"})

        @self._socketio.on("card_action")
        def on_card_action(data):
            """Handle confirm/cancel actions.

            data: {session_id, action, session_key, agent_name, original_answer?}
            """
            try:
                session_id = data.get("session_id")
                room = self._get_room(session_id)

                self._dispatcher.dispatch_card_action(
                    session_id=session_id,
                    action=data.get("action", ""),
                    session_key=data.get("session_key", ""),
                    agent_name=data.get("agent_name", ""),
                    task_text=data.get("original_answer", ""),
                    room=room,
                    message_id=data.get("message_id", ""),
                )
            except Exception:
                logger.exception("Error handling card_action event")
                emit("error", {"error": "Failed to process action"})

        @self._socketio.on("new_session")
        def on_new_session(data):
            """Clear current session and join the new room.

            data: {session_id}
            """
            try:
                session_id = data.get("session_id")
                if not session_id:
                    emit("error", {"error": "session_id is required"})
                    return

                # Join the new session's room so the client receives events
                room = f"session:{session_id}"
                join_room(room)

                with self._rooms_lock:
                    self._rooms[session_id] = room

                message_id = str(uuid.uuid4())

                self._dispatcher.dispatch(
                    session_id=session_id,
                    message_id=message_id,
                    task_text="/new",
                    room=room,
                )
            except Exception:
                logger.exception("Error handling new_session event")
                emit("error", {"error": "Failed to create new session"})

    def start(self):
        """Start the web server (blocking)."""
        logger.info(
            "Starting EvoMaster Web Chat on %s:%s",
            self._config.host,
            self._config.port,
        )
        self._socketio.run(
            self._flask,
            host=self._config.host,
            port=self._config.port,
            debug=False,
            use_reloader=False,
            allow_unsafe_werkzeug=True,
        )

    def stop(self):
        """Shutdown the server and cleanup."""
        logger.info("Stopping EvoMaster Web Chat...")
        self._dispatcher.shutdown(wait=False)
        if self._container_pool:
            self._container_pool.shutdown()
