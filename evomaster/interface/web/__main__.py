"""CLI entry point.

python -m evomaster.interface.web [--config PATH] [--agent NAME] [--host HOST] [--port PORT]
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="EvoMaster Web Chat Interface",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config file path (default: configs/web/config.yaml)",
    )
    parser.add_argument(
        "--agent",
        default=None,
        help="Override default agent name",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Override host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override port to bind to",
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)

    # Determine project root
    # __main__.py lives at evomaster/interface/web/, project_root is 3 levels up
    project_root = Path(__file__).resolve().parent.parent.parent.parent

    # Load config
    config_path = (
        Path(args.config) if args.config else project_root / "configs" / "web" / "config.yaml"
    )

    from .config import load_web_config

    try:
        config = load_web_config(config_path, project_root=project_root)
    except FileNotFoundError as e:
        logger.error("Config file not found: %s", e)
        return 1

    # CLI overrides
    if args.agent:
        config.default_agent = args.agent
    if args.host:
        config.host = args.host
    if args.port is not None:
        config.port = args.port

    # Create the web app
    from .app import WebApp

    app = WebApp(config=config, project_root=project_root)

    # Signal handling
    def _shutdown(signum, _frame):
        logger.info("Received signal %s, shutting down...", signum)
        app.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Start (blocking)
    try:
        app.start()
    except KeyboardInterrupt:
        app.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
