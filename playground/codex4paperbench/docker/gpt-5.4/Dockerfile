ARG BASE_IMAGE=pb-env-codex:0.125.0
FROM ${BASE_IMAGE}

ARG CODEX_MODEL=gpt-5.4

RUN python3 -c 'from pathlib import Path; import os, re; p = Path("/root/.codex/config.toml"); model = os.environ.get("CODEX_MODEL", "gpt-5.4"); s = p.read_text(); s, n = re.subn(r"(?m)^model[ \t]*=[ \t]*\"[^\"]*\"", "model = " + chr(34) + model + chr(34), s, count=1); assert n == 1, "model entry not found in /root/.codex/config.toml"; p.write_text(s)'
