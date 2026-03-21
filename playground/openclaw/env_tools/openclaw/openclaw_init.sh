#!/bin/bash
# OpenClaw initialization script (upload-only).

set -euo pipefail

# Some containers do not define HOME; guard for nounset mode.
export HOME="${HOME:-/tmp}"

if [[ -f "$HOME/.bashrc" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$HOME/.bashrc" || true
  set -u
fi

export IS_SANDBOX=1

RUNTIME_BUNDLE="/tmp/openclaw-runtime-bundle.tar.gz"
STAGE_DIR="/tmp/openclaw-stage"
NODE_DIR="/tmp/node"
OPENCLAW_PREFIX="/tmp/openclaw-npm-prefix"
OPENCLAW_WRAPPER="/tmp/openclaw_wrapper"

if [[ ! -f "$RUNTIME_BUNDLE" || ! -s "$RUNTIME_BUNDLE" ]]; then
  echo "ERROR: missing openclaw runtime bundle: ${RUNTIME_BUNDLE}. Only the fixed /tmp upload is allowed." >&2
  exit 65
fi

extract_archive() {
  local archive="$1"
  local target_dir="$2"
  mkdir -p "$target_dir"
  case "$archive" in
    *.tar.gz|*.tgz) tar -xzf "$archive" -C "$target_dir" ;;
    *.tar.xz|*.txz|*.xz) tar -xJf "$archive" -C "$target_dir" ;;
    *) tar -xf "$archive" -C "$target_dir" ;;
  esac
}

rm -rf "$STAGE_DIR" "$OPENCLAW_PREFIX"
rm -f "$OPENCLAW_WRAPPER"
mkdir -p "$STAGE_DIR" "$OPENCLAW_PREFIX"
extract_archive "$RUNTIME_BUNDLE" "$STAGE_DIR"

node_bin="$(find "$STAGE_DIR" -maxdepth 12 -type f -path '*/bin/node' | head -n 1 || true)"
if [[ -z "$node_bin" ]]; then
  echo "ERROR: node runtime not found in ${RUNTIME_BUNDLE}" >&2
  exit 65
fi
node_root="$(dirname "$(dirname "$node_bin")")"
rm -rf "$NODE_DIR"
mkdir -p "$NODE_DIR"
cp -a "$node_root"/. "$NODE_DIR"/

if [[ ! -x "$NODE_DIR/bin/node" ]]; then
  echo "ERROR: failed to prepare /tmp/node from runtime bundle" >&2
  exit 65
fi

major="$("$NODE_DIR/bin/node" -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)"
if [[ "$major" -lt 20 ]]; then
  echo "ERROR: node >=20 required for openclaw, got $("$NODE_DIR/bin/node" -v)" >&2
  exit 65
fi

openclaw_bin="$(
  find "$STAGE_DIR" -maxdepth 12 -path '*/bin/openclaw' \( -type f -o -type l \) | head -n 1 || true
)"
if [[ -z "$openclaw_bin" ]]; then
  echo "ERROR: openclaw executable not found in ${RUNTIME_BUNDLE}" >&2
  exit 65
fi

openclaw_root="$(dirname "$(dirname "$openclaw_bin")")"
cp -a "$openclaw_root"/. "$OPENCLAW_PREFIX"/

mkdir -p "$OPENCLAW_PREFIX/bin"
openclaw_entry="$OPENCLAW_PREFIX/bin/openclaw"
if [[ ! -e "$openclaw_entry" ]]; then
  echo "ERROR: invalid openclaw runtime bundle, missing ${openclaw_entry}" >&2
  exit 65
fi

openclaw_target="$(readlink -f "$openclaw_entry" 2>/dev/null || true)"
if [[ -z "$openclaw_target" ]]; then
  openclaw_target="$openclaw_entry"
fi
if [[ ! -f "$openclaw_target" ]]; then
  echo "ERROR: openclaw target not found: ${openclaw_target}" >&2
  exit 65
fi

chmod +x "$openclaw_entry" "$openclaw_target" 2>/dev/null || true

openclaw_exec_mode="direct"
if [[ "$openclaw_target" == *.js || "$openclaw_target" == *.mjs ]]; then
  openclaw_exec_mode="node"
else
  first_line="$(head -n 1 "$openclaw_target" 2>/dev/null || true)"
  if [[ "$first_line" == *"node"* ]]; then
    openclaw_exec_mode="node"
  fi
fi

cat > "$OPENCLAW_WRAPPER" <<SH
#!/bin/bash
set -euo pipefail
if [[ "${openclaw_exec_mode}" == "node" ]]; then
  exec "$NODE_DIR/bin/node" "${openclaw_target}" "\$@"
fi
exec "${openclaw_entry}" "\$@"
SH
chmod +x "$OPENCLAW_WRAPPER"

"$OPENCLAW_WRAPPER" --version || true
