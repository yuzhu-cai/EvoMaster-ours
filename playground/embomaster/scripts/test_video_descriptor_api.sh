#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-https://api.gpugeek.com/v1}"
API_KEY="${API_KEY:-u0rgjezq53e5tv01000dg95v0yqn2eecv02b7z3r}"
MODEL="${MODEL:-Vendor2/gemini-3-flash}"
TEMPERATURE="${TEMPERATURE:-0.2}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
MAX_FRAMES="${MAX_FRAMES:-12}"
INPUT_MODE="${INPUT_MODE:-auto}"
PROMPT_TEXT="${PROMPT_TEXT:-请分析这个机器人操作视频，并用中文输出：1) 视频内容概述 2) 关键问题诊断 3) 可执行改进建议（至少3条）}"

usage() {
  cat <<'USAGE'
Usage:
  bash playground/embomaster/scripts/test_video_descriptor_api.sh /abs/path/to/video.mp4
  bash playground/embomaster/scripts/test_video_descriptor_api.sh --text

Optional environment variables:
  BASE_URL      Default: https://api.gpugeek.com/v1
  API_KEY       Default: config value
  MODEL         Default: Vendor2/gemini-3-flash
  TEMPERATURE   Default: 0.2
  MAX_TOKENS    Default: 1024
  MAX_FRAMES    Default: 12
  INPUT_MODE    auto | video | frames   Default: auto
  PROMPT_TEXT   Override the Chinese analysis prompt
USAGE
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" >&2
    exit 1
  fi
}

resolve_input_mode() {
  local mode="$1"
  if [[ "$mode" != "auto" ]]; then
    printf '%s\n' "$mode"
    return
  fi
  case "$BASE_URL" in
    *127.0.0.1*|*localhost*|*0.0.0.0*)
      printf 'video\n'
      ;;
    *)
      printf 'frames\n'
      ;;
  esac
}

b64_file() {
  local path="$1"
  if base64 --help 2>/dev/null | grep -q -- '-w'; then
    base64 -w 0 "$path"
  else
    base64 "$path" | tr -d '\n'
  fi
}

build_payload() {
  local kind="$1"
  local media_list_file="$2"
  KIND="$kind" \
  MEDIA_LIST_FILE="$media_list_file" \
  MODEL="$MODEL" \
  TEMPERATURE="$TEMPERATURE" \
  MAX_TOKENS="$MAX_TOKENS" \
  PROMPT_TEXT="$PROMPT_TEXT" \
  python3 - <<'PY'
import base64
import json
import mimetypes
import os
from pathlib import Path

kind = os.environ['KIND']
media_list_file = Path(os.environ['MEDIA_LIST_FILE'])
model = os.environ['MODEL']
temperature = float(os.environ['TEMPERATURE'])
max_tokens = int(os.environ['MAX_TOKENS'])
prompt_text = os.environ['PROMPT_TEXT']

user_content = []
for raw in media_list_file.read_text().splitlines():
    raw = raw.strip()
    if not raw:
        continue
    path = Path(raw)
    mime_type, _ = mimetypes.guess_type(str(path))
    if kind == 'video':
        if not mime_type:
            mime_type = 'video/mp4'
        media_type = 'video_url'
        key = 'video_url'
    else:
        if not mime_type:
            mime_type = 'image/jpeg'
        media_type = 'image_url'
        key = 'image_url'
    encoded = base64.b64encode(path.read_bytes()).decode('utf-8')
    user_content.append({
        'type': media_type,
        key: {'url': f'data:{mime_type};base64,{encoded}'},
    })

user_content.append({
    'type': 'text',
    'text': prompt_text,
})

payload = {
    'model': model,
    'messages': [
        {
            'role': 'system',
            'content': 'You are a robotics evaluation assistant. Given sampled frames from one robot-operation video, describe behavior and give practical feedback.',
        },
        {
            'role': 'user',
            'content': user_content,
        },
    ],
    'temperature': temperature,
    'max_tokens': max_tokens,
}
print(json.dumps(payload, ensure_ascii=False))
PY
}

extract_frames() {
  local video_path="$1"
  local out_dir="$2"
  require_cmd ffmpeg
  require_cmd ffprobe
  VIDEO_PATH="$video_path" MAX_FRAMES="$MAX_FRAMES" OUT_DIR="$out_dir" python3 - <<'PY'
import json
import math
import os
import subprocess
from pathlib import Path

video_path = Path(os.environ['VIDEO_PATH'])
out_dir = Path(os.environ['OUT_DIR'])
max_frames = max(1, int(os.environ['MAX_FRAMES']))

probe = subprocess.run(
    ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', str(video_path)],
    capture_output=True,
    text=True,
    check=True,
)
duration = float(json.loads(probe.stdout or '{}').get('format', {}).get('duration') or 0.0)
if duration <= 0:
    raise SystemExit(f'Could not determine duration for {video_path}')

step = duration / float(max_frames + 1)
for idx in range(max_frames):
    ts = max(0.0, min(duration, step * (idx + 1)))
    frame_path = out_dir / f'frame_{idx:02d}.jpg'
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
        '-ss', f'{ts:.3f}', '-i', str(video_path),
        '-frames:v', '1', str(frame_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f'ffmpeg failed at {ts:.3f}s: {proc.stderr.strip() or proc.stdout.strip()}')
    if frame_path.exists() and frame_path.stat().st_size > 0:
        print(frame_path)
PY
}

if [[ "${1:-}" == "" ]]; then
  usage
  exit 1
fi

if [[ "${1:-}" == "--text" ]]; then
  curl -sS "${BASE_URL}/chat/completions" \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer ${API_KEY}" \
    -d "{
      \"model\": \"${MODEL}\",
      \"messages\": [
        {\"role\": \"user\", \"content\": \"请只回复：调用成功\"}
      ],
      \"temperature\": ${TEMPERATURE},
      \"max_tokens\": 32
    }"
  exit 0
fi

VIDEO_PATH="$1"
if [[ ! -f "${VIDEO_PATH}" ]]; then
  echo "Video not found: ${VIDEO_PATH}" >&2
  exit 1
fi

MODE="$(resolve_input_mode "${INPUT_MODE}")"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
MEDIA_LIST_FILE="$TMP_DIR/media_paths.txt"
PAYLOAD_FILE="$TMP_DIR/payload.json"

case "$MODE" in
  video)
    printf '%s\n' "$VIDEO_PATH" > "$MEDIA_LIST_FILE"
    build_payload video "$MEDIA_LIST_FILE" > "$PAYLOAD_FILE"
    ;;
  frames)
    extract_frames "$VIDEO_PATH" "$TMP_DIR" > "$MEDIA_LIST_FILE"
    if [[ ! -s "$MEDIA_LIST_FILE" ]]; then
      echo "No frames extracted from ${VIDEO_PATH}" >&2
      exit 1
    fi
    build_payload frames "$MEDIA_LIST_FILE" > "$PAYLOAD_FILE"
    ;;
  *)
    echo "Unsupported INPUT_MODE: $MODE" >&2
    exit 1
    ;;
esac

echo "[video-test] model=${MODEL} input_mode=${MODE} base_url=${BASE_URL}" >&2
if [[ "$MODE" == "frames" ]]; then
  echo "[video-test] frame_count=$(wc -l < "$MEDIA_LIST_FILE" | tr -d ' ')" >&2
fi

curl -sS "${BASE_URL}/chat/completions" \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${API_KEY}" \
  --data-binary @"${PAYLOAD_FILE}"
