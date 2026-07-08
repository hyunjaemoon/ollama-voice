#!/bin/bash
# Run the ollama-voice serving dashboard:
#   - OpenAI-compatible inference API  (http://HOST:PORT/v1)
#   - live monitoring dashboard + backend router  (http://HOST:PORT/)
#
# For the voice agent instead, use ../run.sh (runs main.py).
#
# Config via env vars (all optional):
#   HOST=127.0.0.1  PORT=8000  BACKENDS=echo,ollama  OLLAMA_MODEL=gemma4:12b-mlx
#
# Example:
#   PORT=9000 OLLAMA_MODEL=llama3.2 scripts/run.sh

set -e

# Resolve repo root (this script lives in scripts/) and run from there so the
# script works no matter where it is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Prefer Python 3.13, fall back to python3.
PYTHON_CMD="python3.13"
command -v python3.13 &>/dev/null || PYTHON_CMD="python3"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment with $PYTHON_CMD..."
  "$PYTHON_CMD" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt          # core voice-agent deps
pip install --quiet -r requirements-serve.txt     # serving-layer deps

# Runtime config (override via environment).
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
BACKENDS="${BACKENDS:-echo,ollama}"
OLLAMA_MODEL="${OLLAMA_MODEL:-gemma4:12b-mlx}"
URL="http://$HOST:$PORT"

echo ""
echo "========================================================"
echo " ollama-voice serving dashboard"
echo "   URL:          $URL"
echo "   backends:     $BACKENDS"
echo "   ollama model: $OLLAMA_MODEL"
echo "   Ctrl+C to stop"
echo "========================================================"
echo ""

# Open the browser once the server answers /health (best-effort, backgrounded).
(
  for _ in $(seq 1 40); do
    if curl -s -o /dev/null "$URL/health" 2>/dev/null; then
      if command -v open &>/dev/null; then open "$URL"
      elif command -v xdg-open &>/dev/null; then xdg-open "$URL"
      fi
      break
    fi
    sleep 0.5
  done
) &

# Hand the terminal to the server (foreground; replaces this shell).
exec python serve.py \
  --host "$HOST" \
  --port "$PORT" \
  --backends "$BACKENDS" \
  --ollama-model "$OLLAMA_MODEL"
