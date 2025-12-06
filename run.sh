#!/bin/bash
set -e

# Use Python 3.13 if available, fallback to python3
PYTHON_CMD="python3.13"
if ! command -v python3.13 &> /dev/null; then
    PYTHON_CMD="python3"
fi

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment with $PYTHON_CMD..."
  $PYTHON_CMD -m venv .venv
fi

source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python3 main.py
