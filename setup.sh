#!/bin/bash
# Setup script for ollama-voice

set -e

echo "Setting up ollama-voice..."
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version || { echo "Python 3 is required but not found. Please install Python 3.8+"; exit 1; }

# Check if Ollama is installed
echo "Checking Ollama installation..."
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed."
    echo "Please install it from: https://ollama.com"
    echo "Or use your system's package manager (e.g., brew, apt, pacman)"
    exit 1
fi
echo "Ollama found: $(ollama --version)"

# Check if Ollama is running
echo "Checking if Ollama is running..."
response=$(curl -s http://localhost:11434/api/tags)
if [ -z "$response" ]; then
    echo "Warning: Ollama server doesn't seem to be running."
    echo "Please start it with: ollama serve"
    echo "Or run it in the background"
else
    echo "Ollama server response:"
    # Pretty-print Ollama model list in table format
    if command -v jq &> /dev/null; then
        echo "$response" | jq -r '
            (["Model", "Size (GB)", "Modified At", "Family"] | @tsv),
            (.models[] | [
                (.name // "?"),
                (.size / 1073741824 | tostring),
                (.modified_at // "?"),
                (.details.family // "?")
            ] | @tsv)
        ' | column -t -s $'\t'
    else
        echo "Raw Ollama server models response (install 'jq' for nicer formatting):"
        echo "$response"
    fi
fi