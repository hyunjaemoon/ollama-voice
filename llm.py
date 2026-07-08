"""
LLM processing with switchable backends.

This module handles text processing with the configured LLM backend:
- "ollama": the local Ollama service via the ollama Python package
- "furiosa": a Furiosa LLM server (`furiosa-llm serve`) via its
  OpenAI-compatible HTTP API

Both backends share the same contract: return the response text, or a
spoken-friendly error message if processing fails.
"""

import re
import traceback

import ollama
import requests

import config

# Generic error message returned to the user (and spoken by TTS) on failure
LLM_ERROR_MESSAGE = "I'm sorry, I encountered an error processing your request."


def generate_response(text: str, backend: str, model: str, llm_url: str) -> str:
    """
    Process text with the configured LLM backend.

    Args:
        text: Input text to process
        backend: LLM backend to use ("ollama" or "furiosa")
        model: Name of the model to use
        llm_url: Base URL of the OpenAI-compatible server (furiosa backend only)

    Returns:
        LLM response text, or error message if processing fails
    """
    if backend == "furiosa":
        return process_with_furiosa(text, model, llm_url)
    return process_with_ollama(text, model)


def process_with_furiosa(text: str, model: str, base_url: str) -> str:
    """
    Process text with a Furiosa LLM server via its OpenAI-compatible API.

    Sends a chat completion request to a running `furiosa-llm serve` instance
    and extracts the assistant message from the response.

    Args:
        text: Input text to process
        model: Model name to request ("EMPTY" routes to the served model)
        base_url: Base URL of the server, e.g. "http://localhost:8000/v1"

    Returns:
        LLM response text, or error message if processing fails
    """
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": text}],
            },
            timeout=config.LLM_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()

        # OpenAI-compatible response: choices[0].message.content
        choices = payload.get("choices") or []
        if choices:
            content = (choices[0].get("message") or {}).get("content")
            if content:
                return str(content).strip()

        print(f"⚠️  Unexpected response from Furiosa LLM: {str(payload)[:200]}...")
        return LLM_ERROR_MESSAGE

    except Exception as e:
        print(f"Error processing with Furiosa LLM: {e}")
        traceback.print_exc()
        return LLM_ERROR_MESSAGE


def process_with_ollama(text: str, model: str) -> str:
    """
    Process text with Ollama LLM and extract response.

    This function sends text to the Ollama LLM service and extracts the response.
    It handles multiple response formats that Ollama might return, including
    GenerateResponse objects, dictionaries, and string representations.

    Args:
        text: Input text to process
        model: Name of Ollama model to use

    Returns:
        LLM response text, or error message if processing fails
    """
    try:
        # Generate response from Ollama (non-streaming mode)
        response_obj = ollama.generate(model=model, prompt=text, stream=False)

        # Ollama can return responses in different formats depending on version
        # Try multiple extraction methods to handle various response types

        # Method 1: Check if response has 'response' attribute (GenerateResponse object)
        if hasattr(response_obj, "response"):
            result = response_obj.response
            if result is not None:
                return str(result).strip()

        # Method 2: Use getattr as fallback (handles attribute access edge cases)
        result = getattr(response_obj, "response", None)
        if result is not None:
            return str(result).strip()

        # Method 3: Check if response is a dictionary
        if isinstance(response_obj, dict):
            result = response_obj.get("response", "")
            if result:
                return str(result).strip()

        # Method 4: Extract from string representation using regex
        # This handles cases where response is embedded in a string representation
        response_str = str(response_obj)
        # Look for response="..." or response='...' pattern (handles multi-line)
        match = re.search(r'response=["\'](.*?)["\']', response_str, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            if extracted:
                return extracted

        # If all extraction methods fail, return error message
        print(
            f"⚠️  Could not extract response from Ollama. Type: {type(response_obj)}")
        print(f"   String representation: {response_str[:200]}...")
        return LLM_ERROR_MESSAGE

    except Exception as e:
        print(f"Error processing with Ollama: {e}")
        traceback.print_exc()
        return LLM_ERROR_MESSAGE
