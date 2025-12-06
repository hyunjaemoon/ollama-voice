"""
LLM processing using Ollama.

This module handles text processing with Ollama LLM, including response
extraction and error handling for various response formats.
"""

import re
import traceback

import ollama

import state


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
        return "I'm sorry, I encountered an error processing your request."

    except Exception as e:
        print(f"Error processing with Ollama: {e}")
        traceback.print_exc()
        return "I'm sorry, I encountered an error processing your request."
