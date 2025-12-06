"""
Model initialization for Whisper, TTS, and Ollama.

This module handles the initialization of all models used in the service,
including Whisper for speech recognition, TTS engine for text-to-speech,
and verification of Ollama connectivity.
"""

import sys

import ollama
import pyttsx3
from faster_whisper import WhisperModel

import config
import state


def initialize_models(
    whisper_model_size: str = "base", ollama_model: str = "llama3.2"
) -> str:
    """
    Initialize Whisper, TTS engine, and verify Ollama connection.

    This function loads the Whisper model for speech recognition, initializes the
    TTS engine with preferred voice settings, and performs a connectivity test
    with Ollama to ensure the service is available.

    Args:
        whisper_model_size: Size of Whisper model to load (tiny, base, small, medium, large-v2)
        ollama_model: Name of Ollama model to use for LLM processing

    Returns:
        The name of the verified Ollama model

    Raises:
        SystemExit: If Ollama connection fails or model cannot be initialized
    """
    # Initialize Whisper model for speech-to-text conversion
    # Try to leverage GPU acceleration when available (e.g., CUDA on NVIDIA)
    # Falls back to CPU if GPU acceleration is not available
    print(f"Loading Whisper model '{whisper_model_size}'...")

    # Try GPU acceleration with different compute types in order of preference
    compute_types_to_try = [
        ("auto", "int8_float16"),  # Mixed precision for GPU (best performance)
        ("auto", "int8"),           # Quantized GPU (if supported)
        ("cpu", "int8"),            # CPU fallback (always works)
    ]

    whisper_model = None
    for device, compute_type in compute_types_to_try:
        try:
            print(
                f"   Trying device='{device}', compute_type='{compute_type}'...")
            whisper_model = WhisperModel(
                whisper_model_size,
                device=device,
                compute_type=compute_type,
            )
            if device == "auto":
                print(
                    f"   ✓ GPU acceleration enabled with compute_type='{compute_type}'")
            else:
                print(f"   ✓ Using CPU with compute_type='{compute_type}'")
            break
        except ValueError as e:
            if "compute type" in str(e).lower():
                # Try next compute type
                continue
            else:
                # Re-raise if it's a different error
                raise
        except Exception as e:
            # For other errors, try next option
            print(f"   ⚠️  Failed: {e}")
            continue

    if whisper_model is None:
        raise RuntimeError(
            "Failed to initialize Whisper model with any available configuration")
    print("Whisper model loaded!")

    # Store Whisper model in global state
    state.whisper_model = whisper_model

    # Initialize TTS engine using pyttsx3
    print("Initializing TTS engine...")
    tts_engine = pyttsx3.init()

    # Select preferred voice if available
    voices = tts_engine.getProperty("voices")
    if voices:
        for voice in voices:
            # Check if voice name contains preferred names (case-insensitive)
            if any(
                preferred_name in voice.name.lower()
                for preferred_name in config.PREFERRED_VOICE_NAMES
            ):
                tts_engine.setProperty("voice", voice.id)
                break

    # Configure TTS properties for natural-sounding speech
    tts_engine.setProperty("rate", config.TTS_RATE)
    tts_engine.setProperty("volume", config.TTS_VOLUME)
    print("TTS engine ready!")

    # Store TTS engine in global state
    state.tts_engine = tts_engine

    # Verify Ollama connection by attempting a test generation
    # This ensures the service is running and the model is available before proceeding
    try:
        print(f"Checking Ollama model '{ollama_model}'...")
        response = ollama.generate(model=ollama_model, prompt="test")
        print("Ollama connection verified!")
        return ollama_model
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Please ensure Ollama is running: ollama serve")
        sys.exit(1)
