#!/usr/bin/env python3
"""
Bidirectional Speech-to-Speech service using Ollama or Furiosa LLM.

This module provides a fully local implementation with GPU acceleration support.
It integrates Whisper for speech-to-text, a switchable LLM backend (Ollama or
Furiosa LLM) for processing, and pyttsx3 for text-to-speech synthesis.

The service operates in a continuous loop:
1. Records audio from microphone until silence is detected
2. Transcribes audio to text using Whisper
3. Processes text with the LLM backend
4. Converts LLM response to speech using TTS
5. Repeats the cycle for bidirectional conversation
"""

import argparse
import queue
import sys
import time

import audio
import config
import llm
import models
import state
import transcription
import tts


def main() -> None:
    """
    Main entry point for the Speech-to-Speech service.

    This function handles command-line argument parsing, initializes all models,
    and runs the main conversation loop. The loop continuously:
    1. Records audio until silence is detected
    2. Transcribes audio to text
    3. Processes text with the LLM backend
    4. Converts response to speech
    5. Repeats the cycle

    The function handles graceful shutdown on KeyboardInterrupt (Ctrl+C).
    """
    parser = argparse.ArgumentParser(
        description="Bidirectional Speech-to-Speech service using Ollama or Furiosa LLM"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=config.DEFAULT_LLM_BACKEND,
        choices=["ollama", "furiosa"],
        help=f"LLM backend to use (default: {config.DEFAULT_LLM_BACKEND})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            f"Model to use (default: {config.DEFAULT_OLLAMA_MODEL} for ollama, "
            f"{config.DEFAULT_FURIOSA_MODEL} for furiosa)"
        ),
    )
    parser.add_argument(
        "--llm-url",
        type=str,
        default=config.DEFAULT_FURIOSA_URL,
        help=(
            "Base URL of the OpenAI-compatible Furiosa LLM server, used with "
            f"--backend furiosa (default: {config.DEFAULT_FURIOSA_URL})"
        ),
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v2"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=config.DEFAULT_SAMPLE_RATE,
        help=f"Audio sample rate in Hz (default: {config.DEFAULT_SAMPLE_RATE})",
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=config.DEFAULT_SILENCE_DURATION,
        help=f"Silence duration to detect endpoint in seconds (default: {config.DEFAULT_SILENCE_DURATION})",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Maximum recording duration in seconds (default: None, uses silence detection)",
    )

    args = parser.parse_args()

    # Resolve the per-backend model default when --model is not given
    if args.model is None:
        args.model = (
            config.DEFAULT_FURIOSA_MODEL
            if args.backend == "furiosa"
            else config.DEFAULT_OLLAMA_MODEL
        )

    # Test microphone before proceeding
    # This catches permission issues and hardware problems early
    if not audio.test_microphone(args.sample_rate):
        print("⚠️  Microphone test failed. Please check:")
        print("   1. Microphone permissions in System Settings")
        print("   2. Microphone is connected and working")
        print("   3. No other app is using the microphone")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            sys.exit(1)

    # Initialize all models (Whisper, TTS, verify the LLM backend)
    llm_model = models.initialize_models(
        args.whisper_model, args.model, args.backend, args.llm_url
    )

    # Human-readable backend name for status messages
    backend_name = "Furiosa LLM" if args.backend == "furiosa" else "Ollama"

    # Display startup banner
    print("\n" + "=" * 60)
    print("Speech-to-Speech Service Started")
    print("=" * 60)
    print(f"Backend: {backend_name}")
    print(f"Model: {llm_model}")
    if args.backend == "furiosa":
        print(f"Server: {args.llm_url}")
    print("Press Ctrl+C to exit")
    print("=" * 60 + "\n")

    try:
        # Main conversation loop
        while True:
            # Wait until TTS is completely finished before starting to listen
            while state.is_speaking:
                time.sleep(0.1)

            print("\n🎤 Listening... (speak now)")
            if args.max_duration:
                print(f"   Will record for up to {args.max_duration} seconds")

            # Step 1: Record audio from microphone
            audio_data = audio.record_audio(
                args.sample_rate, args.silence_duration, args.max_duration
            )

            # Validate that we received audio
            if len(audio_data) == 0:
                print("No audio detected. Try again.")
                continue

            # Step 2: Transcribe audio to text
            print("🔍 Transcribing...")
            print(f"   Audio length: {len(audio_data)} samples")
            text = transcription.speech_to_text(audio_data)

            # Validate transcription
            if not text:
                print("❌ Could not understand speech. Please try again.")
                print("   Tips:")
                print("   - Speak clearly and at a normal volume")
                print("   - Try using --max-duration 5 for fixed recording")
                print("   - Check microphone is working properly")
                continue

            print(f"👤 You: {text}")

            # Step 3: Process text with the LLM backend
            print(f"🤖 Processing with {backend_name}...")
            response = llm.generate_response(
                text, args.backend, llm_model, args.llm_url
            )

            print(f"🤖 Assistant: {response}")

            # CRITICAL: Stop recording BEFORE speaking to prevent echo/feedback
            # This must happen before text_to_speech() to ensure recording is stopped
            state.is_recording = False

            # Explicitly stop and close the audio stream if it's still open
            if state.audio_stream is not None:
                try:
                    state.audio_stream.stop()
                    state.audio_stream.close()
                except Exception as e:
                    print(f"Warning: Error closing audio stream: {e}")
                finally:
                    state.audio_stream = None

            # Clear any remaining audio in the queue
            # This prevents processing TTS output that might have been captured
            while not state.audio_queue.empty():
                try:
                    state.audio_queue.get_nowait()
                except queue.Empty:
                    break

            # Step 4: Convert response to speech
            print("🔊 Speaking...")
            tts.text_to_speech(response)

            # Ensure speaking is completely finished before continuing
            # Double-check that TTS is done (safety check)
            while state.is_speaking:
                time.sleep(0.1)

            # Additional delay after speaking to let audio settle before listening again
            # This prevents the microphone from picking up reverb or echo
            time.sleep(config.POST_SPEECH_DELAY)

            print("✅ Complete!\n")

    except KeyboardInterrupt:
        # Graceful shutdown on Ctrl+C
        print("\n\nShutting down...")
        print("Goodbye!")


if __name__ == "__main__":
    main()
