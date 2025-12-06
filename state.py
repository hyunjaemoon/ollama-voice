"""
Global state management for the Speech-to-Speech service.

This module manages global state variables that need to be shared across
multiple modules, including model instances, audio queues, and recording flags.
"""

import queue
from typing import Optional

import pyttsx3
from faster_whisper import WhisperModel
from sounddevice import InputStream

# Global model instances (initialized once, reused throughout execution)
whisper_model: Optional[WhisperModel] = None
tts_engine: Optional[pyttsx3.Engine] = None

# Audio processing queue for streaming audio data from microphone
audio_queue: queue.Queue = queue.Queue()

# Flag to control recording state (prevents processing audio when not actively recording)
is_recording: bool = False

# Flag to indicate TTS is currently speaking (prevents starting new recording)
is_speaking: bool = False

# Global reference to audio input stream (for explicit control)
audio_stream: Optional[InputStream] = None
