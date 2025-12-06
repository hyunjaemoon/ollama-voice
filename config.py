"""
Configuration constants for the Speech-to-Speech service.

This module contains all configuration constants used throughout the application,
including audio processing parameters, detection thresholds, and model settings.
"""

from typing import Tuple

# ============================================================================
# Audio Processing Constants
# ============================================================================

# Audio processing constants
# Standard sample rate for speech recognition (16kHz)
DEFAULT_SAMPLE_RATE: int = 16000
# Seconds of silence before ending recording
DEFAULT_SILENCE_DURATION: float = 1.5
# Maximum seconds to wait for speech before timeout
DEFAULT_MAX_WAIT_TIME: float = 30.0

# Audio detection thresholds
# Minimum energy level to consider audio valid
MIN_AUDIO_ENERGY_THRESHOLD: float = 0.0001
ADAPTIVE_THRESHOLD_BASE: float = 0.01  # Base threshold for speech detection
ADAPTIVE_THRESHOLD_MIN: float = 0.005  # Minimum adaptive threshold
# Multiplier for adaptive threshold calculation
ADAPTIVE_THRESHOLD_MULTIPLIER: float = 0.3
MIN_AUDIO_DURATION: float = 0.1  # Minimum audio duration in seconds to process
# Threshold for warning about low audio levels
LOW_AUDIO_LEVEL_THRESHOLD: float = 0.01

# Audio buffer management
# Block size as ratio of sample rate (0.5 seconds)
AUDIO_BLOCK_SIZE_RATIO: float = 0.5
TEST_RECORDING_DURATION: float = 0.5  # Duration for microphone test in seconds
TEST_BLOCK_SIZE_RATIO: float = 0.1  # Block size for test recording
# Delay after starting audio stream to allow initialization
STREAM_STARTUP_DELAY: float = 0.2
# Interval between audio queue checks in seconds
RECORDING_POLL_INTERVAL: float = 0.01
# Interval for fixed-duration recording checks
RECORDING_CHUNK_INTERVAL: float = 0.1

# Speech detection parameters
RECENT_ENERGIES_BUFFER_SIZE: int = 10  # Number of recent energy values to track
# Minimum samples needed for adaptive threshold
ADAPTIVE_THRESHOLD_MIN_SAMPLES: int = 3
# Window size for calculating adaptive threshold
ADAPTIVE_THRESHOLD_WINDOW: int = 5
# Buffer size before checking for speech
SPEECH_DETECTION_BUFFER_THRESHOLD: int = 20
# Number of recent chunks to check for energy
SPEECH_DETECTION_ENERGY_CHECK: int = 10
# Energy threshold for speech detection
SPEECH_DETECTION_ENERGY_THRESHOLD: float = 0.005

# Timeout and error detection
NO_DATA_TIMEOUT_COUNT: int = 100  # Consecutive empty queue checks before warning
# Empty queue checks before processing existing audio
NO_DATA_PROCESSING_COUNT: int = 50
# Minimum chunks before processing without new data
MIN_AUDIO_CHUNKS_FOR_PROCESSING: int = 10

# TTS configuration
TTS_RATE: int = 180  # Speech rate in words per minute
TTS_VOLUME: float = 0.9  # Speech volume (0.0 to 1.0)
PREFERRED_VOICE_NAMES: Tuple[str, ...] = (
    "samantha", "alex")  # Preferred voice names

# Whisper transcription parameters
WHISPER_BEAM_SIZE: int = 5  # Beam size for Whisper decoding
WHISPER_LANGUAGE: str = "en"  # Default language for transcription
# Minimum silence duration for VAD in milliseconds
VAD_MIN_SILENCE_DURATION_MS: int = 500
VAD_THRESHOLD: float = 0.5  # VAD threshold (0.0 to 1.0)

# Post-processing delays
TTS_SETTLE_DELAY: float = 0.3  # Delay after TTS to let audio settle
# Additional delay after speaking before listening again
POST_SPEECH_DELAY: float = 0.5
