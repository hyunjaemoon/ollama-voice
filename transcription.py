"""
Speech-to-text transcription using Whisper.

This module handles the conversion of audio data to text using the Whisper
speech recognition model, including audio preprocessing and Voice Activity Detection.
"""

import traceback
from typing import Optional

import numpy as np

import config
import state


def speech_to_text(audio_data: np.ndarray) -> Optional[str]:
    """
    Convert speech audio to text using Whisper model.

    This function processes audio data through the Whisper speech recognition model.
    It performs audio preprocessing (normalization, format conversion) and attempts
    to use Voice Activity Detection (VAD) for better accuracy, falling back to
    standard transcription if VAD fails.

    Args:
        audio_data: Numpy array of audio samples (float32, mono, 16kHz)

    Returns:
        Transcribed text string, or None if transcription fails or audio is invalid
    """
    # Validate audio data
    if len(audio_data) == 0:
        print("⚠️  Empty audio data")
        return None

    # Calculate audio duration (assuming 16kHz sample rate)
    duration = len(audio_data) / config.DEFAULT_SAMPLE_RATE
    print(f"   Audio duration: {duration:.2f}s")

    # Reject audio that's too short (likely noise or error)
    if duration < config.MIN_AUDIO_DURATION:
        print("⚠️  Audio too short (< 0.1s)")
        return None

    # Ensure audio is in the correct format (float32)
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    # Normalize audio levels
    # If audio exceeds [-1, 1] range, normalize it
    max_val = np.abs(audio_data).max()
    if max_val > 1.0:
        audio_data = audio_data / max_val
    elif max_val < config.LOW_AUDIO_LEVEL_THRESHOLD:
        # Warn about low audio levels but still try to process
        print(f"⚠️  Audio level very low (max: {max_val:.4f})")

    # Ensure audio is 1D (mono) - flatten if multi-dimensional
    if len(audio_data.shape) > 1:
        audio_data = audio_data.flatten()

    try:
        # Attempt transcription with Voice Activity Detection (VAD) first
        # VAD helps filter out background noise and improves accuracy
        segments = None
        try:
            segments, info = state.whisper_model.transcribe(
                audio_data,
                beam_size=config.WHISPER_BEAM_SIZE,
                language=config.WHISPER_LANGUAGE,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=config.VAD_MIN_SILENCE_DURATION_MS,
                    threshold=config.VAD_THRESHOLD,
                ),
            )
            print(f"   Using VAD, detected language: {info.language}")
        except Exception as vad_error:
            # VAD failed - fall back to transcription without VAD
            # This handles cases where VAD parameters are incompatible or model doesn't support VAD
            print(f"   VAD failed, using regular transcription")
            try:
                segments, info = state.whisper_model.transcribe(
                    audio_data,
                    beam_size=config.WHISPER_BEAM_SIZE,
                    language=config.WHISPER_LANGUAGE,
                    vad_filter=False,
                )
                print(f"   Detected language: {info.language}")
            except Exception as e:
                print(f"❌ Transcription failed: {e}")
                return None

        # Validate that segments were returned
        if segments is None:
            print("⚠️  No segments returned from Whisper")
            return None

        # Collect text from all segments
        # Note: segments is a generator, so we need to iterate through it
        text_parts = []
        segment_count = 0
        for segment in segments:
            segment_count += 1
            text = segment.text.strip()
            if text:
                text_parts.append(text)
                print(f"   Segment {segment_count}: {text}")

        # Validate that we found at least one segment
        if segment_count == 0:
            print("⚠️  No segments found in audio")
            return None

        # Join all segments into final text
        final_text = " ".join(text_parts).strip()

        # Final validation
        if not final_text:
            print("⚠️  Whisper returned empty transcription after processing segments")
            return None

        return final_text

    except Exception as e:
        print(f"❌ Error in speech recognition: {e}")
        traceback.print_exc()
        return None
