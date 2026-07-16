"""
Text-to-speech synthesis using pyttsx3.

This module handles the conversion of text to speech, including proper
management of recording state to prevent audio feedback loops.
"""

import time

import config
import state


def begin_speaking() -> None:
    """
    Prepare for TTS output by silencing the microphone.

    CRITICAL: Recording must stop while speaking to prevent echo/feedback —
    the microphone would otherwise pick up the TTS output and create a
    feedback loop. Call this once before the first sentence of a response.
    """
    state.is_recording = False
    state.is_speaking = True

    # Explicitly stop and close the audio stream if it's still open
    if state.audio_stream is not None:
        try:
            state.audio_stream.stop()
            state.audio_stream.close()
        except Exception as e:
            print(f"Warning: Error closing audio stream: {e}")
        finally:
            state.audio_stream = None


def speak_sentence(text: str) -> None:
    """
    Speak one sentence, blocking until playback finishes.

    Must run on the thread that created the TTS engine (pyttsx3 engines are
    not thread-safe). Call between begin_speaking() and end_speaking().

    Args:
        text: Text to convert to speech
    """
    try:
        # runAndWait() blocks until speech is completely finished
        state.tts_engine.say(text)
        state.tts_engine.runAndWait()
    except Exception as e:
        print(f"Error in text-to-speech: {e}")


def end_speaking() -> None:
    """
    Mark speech finished and let the audio settle before listening again.

    The extra delays prevent the microphone from immediately picking up
    echo/reverb from the tail of the TTS output.
    """
    # Some TTS engines may return before audio playback finishes
    time.sleep(0.2)
    state.is_speaking = False
    time.sleep(config.TTS_SETTLE_DELAY)


def text_to_speech(text: str) -> None:
    """
    Convert text to speech using TTS engine.

    Convenience wrapper for speaking a complete response in one call. For
    streaming responses, use begin_speaking() / speak_sentence() /
    end_speaking() directly to speak sentence-by-sentence.

    Args:
        text: Text to convert to speech
    """
    begin_speaking()
    try:
        speak_sentence(text)
    finally:
        end_speaking()
