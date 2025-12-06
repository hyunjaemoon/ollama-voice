"""
Text-to-speech synthesis using pyttsx3.

This module handles the conversion of text to speech, including proper
management of recording state to prevent audio feedback loops.
"""

import queue
import time

import config
import state


def text_to_speech(text: str) -> None:
    """
    Convert text to speech using TTS engine.

    This function stops recording before speaking to prevent audio feedback loops
    (the microphone would pick up the TTS output). After speaking, it adds a
    small delay to let audio settle before resuming recording.

    Args:
        text: Text to convert to speech
    """
    # CRITICAL: Stop recording while speaking to prevent echo/feedback
    # The microphone would otherwise pick up the TTS output and create a feedback loop
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

    try:
        # Queue text for speech synthesis and wait for completion
        # runAndWait() blocks until speech is completely finished
        state.tts_engine.say(text)
        state.tts_engine.runAndWait()

        # Additional wait to ensure TTS is completely done
        # Some TTS engines may return before audio playback finishes
        time.sleep(0.2)
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
    finally:
        # Mark speaking as complete
        state.is_speaking = False
        # Small delay after speaking to let audio settle
        # This prevents the microphone from immediately picking up echo/reverb
        time.sleep(config.TTS_SETTLE_DELAY)
