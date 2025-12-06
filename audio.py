"""
Audio capture and recording functionality.

This module handles all audio-related operations including microphone testing,
audio recording with silence detection, and audio stream management.
"""

import queue
import time
from typing import Optional

import numpy as np
import sounddevice as sd
from sounddevice import InputStream

import config
import state


def audio_callback(
    indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags
) -> None:
    """
    Callback function for audio input stream.

    This function is called by sounddevice for each audio block. It copies the
    audio data to the global queue only when recording is active, allowing
    the main recording loop to process audio asynchronously.

    Args:
        indata: Input audio data as numpy array (shape: [frames, channels])
        frames: Number of frames in this block
        time_info: Dictionary with timing information (not used)
        status: CallbackFlags object indicating stream status (not used)
    """
    if state.is_recording:
        # Copy the audio data to avoid reference issues (indata is a view, not a copy)
        state.audio_queue.put(indata.copy())


def test_microphone(sample_rate: int = config.DEFAULT_SAMPLE_RATE) -> bool:
    """
    Test microphone functionality by recording a short sample.

    This function performs a quick test to verify that the microphone is
    accessible and producing audio data. It records for a short duration
    and checks if the audio energy exceeds a minimum threshold.

    Args:
        sample_rate: Audio sample rate in Hz (default: 16000)

    Returns:
        True if microphone is working and producing audio, False otherwise
    """
    print("Testing microphone...", end="", flush=True)
    test_buffer = []
    test_queue = queue.Queue()

    def test_callback(
        indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags
    ) -> None:
        """Local callback for test recording"""
        test_queue.put(indata.copy())

    try:
        # Open audio input stream for test recording
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,  # Mono audio
            dtype=np.float32,  # 32-bit float for audio processing
            callback=test_callback,
            blocksize=int(sample_rate * config.TEST_BLOCK_SIZE_RATIO),
        ):
            # Record for a short duration to test microphone
            time.sleep(config.TEST_RECORDING_DURATION)
            # Collect all audio chunks from the queue
            while not test_queue.empty():
                test_buffer.append(test_queue.get())

        if test_buffer:
            # Concatenate all audio chunks and calculate RMS energy
            test_audio = np.concatenate(test_buffer)
            # RMS (Root Mean Square) energy indicates audio level
            energy = np.sqrt(np.mean(test_audio**2))
            print(f" ✓ (energy: {energy:.4f})")
            # Return True if energy exceeds minimum threshold
            return energy > config.MIN_AUDIO_ENERGY_THRESHOLD
        else:
            print(" ✗ No audio data received")
            return False
    except Exception as e:
        print(f" ✗ Error: {e}")
        return False


def record_audio(
    sample_rate: int = config.DEFAULT_SAMPLE_RATE,
    silence_duration: float = config.DEFAULT_SILENCE_DURATION,
    max_duration: Optional[float] = None,
) -> np.ndarray:
    """
    Record audio from microphone until silence is detected or max duration reached.

    This function implements two recording modes:
    1. Fixed duration: Records for exactly max_duration seconds
    2. Silence detection: Records until silence_duration seconds of silence after speech

    The silence detection mode uses adaptive thresholding based on recent audio energy
    to handle varying microphone sensitivity and background noise levels.

    Args:
        sample_rate: Audio sample rate in Hz (default: 16000)
        silence_duration: Seconds of silence to wait before ending recording (default: 1.5)
        max_duration: If set, record for exactly this duration instead of using silence detection

    Returns:
        Numpy array of recorded audio samples (float32, mono)
        Empty array if no audio was recorded
    """
    state.is_recording = True
    state.audio_stream = None
    audio_buffer = []
    silence_start: Optional[float] = None
    speech_started = False
    adaptive_threshold = config.ADAPTIVE_THRESHOLD_BASE
    recent_energies = []
    start_time = time.time()
    no_data_count = 0

    # ========================================================================
    # Fixed Duration Recording Mode
    # ========================================================================
    if max_duration:
        try:
            state.audio_stream = sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32,
                callback=audio_callback,
                blocksize=int(sample_rate * config.AUDIO_BLOCK_SIZE_RATIO),
            )
            state.audio_stream.start()
            # Small delay to allow stream to initialize
            time.sleep(0.1)
            elapsed = 0
            # Record for the specified duration
            while elapsed < max_duration:
                if not state.audio_queue.empty():
                    chunk = state.audio_queue.get()
                    audio_buffer.append(chunk)
                    print(".", end="", flush=True)
                time.sleep(config.RECORDING_CHUNK_INTERVAL)
                elapsed = time.time() - start_time
            print()  # New line after recording dots
        except KeyboardInterrupt:
            print("\n⚠️  Recording interrupted.")
        finally:
            state.is_recording = False
            if state.audio_stream is not None:
                state.audio_stream.stop()
                state.audio_stream.close()
                state.audio_stream = None

        if not audio_buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(audio_buffer)

    # ========================================================================
    # Silence Detection Recording Mode
    # ========================================================================
    try:
        state.audio_stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32,
            callback=audio_callback,
            blocksize=int(sample_rate * config.AUDIO_BLOCK_SIZE_RATIO),
        )
        state.audio_stream.start()
        # Give the stream a moment to start before processing audio
        time.sleep(config.STREAM_STARTUP_DELAY)

        while True:
            # Check for timeout - if we've been waiting too long, process what we have
            elapsed = time.time() - start_time
            if elapsed > config.DEFAULT_MAX_WAIT_TIME:
                print(
                    f"\n⏱️  Timeout after {config.DEFAULT_MAX_WAIT_TIME}s. Processing recorded audio..."
                )
                break

            if not state.audio_queue.empty():
                # Reset no-data counter when we receive audio
                no_data_count = 0
                chunk = state.audio_queue.get()
                audio_buffer.append(chunk)

                # Calculate RMS energy for this audio chunk
                # Energy indicates how loud the audio is
                energy = np.sqrt(np.mean(chunk**2))
                recent_energies.append(energy)

                # Maintain a sliding window of recent energy values
                # This allows adaptive threshold calculation based on recent audio levels
                if len(recent_energies) > config.RECENT_ENERGIES_BUFFER_SIZE:
                    recent_energies.pop(0)

                # Update adaptive threshold based on recent maximum energy
                # This adapts to varying microphone sensitivity and background noise
                if len(recent_energies) > config.ADAPTIVE_THRESHOLD_MIN_SAMPLES:
                    # Use the maximum energy from recent window
                    max_recent = max(
                        recent_energies[-config.ADAPTIVE_THRESHOLD_WINDOW:])
                    # Threshold is 30% of recent max, but never below minimum
                    adaptive_threshold = max(
                        config.ADAPTIVE_THRESHOLD_MIN, max_recent * config.ADAPTIVE_THRESHOLD_MULTIPLIER
                    )

                # Speech detection: compare current energy to adaptive threshold
                if energy > adaptive_threshold:
                    # Speech detected
                    if not speech_started:
                        # First detection of speech - mark that speech has begun
                        speech_started = True
                        silence_start = None
                        print("🎤 Speech detected...", end="", flush=True)
                    else:
                        # Speech continuing - reset silence timer
                        silence_start = None
                        print(".", end="", flush=True)
                else:
                    # Energy below threshold - could be silence
                    if speech_started:
                        # We were in speech, now checking for silence
                        if silence_start is None:
                            # Start timing silence
                            silence_start = time.time()
                        elif time.time() - silence_start >= silence_duration:
                            # Silence duration exceeded - end recording
                            print()  # New line
                            break
                    elif len(audio_buffer) > config.SPEECH_DETECTION_BUFFER_THRESHOLD:
                        # We've been recording for a while without detecting speech
                        # Check if we have enough audio energy to process anyway
                        # This handles cases where speech detection threshold is too high
                        total_energy = np.sqrt(
                            np.mean(np.concatenate(
                                audio_buffer[-config.SPEECH_DETECTION_ENERGY_CHECK:]) ** 2)
                        )
                        if total_energy > config.SPEECH_DETECTION_ENERGY_THRESHOLD:
                            # We have some audio, might be speech - process it
                            speech_started = True
                            print("🎤 Processing audio...", end="", flush=True)
                            silence_start = time.time()
            else:
                # No audio data in queue
                no_data_count += 1

                # If we're not getting any data and have no audio, something is wrong
                if no_data_count > config.NO_DATA_TIMEOUT_COUNT and len(audio_buffer) == 0:
                    print("\n⚠️  No audio data received. Check microphone permissions.")
                    break

                # If we have audio but no new data for a while, process what we have
                # This handles cases where audio stream ends unexpectedly
                if (
                    speech_started
                    and len(audio_buffer) > config.MIN_AUDIO_CHUNKS_FOR_PROCESSING
                    and no_data_count > config.NO_DATA_PROCESSING_COUNT
                ):
                    print("\n📝 Processing recorded audio...")
                    break

            # Small sleep to prevent busy-waiting
            time.sleep(config.RECORDING_POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n⚠️  Recording interrupted.")
    except Exception as e:
        print(f"\n❌ Error during recording: {e}")
    finally:
        # Always stop recording, even on error
        state.is_recording = False
        if state.audio_stream is not None:
            state.audio_stream.stop()
            state.audio_stream.close()
            state.audio_stream = None

    # Return empty array if no audio was recorded
    if not audio_buffer:
        return np.array([], dtype=np.float32)

    # Concatenate all audio chunks into a single array
    return np.concatenate(audio_buffer)
