# Ollama Voice

A fully local, bidirectional Speech-to-Speech service using Ollama, Whisper, and TTS. Supports GPU acceleration when available.

## Features

- 🎤 **Speech-to-Text**: Uses `faster-whisper` for accurate, local speech recognition
- 🤖 **LLM Processing**: Integrates with Ollama for local language model processing
- 🔊 **Text-to-Speech**: Uses system voices via `pyttsx3` for natural speech synthesis
- ⚡ **GPU Accelerated**: Leverages GPU acceleration when available
- 🔒 **Fully Local**: All processing happens locally, no cloud services required
- 🎯 **Real-time**: Continuous listening with silence detection for natural conversation flow

## Prerequisites

1. **Operating System**: Linux, Windows, or macOS
2. **Python 3.8+**
3. **Ollama** installed and running
   ```bash
   # Download from https://ollama.com
   # Or use your system's package manager
   ```

4. **Ollama Model** - Pull a model (e.g., llama3.2):
   ```bash
   ollama pull llama3.2
   ```

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd ollama-voice
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure Ollama is running:
   ```bash
   ollama serve
   ```

## Usage

Run the service:
```bash
python main.py
```

### Command-line Options

```bash
python main.py --help
```

Options:
- `--model`: Ollama model to use (default: `llama3.2`)
- `--whisper-model`: Whisper model size - `tiny`, `base`, `small`, `medium`, `large-v2` (default: `base`)
- `--sample-rate`: Audio sample rate in Hz (default: `16000`)

### Example

```bash
# Use a different Ollama model
python main.py --model llama3.1

# Use a larger Whisper model for better accuracy
python main.py --whisper-model medium
```

## How It Works

1. **Audio Capture**: Records audio from your microphone using `sounddevice`
2. **Speech Recognition**: Converts speech to text using `faster-whisper` (local Whisper implementation)
3. **LLM Processing**: Sends transcribed text to Ollama for processing
4. **Speech Synthesis**: Converts LLM response to speech using system TTS voices
5. **Continuous Loop**: Repeats the process for bidirectional conversation

## Troubleshooting

### Ollama Connection Issues
- Ensure Ollama is running: `ollama serve`
- Verify your model is available: `ollama list`
- Pull the model if missing: `ollama pull <model-name>`

### Audio Issues
- Check microphone permissions in your system settings
- Verify audio input device: `python -c "import sounddevice; print(sounddevice.query_devices())"`

### Whisper Model Download
- First run will download the Whisper model automatically
- Models are cached in `~/.cache/huggingface/`

### TTS Voice Issues
- List available voices: `python -c "import pyttsx3; e = pyttsx3.init(); print([v.name for v in e.getProperty('voices')])"`

## Performance Tips

- **Whisper Model Size**: 
  - `tiny`: Fastest, lower accuracy
  - `base`: Good balance (default)
  - `medium`/`large-v2`: Better accuracy, slower
  
- **Ollama Model**: Smaller models (like `llama3.2`) are faster, larger models provide better responses

- **GPU Acceleration**: Ollama automatically uses GPU acceleration when available

## License

See LICENSE file for details.