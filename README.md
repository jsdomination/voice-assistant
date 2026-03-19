# Voice Assistant – Lokales Siri auf Deutsch

Lokaler Voice Assistant mit LangGraph, Ollama, faster-whisper und Piper TTS.

## Voraussetzungen

- Python 3.11+
- [Ollama](https://ollama.com/download) installiert und gestartet
- [PortAudio](http://www.portaudio.com/) – Systemabhängigkeit für Audio I/O

## Setup

```bash
# 1. Virtuelle Umgebung erstellen und aktivieren
python -m venv .venv
source .venv/bin/activate

# 2. Python-Abhängigkeiten installieren
pip install -e .

# 3. LLM-Modelle herunterladen
ollama pull glm-4.7-flash   # Standard
ollama pull qwen2.5          # Schneller Modus

# 4. Deutsches Piper-Sprachmodell herunterladen
mkdir -p models
curl -L -o models/de_DE-thorsten-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx
curl -L -o models/de_DE-thorsten-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx.json

# 5. Starten
python src/main.py
```

## Modi

```bash
python src/main.py --mode accurate   # Standard – höhere Qualität (glm-4.7-flash + Whisper medium)
python src/main.py --mode fast       # Niedrige Latenz (qwen2.5 + Whisper small)
```

## Eingebaute Tools

| Tool | Beschreibung |
|---|---|
| `search_web` | Web-Suche via DuckDuckGo |
| `create_new_tool` | LLM generiert zur Laufzeit neue Tools |
| `list_tools` | Listet alle verfügbaren Tools |
| `save_text` | Speichert Text in einer Datei |
