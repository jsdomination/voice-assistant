"""
Voice Assistant – vibecoded entry point.
Stack: Ollama (LLM) + faster-whisper (STT) + Piper (TTS) + LangGraph (orchestration)
"""

import argparse
import asyncio

from audio.listener import AudioListener
from audio.speaker import Speaker
from graph.agent import build_graph
from config.modes import MODES
from utils.runner import run_assistant


def parse_args():
	parser = argparse.ArgumentParser(description="Voice Assistant")
	parser.add_argument(
		"--mode",
		choices=MODES.keys(),
		default="accurate",
		help="Operating mode: 'fast' for low-latency, 'accurate' for high-quality (default: accurate)",
	)
	return parser.parse_args()


async def main(mode):
	print(f"🎙️  Voice Assistant gestartet – Modus: {mode.name}")
	print("=" * 50)

	listener = AudioListener(
		model_size=mode.whisper_model_size,
		beam_size=mode.beam_size,
		vad_filter=mode.vad_filter,
	)
	graph = build_graph(mode, listener)
	speaker = Speaker()

	await run_assistant(graph, listener, speaker)


if __name__ == "__main__":
	args = parse_args()
	mode = MODES[args.mode]()
	asyncio.run(main(mode))
