"""
Shared voice assistant run loop.
Used by both main.py (vibecoded) and my_version.py (custom).
"""

from langgraph.errors import GraphRecursionError

_RECURSION_ERROR_MSG = (
	"Entschuldigung, die Anfrage war zu komplex. Kannst du es einfacher formulieren?"
)


async def run_assistant(graph, listener, speaker, session_id: str = "voice-session"):
	"""
	Core event loop: listen → graph → speak.

	Args:
	    graph:      Compiled LangGraph (ainvoke interface).
	    listener:   AudioListener instance (listen() → audio bytes).
	    speaker:    Speaker instance (speak(text) → TTS playback).
	    session_id: Thread ID for MemorySaver conversation history.
	"""
	while True:
		print("\n🟢 Höre zu...")
		audio = await listener.listen()

		if audio is None:
			continue

		try:
			result = await graph.ainvoke(
				{"audio": audio},
				config={
					"configurable": {"thread_id": session_id},
					"recursion_limit": 10,
				},
			)
		except GraphRecursionError:
			print("⚠️  Rekursionslimit erreicht – breche Anfrage ab.")
			result = {"response_text": _RECURSION_ERROR_MSG}

		if result.get("response_text"):
			print(f"🤖 Assistent: {result['response_text']}")
			await speaker.speak(result["response_text"])
