from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tools.tool_creator import get_dynamic_tools
from tools.tools import ALL_TOOLS
from utils.state import AssistantState

SYSTEM_PROMPT = """Du bist ein hilfreicher deutschsprachiger Sprachassistent.
Dir stehen Tools zur Verfügung. Wenn ein Tool die Frage beantworten kann, MUSST du es aufrufen.
Beschreibe niemals wie man ein Tool benutzt – rufe es direkt auf.
Antworte immer auf Deutsch in kurzen, gesprochenen Sätzen (keine Code-Blöcke, kein Markdown).
Halte deine Antworten SEHR kurz – maximal 1-2 Sätze. Keine Aufzählungen, keine Erklärungen, keine Nachfragen außer wenn unbedingt nötig.
WICHTIG: Rufe ein Tool NIEMALS mehrfach mit der gleichen oder ähnlichen Anfrage auf. Wenn ein Tool bereits Ergebnisse geliefert hat, fasse diese zusammen und antworte direkt."""


def transcribe_node(state: AssistantState, whisper) -> AssistantState:
	"""Audio → Text via faster-whisper."""
	audio = state.get("audio")

	if audio is None:
		return {"user_text": None, "response_text": None}

	# Wenn audio bereits Text ist (z.B. im Test-Modus)
	if isinstance(audio, str):
		return {"user_text": audio, "response_text": None}

	text = whisper._transcribe(audio)
	return {"user_text": text, "response_text": None}


def agent_node(state: AssistantState, llm) -> AssistantState:
	"""Hauptlogik: LLM entscheidet ob Tool-Call oder direkte Antwort.
	Resolves tools dynamically so hot-loaded tools are picked up.
	"""
	user_text = state.get("user_text")

	if not user_text:
		return {
			"messages": [AIMessage(content="Ich habe dich leider nicht verstanden.")],
			"response_text": "Ich habe dich leider nicht verstanden.",
		}

	# Bind tools at call time (picks up dynamically created tools)
	all_tools = ALL_TOOLS + get_dynamic_tools()
	llm_with_tools = llm.bind_tools(all_tools)

	existing = state.get("messages", [])
	if not existing:
		messages = [
			SystemMessage(content=SYSTEM_PROMPT),
			HumanMessage(content=user_text),
		]
	else:
		messages = existing + (
			[HumanMessage(content=user_text)]
			if not any(
				isinstance(m, HumanMessage) and m.content == user_text for m in existing
			)
			else []
		)

	response = llm_with_tools.invoke(messages)

	if response.tool_calls:
		for tc in response.tool_calls:
			print(f"🧠 Agent → Tool-Call: {tc['name']}({tc.get('args', {})})")
	else:
		print("🧠 Agent → Direkte Antwort")

	new_messages = []
	if not existing:
		new_messages.append(SystemMessage(content=SYSTEM_PROMPT))

	if not any(
		isinstance(m, HumanMessage) and m.content == user_text for m in existing
	):
		new_messages.append(HumanMessage(content=user_text))

	new_messages.append(response)

	return {
		"messages": new_messages,
		"response_text": response.content if not response.tool_calls else None,
	}


def finalize_node(state: AssistantState) -> AssistantState:
	"""Nach Tool-Execution: Letzte AI-Nachricht als response_text extrahieren."""
	messages = state.get("messages", [])

	for msg in reversed(messages):
		if isinstance(msg, AIMessage) and msg.content:
			return {"response_text": msg.content}

	return {"response_text": "Ich konnte keine Antwort generieren."}


def should_use_tools(state: AssistantState) -> str:
	"""Prüft ob der Agent ein Tool aufgerufen hat."""
	messages = state.get("messages", [])

	for msg in reversed(messages):
		if isinstance(msg, AIMessage):
			if msg.tool_calls:
				return "tools"
			break

	return "end"
