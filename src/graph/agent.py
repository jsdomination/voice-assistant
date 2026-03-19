"""
LangGraph Agent – Kern-Orchestrierung des Voice Assistants

Graph-Flow:
  transcribe → agent → (tool_call?) → tools → agent → respond
                  ↑___________________________|
"""

import os

from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from config.modes import ConfigurationProfile
from tools.tool_creator import get_dynamic_tools
from tools.tools import ALL_TOOLS
from utils.nodes import agent_node, should_use_tools, transcribe_node
from utils.state import AssistantState


def build_graph(mode: ConfigurationProfile, listener):
	"""
	Baut und kompiliert den LangGraph-Agenten.

	Args:
		mode:     ConfigurationProfile (LLM model, Whisper config, …).
		listener: AudioListener instance shared with the run loop
		          (avoids loading Whisper twice).
	"""
	llm = ChatOllama(
		model=mode.llm_model,
		base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
		temperature=mode.temperature,
	)

	def dynamic_tool_node(state: AssistantState) -> AssistantState:
		all_tools = ALL_TOOLS + get_dynamic_tools()
		node = ToolNode(all_tools)
		return node.invoke(state)

	workflow = StateGraph(AssistantState)

	workflow.add_node("transcribe", lambda state: transcribe_node(state, listener))
	workflow.add_node("agent", lambda state: agent_node(state, llm))
	workflow.add_node("tools", dynamic_tool_node)

	workflow.set_entry_point("transcribe")
	workflow.add_edge("transcribe", "agent")
	workflow.add_conditional_edges(
		"agent",
		should_use_tools,
		{
			"tools": "tools",
			"end": END,
		},
	)
	workflow.add_edge("tools", "agent")

	return workflow.compile(checkpointer=MemorySaver())
