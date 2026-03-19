import operator
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import BaseMessage


class AssistantState(TypedDict):
	audio: Optional[bytes]  # Roh-Audio Input
	user_text: Optional[str]  # Transkribierter Text
	messages: Annotated[list[BaseMessage], operator.add]  # Gesprächsverlauf
	response_text: Optional[str]  # Finale Antwort für TTS
