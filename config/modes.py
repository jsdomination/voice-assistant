"""
ConfigurationProfile: Konfigurationsprofile für den Voice Assistant.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ConfigurationProfile(ABC):
    """Abstract base for assistant operation modes."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    # ── Whisper STT ──
    @property
    @abstractmethod
    def whisper_model_size(self) -> str: ...

    @property
    @abstractmethod
    def beam_size(self) -> int: ...

    @property
    @abstractmethod
    def vad_filter(self) -> bool: ...

    # ── LLM ──
    @property
    @abstractmethod
    def llm_model(self) -> str: ...

    @property
    @abstractmethod
    def temperature(self) -> float: ...


@dataclass
class AccurateMode(ConfigurationProfile):
    """High-quality: larger Whisper model, smarter LLM, beam search."""

    name: str = "accurate"
    whisper_model_size: str = "medium"
    beam_size: int = 5
    vad_filter: bool = True
    llm_model: str = "glm-4.7-flash:latest"
    temperature: float = 0.3


@dataclass
class FastMode(ConfigurationProfile):
    """Low-latency: small Whisper model, lighter LLM, greedy decode."""

    name: str = "fast"
    whisper_model_size: str = "small"
    beam_size: int = 1
    vad_filter: bool = False
    llm_model: str = "qwen2.5"
    temperature: float = 0.1


MODES: dict[str, type[ConfigurationProfile]] = {
    "accurate": AccurateMode,
    "fast": FastMode,
}
