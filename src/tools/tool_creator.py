"""
Tool Creator: Multi-step wizard that lets the user define new tools at runtime.

Flow:
  1. Agent calls create_new_tool(idea="...")
  2. Wizard asks clarifying questions via console
  3. LLM generates tool code + test cases
  4. Tests are executed to validate the tool
  5. Tool is hot-loaded into the running application
"""

import importlib
import importlib.util
import os
import re
import sys
import textwrap
from pathlib import Path

from langchain_core.tools import tool as tool_decorator
from langchain_ollama import ChatOllama

CUSTOM_TOOLS_DIR = Path(__file__).parent / "custom"
CUSTOM_TOOLS_DIR.mkdir(exist_ok=True)

# Registry of dynamically loaded tools (shared with tools.py)
_dynamic_tools: list = []


def get_dynamic_tools() -> list:
    return list(_dynamic_tools)


def _ask_user(prompt: str) -> str:
    """Interactive console input during wizard."""
    print(f"\n🔧 Tool-Wizard: {prompt}")
    return input("👉 ").strip()


def _generate_tool_code(llm: ChatOllama, spec: dict) -> str:
    """Use the LLM to generate tool code from the gathered spec."""
    prompt = textwrap.dedent(f"""\
        Generate a single Python function decorated with @tool for a LangChain tool.
        
        Tool name: {spec['name']}
        Purpose: {spec['purpose']}
        Parameters: {spec['parameters']}
        Expected return: {spec['return_description']}
        Example usage: {spec['example']}
        
        Rules:
        - Use this exact import: from langchain_core.tools import tool
        - Decorate with @tool
        - Add a clear docstring in German
        - Only use stdlib + httpx + json (no other dependencies)
        - Return a string result
        - Handle errors with try/except
        - Output ONLY the Python code, no markdown fences, no explanation
    """)

    response = llm.invoke(prompt)
    code = response.content.strip()
    # Strip markdown fences if the LLM adds them
    code = re.sub(r"^```python\s*\n?", "", code)
    code = re.sub(r"\n?```\s*$", "", code)
    return code


def _generate_tests(llm: ChatOllama, spec: dict, tool_code: str) -> str:
    """Generate simple test cases for the tool."""
    prompt = textwrap.dedent(f"""\
        Generate pytest test functions for this tool.
        
        Tool name: {spec['name']}
        Purpose: {spec['purpose']}
        Example: {spec['example']}
        
        Tool code:
        {tool_code}
        
        Rules:
        - Import the tool function from the generated module
        - Use: from tools.custom.{spec['name']} import {spec['name']}
        - Call .invoke({{"input": ...}}) since it's a LangChain @tool
        - Test that the result is a non-empty string
        - Test one valid input and one edge case
        - Output ONLY Python code, no markdown fences
    """)

    response = llm.invoke(prompt)
    code = response.content.strip()
    code = re.sub(r"^```python\s*\n?", "", code)
    code = re.sub(r"\n?```\s*$", "", code)
    return code


def _save_module(name: str, code: str) -> Path:
    """Save generated code as a Python module in tools/custom/."""
    init_path = CUSTOM_TOOLS_DIR / "__init__.py"
    if not init_path.exists():
        init_path.write_text("")

    path = CUSTOM_TOOLS_DIR / f"{name}.py"
    path.write_text(code)
    return path


def _load_tool_from_module(name: str, module_path: Path):
    """Dynamically import a tool function from a generated module."""
    spec = importlib.util.spec_from_file_location(f"tools.custom.{name}", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"tools.custom.{name}"] = module
    spec.loader.exec_module(module)

    func = getattr(module, name, None)
    if func is None:
        raise ImportError(f"Module {module_path} has no function named '{name}'")
    return func


def _run_tests(test_code: str, tool_name: str) -> tuple[bool, str]:
    """Execute generated tests and return (passed, output)."""
    import subprocess

    test_path = CUSTOM_TOOLS_DIR / f"test_{tool_name}.py"
    test_path.write_text(test_code)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=Path(__file__).parent.parent,
    )

    output = result.stdout + result.stderr
    passed = result.returncode == 0
    return passed, output


def run_wizard(idea: str, llm_model: str = "qwen2.5") -> str:
    """
    Multi-step wizard: gather requirements → generate → test → load.
    Returns a status message for the agent to relay to the user.
    """
    print("\n" + "=" * 50)
    print("🔧 Tool-Erstellungs-Wizard gestartet")
    print("=" * 50)
    print(f"💡 Idee: {idea}")

    # ── Step 1: Gather details ──
    name = _ask_user(
        "Wie soll das Tool heißen? (snake_case, z.B. 'translate_text')"
    )
    name = re.sub(r"[^a-z0-9_]", "_", name.lower().strip())
    if not name:
        return "Tool-Erstellung abgebrochen: Kein Name angegeben."

    purpose = _ask_user(
        "Was genau soll das Tool tun? Beschreibe den Zweck in 1-2 Sätzen."
    )

    parameters = _ask_user(
        "Welche Parameter braucht das Tool? (z.B. 'text: str, target_lang: str')"
    )

    return_desc = _ask_user(
        "Was soll das Tool zurückgeben? (z.B. 'Den übersetzten Text als String')"
    )

    example = _ask_user(
        "Gib ein Beispiel: Eingabe → erwartete Ausgabe"
    )

    spec = {
        "name": name,
        "purpose": purpose,
        "parameters": parameters,
        "return_description": return_desc,
        "example": example,
    }

    # ── Step 2: Generate tool code ──
    print("\n⏳ Generiere Tool-Code...")
    llm = ChatOllama(model=llm_model, base_url="http://localhost:11434", temperature=0.2)

    tool_code = _generate_tool_code(llm, spec)
    module_path = _save_module(name, tool_code)
    print(f"✅ Code gespeichert: {module_path}")
    print(f"📋 Generierter Code:\n{tool_code}\n")

    # ── Step 3: Generate and run tests ──
    print("⏳ Generiere Tests...")
    test_code = _generate_tests(llm, spec, tool_code)
    print(f"📋 Tests:\n{test_code}\n")

    print("🧪 Führe Tests aus...")
    passed, output = _run_tests(test_code, name)
    print(output)

    if not passed:
        print("❌ Tests fehlgeschlagen. Tool wird NICHT geladen.")
        return (
            f"Das Tool '{name}' konnte erstellt werden, aber die Tests sind fehlgeschlagen. "
            f"Der Code liegt unter {module_path} und kann manuell korrigiert werden."
        )

    # ── Step 4: Hot-load into running app ──
    print("⏳ Lade Tool in laufende Anwendung...")
    new_tool = _load_tool_from_module(name, module_path)
    _dynamic_tools.append(new_tool)

    print(f"✅ Tool '{name}' erfolgreich geladen und verfügbar!")
    print("=" * 50)

    return (
        f"Neues Tool '{name}' wurde erstellt, getestet und geladen. "
        f"Es steht ab sofort zur Verfügung."
    )
