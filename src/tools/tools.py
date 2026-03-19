"""
Tools für den Voice Assistant.
Jedes Tool ist eine einfache Python-Funktion mit @tool Decorator.
"""

import httpx
import trafilatura
from langchain_core.tools import tool

_FETCH_TIMEOUT = 5  # seconds per page fetch
_MAX_CONTENT_CHARS = 2000  # max chars extracted per page


def _fetch_page_text(url: str) -> str:
    """Fetch a URL and extract meaningful text content using trafilatura."""
    try:
        resp = httpx.get(url, timeout=_FETCH_TIMEOUT, follow_redirects=True,
                         headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        text = trafilatura.extract(resp.text, include_comments=False,
                                   include_tables=True, favor_recall=True)
        if text:
            return text[:_MAX_CONTENT_CHARS]
        return ""
    except Exception:
        return ""


@tool
def search_web(query: str) -> str:
    """
    Sucht im Web nach aktuellen Informationen.
    
    Args:
        query: Suchanfrage auf Deutsch oder Englisch
    """
    try:
        from ddgs import DDGS

        print(f"🔧 Tool aufgerufen: search_web(query='{query}')")
        print(f"   🔍 Suche läuft...")

        with DDGS() as ddgs:
            results = list(ddgs.text(query, region="de-de", max_results=3))

        if not results:
            print("   ⚠️  Keine Ergebnisse gefunden")
            return f"Keine Ergebnisse für '{query}' gefunden."

        print(f"   ✅ {len(results)} Ergebnis(se) gefunden:")
        summaries = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            print(f"   [{i}] {title}")
            print(f"       {href}")
            print(f"       {body[:120]}{'...' if len(body) > 120 else ''}")
            summaries.append(f"{title}: {body}")

        # Fetch full content from all results for detailed answers
        for r in results:
            url = r.get("href", "")
            title = r.get("title", "")
            if not url:
                continue
            print(f"   📄 Lade Inhalt von: {url}")
            page_text = _fetch_page_text(url)
            if page_text:
                print(f"   📄 {len(page_text)} Zeichen extrahiert")
                summaries.append(f"\n--- Inhalt von {title} ---\n{page_text}")
            else:
                print(f"   ⚠️  Inhalt konnte nicht geladen werden")

        return "\n".join(summaries)

    except Exception as e:
        return f"Suche fehlgeschlagen: {str(e)}"


@tool
def create_new_tool(idea: str) -> str:
    """
    Erstellt ein neues Tool für den Assistenten basierend auf einer Idee.
    Verwende dieses Tool wenn keine der vorhandenen Tools die Anfrage erfüllen kann
    und der Nutzer eine neue Fähigkeit benötigt.

    Args:
        idea: Beschreibung was das neue Tool können soll
    """
    from tools.tool_creator import run_wizard
    return run_wizard(idea)


@tool
def list_tools() -> str:
    """
    Listet alle verfügbaren Tools auf, inklusive dynamisch erstellter Tools.
    Verwende dieses Tool wenn der Nutzer wissen will, was du alles kannst.
    """
    from tools.tool_creator import get_dynamic_tools

    print("🔧 Tool aufgerufen: list_tools")
    lines = ["Verfügbare Tools:"]
    for t in ALL_TOOLS:
        lines.append(f"- {t.name}: {t.description.split(chr(10))[0]}")
    dynamic = get_dynamic_tools()
    if dynamic:
        lines.append("Dynamisch erstellt:")
        for t in dynamic:
            lines.append(f"- {t.name}: {t.description.split(chr(10))[0]}")
    result = "\n".join(lines)
    print(f"   ↳ {len(ALL_TOOLS) + len(dynamic)} Tools gefunden")
    return result

@tool
def save_text(text: str) -> None:
    """
    Speichert die übergebene Text-Daten in einer Datei.

    :param text: Der Text, der gespeichert werden soll.
    :return: Keine Rückgabe.
    """
    try:
        with open("saved_data.txt", "w") as file:
            file.write(text)
    except Exception as e:
        return f"Fehler beim Speichern der Daten: {e}"

# Alle Tools als Liste exportieren
ALL_TOOLS = [search_web, create_new_tool, list_tools, save_text]