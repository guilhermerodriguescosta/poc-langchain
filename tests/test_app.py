from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from streamlit.testing.v1 import AppTest

import summarizer

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def screen(monkeypatch):
    monkeypatch.setattr(summarizer, "ChatOllama", lambda **_: RunnableLambda(
        lambda _: AIMessage(content="## Resumo\nBiblioteca aberta.\n\n## Pontos principais\n- Sábados.")
    ))
    return AppTest.from_file(str(APP)).run()


def test_initial_screen(screen):
    assert not screen.exception
    assert screen.button[0].disabled
    assert screen.title[0].value == "Resumo de documentos"


def test_upload_generate_and_replace_document(screen):
    screen.file_uploader[0].upload("teste.txt", "Biblioteca aos sábados.".encode(), "text/plain").run()
    assert not screen.button[0].disabled
    assert "summary" not in screen.session_state
    screen.button[0].click().run()
    assert not screen.exception
    assert any("Biblioteca aberta" in item.value for item in screen.markdown)
    screen.file_uploader[0].upload("outro.txt", b"Outro documento", "text/plain").run()
    assert "summary" not in screen.session_state


def test_invalid_upload_shows_error(screen):
    screen.file_uploader[0].upload("vazio.txt", b"", "text/plain").run()
    screen.button[0].click().run()
    assert not screen.exception
    assert "vazio" in screen.error[0].value


def test_connection_error_shows_recovery(screen, monkeypatch):
    def disconnected(_):
        raise ConnectionError("conexão recusada")
    monkeypatch.setattr(summarizer, "ChatOllama", lambda **_: RunnableLambda(disconnected))
    screen.file_uploader[0].upload("teste.txt", b"Documento", "text/plain").run()
    screen.button[0].click().run()
    assert not screen.exception
    assert "ollama serve" in screen.error[0].value
