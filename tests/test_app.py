from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from streamlit.testing.v1 import AppTest

import summarizer
import candidate_evaluator
from candidate_evaluator import Evaluation, ESSENTIAL
from langchain_core.documents import Document

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


def test_evaluation_upload_result_and_replacement(monkeypatch):
    def evaluate_fake(text, on_progress=None):
        assert "Currículo" in text
        return Evaluation("Recusar", {key: () for key in (*ESSENTIAL, "DIFERENCIAIS")},
                          (Document(page_content="Trecho da vaga", metadata={"title": "Missão"}),))

    monkeypatch.setattr(candidate_evaluator, "evaluate", evaluate_fake)
    screen = AppTest.from_file(str(APP)).run()
    assert screen.button[1].disabled
    screen.file_uploader[1].upload("cv.txt", "Currículo fictício".encode(), "text/plain").run()
    assert not screen.button[1].disabled
    screen.button[1].click().run()
    assert not screen.exception
    assert any("Recusar" in item.value for item in screen.markdown)
    assert "evaluation" in screen.session_state
    screen.file_uploader[1].upload("outro.txt", "Outro currículo".encode(), "text/plain").run()
    assert "evaluation" not in screen.session_state


def test_invalid_curriculum_shows_error_without_inference(monkeypatch):
    monkeypatch.setattr(candidate_evaluator, "evaluate", lambda *_args, **_kwargs: pytest.fail("Não deveria chamar o modelo"))
    screen = AppTest.from_file(str(APP)).run()
    screen.file_uploader[1].upload("vazio.txt", b"", "text/plain").run()
    screen.button[1].click().run()
    assert not screen.exception
    assert "vazio" in screen.error[0].value
