import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from ollama import ResponseError

import summarizer
from summarizer import SummaryError, summarize


def test_chain_prompt_model_parser(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    captured = {}

    def answer(prompt):
        captured["messages"] = prompt.to_messages()
        return AIMessage(content="## Resumo\nBiblioteca aos sábados.\n\n## Pontos principais\n- Empréstimo gratuito.")

    def model(**kwargs):
        captured["options"] = kwargs
        return RunnableLambda(answer)

    monkeypatch.setattr(summarizer, "ChatOllama", model)
    text = "Biblioteca aos sábados. Texto literal: {não é variável}."
    result = summarize(text)
    assert "## Resumo" in result and "## Pontos principais" in result
    assert text in captured["messages"][1].content
    assert "português" in captured["messages"][0].content
    assert captured["options"]["model"] == "llama3.2:3b"
    assert captured["options"]["base_url"] == "http://127.0.0.1:11434"
    assert captured["options"]["client_kwargs"]["trust_env"] is False
    assert captured["options"]["num_gpu"] == 0


def test_local_model_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:1b")
    def model(**kwargs):
        assert kwargs["model"] == "llama3.2:1b"
        return RunnableLambda(lambda _: AIMessage(content="Resumo"))
    monkeypatch.setattr(summarizer, "ChatOllama", model)
    assert summarize("Documento") == "Resumo"


@pytest.mark.parametrize(("text", "message"), [(" \n", "Não há texto"), ("a" * 10_000_001, "10 MB")], ids=["empty", "over-10-mb"])
def test_invalid_text_never_calls_model(monkeypatch, text, message):
    def forbidden(**kwargs):
        pytest.fail("Modelo não deveria ser chamado")
    monkeypatch.setattr(summarizer, "ChatOllama", forbidden)
    with pytest.raises(SummaryError, match=message):
        summarize(text)


def test_cloud_model_rejected(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "modelo:cloud")
    with pytest.raises(SummaryError, match="modelo local"):
        summarize("Documento")


@pytest.mark.parametrize(("error", "message"), [
    (ConnectionError("detalhes internos"), "O Ollama não está disponível"),
    (httpx.ConnectError("conexão recusada"), "O Ollama não está disponível"),
    (httpx.ReadTimeout("timeout"), "demorou demais"),
    (ResponseError("not found", status_code=404), "ollama pull"),
    (ResponseError("memória insuficiente", status_code=500), "memória disponível"),
])
def test_runtime_errors(monkeypatch, error, message):
    def answer(_):
        raise error
    monkeypatch.setattr(summarizer, "ChatOllama", lambda **_: RunnableLambda(answer))
    with pytest.raises(SummaryError, match=message):
        summarize("Documento")


def test_empty_model_response(monkeypatch):
    monkeypatch.setattr(summarizer, "ChatOllama", lambda **_: RunnableLambda(lambda _: AIMessage(content=" ")))
    with pytest.raises(SummaryError, match="resposta vazia"):
        summarize("Documento")


@pytest.mark.parametrize("value", ["abc", "-2"])
def test_invalid_gpu_configuration(monkeypatch, value):
    monkeypatch.setenv("OLLAMA_NUM_GPU", value)
    with pytest.raises(SummaryError, match="OLLAMA_NUM_GPU"):
        summarize("Documento")


def test_split_preserves_unicode_and_all_content():
    text = ("Ação na biblioteca 📚.\n" * 900) + "FIM"
    parts = list(summarizer.split_text(text))
    assert "".join(parts) == text
    assert len(parts) > 1
    assert all(len(part.encode("utf-8")) <= summarizer.CHUNK_BYTES for part in parts)


def test_ten_mb_text_is_processed_in_full_and_reduced(monkeypatch):
    original_bytes = 0
    calls = 0
    progress = []

    def answer(prompt):
        nonlocal original_bytes, calls
        body = prompt.to_messages()[1].content.split("\n\n", 1)[1]
        assert len(body.encode("utf-8")) <= summarizer.CHUNK_BYTES
        if body.startswith("a"):
            original_bytes += len(body)
        calls += 1
        return AIMessage(content="Resumo parcial.")

    monkeypatch.setattr(summarizer, "ChatOllama", lambda **_: RunnableLambda(answer))
    result = summarize("a" * 10_000_000, lambda *args: progress.append(args))
    assert result == "Resumo parcial."
    assert original_bytes == 10_000_000
    assert calls > 1667  # Partes originais, redução dos resumos e chamada final.
    assert any("Consolidando" in label for _, _, label in progress)
    assert progress[-1] == (1, 1, "Resumo concluído")


def test_non_converging_model_fails_instead_of_looping(monkeypatch):
    def answer(prompt):
        body = prompt.to_messages()[1].content.split("\n\n", 1)[1]
        return AIMessage(content=body)
    monkeypatch.setattr(summarizer, "ChatOllama", lambda **_: RunnableLambda(answer))
    with pytest.raises(SummaryError, match="não conseguiu compactar"):
        summarize("a" * 6001)


def test_empty_partial_response_stops_processing(monkeypatch):
    monkeypatch.setattr(summarizer, "ChatOllama", lambda **_: RunnableLambda(lambda _: AIMessage(content="")))
    with pytest.raises(SummaryError, match="resposta vazia"):
        summarize("a" * 6001)
