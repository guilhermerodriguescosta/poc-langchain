import pytest


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch):
    """Manter os testes independentes da configuração pessoal em .env."""
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:3b")
    monkeypatch.setenv("OLLAMA_NUM_GPU", "0")
