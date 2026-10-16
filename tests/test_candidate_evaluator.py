import json

import httpx
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from candidate_evaluator import (
    ESSENTIAL, EvaluationError, _parse_evidence, evaluate, load_vacancy, retrieve_vacancy,
    run_evaluation,
)
import candidate_evaluator

QUOTES = {
    "PM": "Liderei como PM sênior o roadmap e o discovery de pagamentos digitais.",
    "PIX": "Conduzi a evolução do Pix para pagamentos de lojistas em canais digitais.",
    "DADOS": "Usei métricas de conversão e sucesso de transações para priorizar o produto.",
    "COLABORACAO": "Trabalhei com engenharia, negócio, risco e compliance nas entregas.",
}


def fake_model(prompt):
    body = prompt.to_messages()[-1].content
    found = [{"criterion_id": key, "quote": quote} for key, quote in QUOTES.items() if quote in body]
    return AIMessage(content=json.dumps({"evidence": found}))


def test_vacancy_retrieval_and_advancing_verdict():
    documents = load_vacancy()
    retrieved = retrieve_vacancy(documents, DeterministicFakeEmbedding(size=16))
    assert set(ESSENTIAL).issubset({doc.metadata["id"] for doc in retrieved})
    text = "\n".join(QUOTES.values())
    result = run_evaluation(text, DeterministicFakeEmbedding(size=16), RunnableLambda(fake_model))
    assert result.verdict == "Avançar para a próxima etapa"
    assert not result.missing
    assert all(result.evidence[key] == (quote,) for key, quote in QUOTES.items())


def test_missing_essential_recommends_refusal_without_claiming_lack_of_skill():
    text = "\n".join(quote for key, quote in QUOTES.items() if key != "PIX")
    result = run_evaluation(text, DeterministicFakeEmbedding(size=16), RunnableLambda(fake_model))
    assert result.verdict == "Recusar"
    assert result.missing == ("PIX",)
    assert result.evidence["PIX"] == ()


def test_unverifiable_quotes_do_not_support_advancement():
    response = json.dumps({"evidence": [{"criterion_id": "PIX", "quote": QUOTES["PIX"]}]})
    assert _parse_evidence(response, "Não trabalhei com esse produto.") == []


def test_repeated_evidence_keys_from_small_model_are_preserved():
    response = ("{\"evidence\":[{\"criterion_id\":\"PM\",\"quote\":\"" + QUOTES["PM"] +
                "\"}],\"evidence\":[{\"criterion_id\":\"PIX\",\"quote\":\"" + QUOTES["PIX"] + "\"}]}")
    assert _parse_evidence(response, "\n".join(QUOTES.values())) == [
        ("PM", QUOTES["PM"]), ("PIX", QUOTES["PIX"]),
    ]


@pytest.mark.parametrize("response", ["", "não é JSON", "{}", '{"evidence":{}}'])
def test_incomplete_model_response_fails(response):
    with pytest.raises(EvaluationError):
        _parse_evidence(response, "Currículo")


def test_ollama_connection_error_is_readable(monkeypatch):
    monkeypatch.setattr(candidate_evaluator, "OllamaEmbeddings", lambda **_: (_ for _ in ()).throw(httpx.ConnectError("offline")))
    with pytest.raises(EvaluationError, match="Ollama não está disponível"):
        evaluate("Experiência em produto.")
