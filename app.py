"""Execute com: python -m streamlit run app.py."""

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from documents import DocumentError, extract_text
from candidate_evaluator import Evaluation, EvaluationError, LABELS, ESSENTIAL, evaluate
from summarizer import SummaryError, summarize

load_dotenv(Path(__file__).with_name(".env"))


def clear_result() -> None:
    st.session_state.pop("summary", None)


def clear_evaluation() -> None:
    st.session_state.pop("evaluation", None)


def summary_screen() -> None:
    st.title("Resumo de documentos")
    st.write("Anexe um documento e receba um resumo com os pontos principais em português.")
    st.caption("TXT em UTF-8, PDF textual ou DOCX · Arquivo e texto extraído de até 10 MB")
    st.info("Resumo processado pelo Ollama neste computador. PDFs digitalizados como imagem não são suportados.")
    st.caption("Documentos longos são resumidos em partes e consolidados ao final. Na CPU, arquivos grandes podem levar horas.")

    uploaded = st.file_uploader(
        "Escolha um documento", type=["txt", "pdf", "docx"],
        key="document", on_change=clear_result,
    )
    if st.button("Gerar resumo", type="primary", disabled=uploaded is None):
        clear_result()
        progress = st.progress(0, text="Preparando documento...")

        def update_progress(done: int, total: int, label: str) -> None:
            progress.progress(done / total, text=f"{label}: {done}/{total}")

        try:
            with st.spinner("Lendo o documento e gerando o resumo..."):
                text = extract_text(uploaded.name, uploaded.getvalue())
                st.session_state["summary"] = summarize(text, on_progress=update_progress)
        except (DocumentError, SummaryError) as error:
            progress.empty()
            st.error(str(error))
        except Exception:
            progress.empty()
            # Não expor traceback, texto do documento ou mensagens com credenciais.
            st.error("Não foi possível gerar o resumo. Verifique a configuração e tente novamente.")

    if "summary" in st.session_state:
        st.markdown(st.session_state["summary"])


def evaluation_screen() -> None:
    st.title("Avaliação de currículo com RAG")
    st.write("Compare um currículo com a vaga simulada de Product Manager Sênior para Pix na Getnet.")
    st.info("Vaga fictícia para estudo. O parecer é orientativo e exige revisão humana.")
    st.caption("TXT em UTF-8, PDF textual ou DOCX · Arquivo e texto extraído de até 10 MB")

    uploaded = st.file_uploader(
        "Escolha o currículo", type=["txt", "pdf", "docx"],
        key="curriculum", on_change=clear_evaluation,
    )
    if st.button("Avaliar candidato", type="primary", disabled=uploaded is None):
        clear_evaluation()
        progress = st.progress(0, text="Preparando avaliação...")

        def update_progress(done: int, total: int, label: str) -> None:
            progress.progress(done / total, text=f"{label}: {done}/{total}")

        try:
            with st.spinner("Recuperando a vaga e analisando o currículo..."):
                text = extract_text(uploaded.name, uploaded.getvalue())
                st.session_state["evaluation"] = evaluate(text, on_progress=update_progress)
        except (DocumentError, EvaluationError) as error:
            progress.empty()
            st.error(str(error))
        except Exception:
            progress.empty()
            st.error("Não foi possível avaliar o currículo. Verifique a configuração e tente novamente.")

    if "evaluation" in st.session_state:
        show_evaluation(st.session_state["evaluation"])


def show_evaluation(result: Evaluation) -> None:
    st.subheader("Visão geral")
    st.write(f"{len(result.matched)} de {len(ESSENTIAL)} requisitos essenciais têm evidência no currículo.")
    st.subheader("Aderência por requisito")
    for criterion_id in ESSENTIAL:
        quotes = result.evidence[criterion_id]
        st.write(f"{LABELS[criterion_id]}: {'Comprovado' if quotes else 'Não comprovado'}")
        for quote in quotes:
            st.text(f"Evidência do currículo: {quote}")
    st.subheader("Pontos positivos")
    if result.matched:
        for key in result.matched:
            st.write(f"{LABELS[key]} — evidência citada acima.")
    else:
        st.write("Nenhum requisito essencial ficou comprovado pelo currículo.")
    for quote in result.evidence["DIFERENCIAIS"]:
        st.text(f"Diferencial observado no currículo: {quote}")
    st.subheader("Lacunas e pontos negativos")
    if result.missing:
        for key in result.missing:
            st.write(f"{LABELS[key]}: não comprovado pelo currículo. Confirme em revisão humana.")
    else:
        st.write("Nenhuma lacuna nos requisitos essenciais foi identificada no currículo.")
    st.subheader("Veredito orientativo")
    st.write(result.verdict)
    if result.missing:
        st.write("Motivo: faltam evidências para " + ", ".join(LABELS[key] for key in result.missing) + ".")
    else:
        st.write("Motivo: há evidência no currículo para todos os requisitos essenciais.")
    with st.expander("Trechos da vaga recuperados pelo RAG"):
        for document in result.retrieved:
            st.write(document.metadata["title"])
            st.text(document.page_content)


def main() -> None:
    st.set_page_config(page_title="POC LangChain", page_icon="📄")
    summary_tab, evaluation_tab = st.tabs(["Resumo", "Avaliação com RAG"])
    with summary_tab:
        summary_screen()
    with evaluation_tab:
        evaluation_screen()


if __name__ == "__main__":
    main()
