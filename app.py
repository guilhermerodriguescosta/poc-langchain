"""Execute com: python -m streamlit run app.py."""

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from documents import DocumentError, extract_text
from summarizer import SummaryError, summarize

load_dotenv(Path(__file__).with_name(".env"))


def clear_result() -> None:
    st.session_state.pop("summary", None)


def main() -> None:
    st.set_page_config(page_title="Resumo de documentos", page_icon="📄")
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


if __name__ == "__main__":
    main()
