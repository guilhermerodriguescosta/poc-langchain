# POC LangChain + Ollama: resumo de documentos

Página Streamlit que recebe TXT, PDF ou DOCX e gera **Resumo** e **Pontos principais** em português com um modelo local. Sem chave OpenAI ou tarifa de API. Depois da instalação e do download do modelo, a inferência pode funcionar sem internet.

## Objetivo e funcionamento

Estudar o fluxo básico do LangChain: upload, extração de texto, construção do prompt, execução do modelo e apresentação da resposta.

O usuário anexa um documento e clica em **Gerar resumo**. A aplicação solicita um parágrafo conciso e até cinco pontos principais em português, usando apenas os fatos do texto. O prompt trata o documento como referência e orienta o modelo a ignorar instruções contidas nele. O resultado deve ser revisado, pois o modelo pode omitir ou inventar informações.

Esta POC não inclui banco de dados, histórico persistente, login, RAG, embeddings ou modelos cloud. O processamento utiliza a memória, CPU/GPU e energia do computador.

## Arquitetura

Veja [DOCs/arquitetura.md](DOCs/arquitetura.md) para as visões C4 de contexto, contêineres e componentes, o fluxo de processamento e as decisões técnicas.

## Preparar o Ollama no Windows

Instale pelo [site oficial](https://ollama.com/download/windows) ou pelo PowerShell:

```powershell
winget install --id Ollama.Ollama --exact
```

Abra um novo terminal, inicie o aplicativo Ollama e baixe o modelo:

```powershell
ollama pull llama3.2:3b
ollama list
```

Se o serviço não estiver iniciado, execute `ollama serve` em outro terminal. A aplicação conversa exclusivamente com `http://127.0.0.1:11434`. O download do modelo ocupa alguns GB; a velocidade depende da CPU/GPU e da memória disponíveis.

## Executar

Requer Python 3.11 ou superior. No diretório do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Acesse <http://127.0.0.1:8501>. Anexe `exemplos/documento.txt` e clique em **Gerar resumo**. Confira se o resultado menciona horário aos sábados, prazo e limite de empréstimos e avaliação após três meses. Repita com PDF textual e DOCX.

O arquivo `.env` é opcional. Para experimentar um modelo mais leve:

```powershell
ollama pull llama3.2:1b
Copy-Item .env.example .env
```

Altere para `OLLAMA_MODEL=llama3.2:1b` e reinicie o Streamlit. Variáveis do ambiente têm prioridade. O modelo 1b pode produzir resumos menos precisos. Não são aceitos modelos com `cloud` no nome.

A POC usa CPU por padrão (`OLLAMA_NUM_GPU=0`). Nesta máquina, a tentativa com GTX 1060 apresentou erro de compatibilidade CUDA/driver. Em um ambiente com GPU compatível, configure `OLLAMA_NUM_GPU=-1` para seleção automática. Não é necessário alterar drivers para usar a POC na CPU.

## Como estudar o código

1. `app.py`: upload e interação; selecionar o arquivo não chama o modelo.
2. `documents.py`: converte os bytes em texto na memória.
3. `summarizer.py`: monta o prompt, chama `ChatOllama` e extrai a resposta.

O operador `|` conecta as etapas em `PROMPT | model | StrOutputParser()`. `chain.invoke({"document": text})` insere o texto no prompt, executa o modelo e devolve uma string. Usamos os pacotes modulares `langchain-core` e `langchain-ollama`, suficientes para esse fluxo.

Textos longos passam por resumos parciais: todas as partes são processadas, os resumos são agrupados e reduzidos em níveis e uma chamada final produz as duas seções. A tela mostra o progresso de cada etapa. O texto original não é cortado para caber em uma única chamada; a consolidação pode perder detalhes, como qualquer resumo.

A extração utiliza `pypdf` para PDF, `python-docx` para parágrafos e tabelas DOCX e leitura direta para TXT UTF-8. O modelo usa temperatura 0. A comunicação com o Ollama fica restrita ao endpoint local, sem proxy de ambiente, e o tracing externo do LangChain está desabilitado.

## Critérios de aceite

- Receber um documento por vez, nos formatos TXT, PDF textual ou DOCX, aceitando arquivo e texto extraído de até 10 MB, inclusive no limite exato.
- Chamar o modelo somente ao clicar em **Gerar resumo**, sem exigir conta ou chave de API.
- Apresentar resumo e até cinco pontos principais em português; processar documentos longos em partes, com progresso por etapa e consolidação final.
- Rejeitar formato inválido, arquivo vazio, corrompido, protegido, sem texto extraível ou acima dos limites antes da inferência.
- Exibir mensagens claras para Ollama indisponível, modelo ausente, timeout, falha de inferência, resposta vazia ou impossibilidade de compactar os resumos intermediários.
- Manter anexos em memória e resultado apenas na sessão; limpar o resumo anterior ao trocar o documento.
- Validar extração, fluxo LangChain e interface com testes automatizados independentes do Ollama.

## Testes

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Os testes geram documentos em memória e simulam o modelo; não precisam de Ollama ou de baixar modelos. A qualidade real deve ser conferida pelo roteiro manual.

A cobertura inclui os três formatos, Unicode e tabelas; documentos inválidos, vazios, corrompidos, protegidos ou sem texto; limites de arquivo e texto, inclusive 10 MB; divisão sem perda de conteúdo; consolidação em múltiplos níveis e progresso; redução sem convergência; resposta vazia; falhas do Ollama; configuração de CPU/GPU e interação da tela.

## Limites e problemas comuns

- Um arquivo por vez, até 10 MB (10.000.000 bytes); TXT UTF-8 com ou sem BOM. O texto extraído também pode ter até 10 MB.
- Cada chamada recebe até 6.000 bytes de conteúdo, preservando Unicode e respeitando o contexto de 8.192 tokens do modelo padrão. Textos maiores são processados em partes com consolidação hierárquica. Saída de até 800 tokens por chamada.
- Arquivos perto de 10 MB podem exigir milhares de chamadas locais e levar horas, especialmente na CPU. O limite permite processá-los, mas não garante geração rápida.
- PDF textual sem OCR. Em PDFs mistos, páginas de imagem podem não fornecer texto.
- DOCX: parágrafos e tabelas do corpo, sem imagens, cabeçalhos ou rodapés.
- Anexos não são gravados no disco; resultado somente na sessão Streamlit. O modelo permanece carregado por até cinco minutos de inatividade para acelerar as chamadas das partes.
- Ollama indisponível: abra o aplicativo ou execute `ollama serve`.
- Modelo ausente: execute `ollama pull` com o nome configurado em `OLLAMA_MODEL`.
- Lentidão ou falta de memória: feche aplicativos, reduza o documento ou experimente `llama3.2:1b`. Prazo da chamada: 300 segundos.

## Referências

- [ChatOllama no LangChain](https://docs.langchain.com/oss/python/integrations/chat/ollama)
- [Ollama no Windows](https://docs.ollama.com/windows)
- [Modelo Llama 3.2](https://ollama.com/library/llama3.2)
- [Testes Streamlit](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)
