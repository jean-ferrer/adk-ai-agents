# Agent Development Kit (ADK) AI Agents

Este projeto demonstra uma pipeline de Machine Learning automatizada. Ele utiliza múltiplos agentes de IA (construídos com o Google ADK e Gemini) para, a partir de uma simples pergunta do usuário (query), criar de forma autônoma um modelo de **classificação ou regressão**.

Como exemplo padrão, o projeto utiliza os [microdados do Censo Escolar de 2024](https://download.inep.gov.br/dados_abertos/microdados_censo_escolar_2024.zip), um dataset complexo com mais de 400 colunas e 200 mil linhas.

## 🔄 Workflow

A pipeline é flexível e pode construir tanto modelos de classificação quanto de regressão, dependendo do objetivo definido na query. Ela é orquestrada por um agente principal que gerencia um ciclo de trabalho entre três agentes especializados:

1.  **`DataEngineerAgent`**: Explora os arquivos de dados nas pastas e subpastas do projeto, utiliza os dicionários para entender as variáveis e seleciona o target e as features mais relevantes para o objetivo.
2.  **`DataScientistAgent`**: Carrega os dados, limpa, pré-processa, treina um modelo XGBoost (seja `XGBClassifier` ou `XGBRegressor`), avalia seu desempenho e, se necessário, realiza a otimização de hiperparâmetros.
3.  **`CritiqueAgent`**: Analisa as métricas do modelo (`F1-Score` para classificação, `R-squared` para regressão). Se o desempenho for satisfatório, o processo é finalizado. Caso contrário, ele instrui os outros agentes a tentarem uma nova abordagem, como uma nova feature engineering, outro processamento de dados ou uma busca de hiperparâmetros automática, iniciando um novo ciclo.

Ao final do processo, o melhor modelo e seus metadados são salvos localmente.

## 🛠️ Instalação

### 1\. Pré-requisitos

  - Python 3.9+
  - Uma chave de API do Google Gemini. Você pode obter uma no [Google AI Studio](https://aistudio.google.com/app/apikey).

### 2\. Instalar

**a. Clone o repositório:**

```bash
git clone https://github.com/jean-ferrer/adk-gov-ai-agents.git
cd adk-gov-ai-agents
```

**b. Crie um ambiente virtual e instale as dependências:**

```bash
# Crie e ative o ambiente virtual
python -m venv .venv
.venv\Scripts\activate # Windows
source .venv/bin/activate # Linux

# Instale as bibliotecas necessárias
pip install -r requirements.txt
```

**c. Configure suas credenciais:**
Crie um arquivo chamado `.env` na pasta raiz do projeto. Este arquivo guardará suas variáveis de ambiente de forma segura. Adicione o seguinte conteúdo a ele:

```env
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_API_KEY='SUA_API_KEY_AQUI'
```

**Importante:** Substitua `'SUA_API_KEY_AQUI'` pela chave de API que você obteve do Google AI Studio.

### 3\. Executar

Você pode executar a pipeline de duas maneiras:

**Opção A: Via Terminal (Script Python)**

Para iniciar a pipeline diretamente, execute o script principal no seu terminal:

```bash
python ADK_agents.py
```

O script irá baixar e extrair os dados automaticamente e iniciará a interação dos agentes, exibindo todo o progresso no console.

**Opção B: Via Jupyter Notebook**

Se preferir uma abordagem mais interativa, abra o arquivo `ADK_agents.ipynb` em um ambiente Jupyter (como o VS Code, Jupyter Lab ou Google Colab). Execute cada célula em ordem para inspecionar os passos e as saídas de cada agente de forma detalhada.

## ⚙️ Customização

Você pode facilmente adaptar o projeto para outros objetivos ou configurações. Todas as principais variáveis estão centralizadas na seção `### === Definições Iniciais === ###` do script.

#### Alterando a Pergunta (Objetivo do Modelo)

A variável `INITIAL_QUERY` define a tarefa que os agentes irão executar. Ao alterar o objetivo, você pode instruí-los a criar tanto um modelo de classificação quanto um de regressão.

**Exemplo de Classificação (prever uma categoria):**

```python
# Query do usuário
INITIAL_QUERY = (
    f"Verifique os dados contidos na pasta '{DATA_DIR}'."
    "O objetivo é prever se uma escola possui internet."  # <-- Objetivo de Classificação
)
```

**Exemplo de Regressão (prever um número):**

```python
# Query do usuário
INITIAL_QUERY = (
    f"Verifique os dados contidos na pasta '{DATA_DIR}'."
    "O objetivo é prever a relação/proporção de alunos por docente em cada escola."  # <-- Objetivo de Regressão
)
```

#### Outras Configurações

Você também pode ajustar outros parâmetros, como:

  - `GEMINI_MODEL`: Para testar outros modelos da família Gemini (ex: `'gemini-2.5-flash'`).
  - `URL`: Para apontar para um dataset diferente.
  - `MAX_ITERATIONS`: Para controlar o número máximo de ciclos de melhoria que os agentes podem executar.

## 📁 Resultados

Após a execução bem-sucedida, os seguintes artefatos serão salvos em uma nova pasta chamada `trained_model_artifacts/`:

  - **`xgb_..._model.json`**: O modelo XGBoost treinado.
  - **`model_metadata.json`**: Um arquivo com os metadados do modelo, incluindo as features utilizadas e os hiperparâmetros.
