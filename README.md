# Classificação de Qualidade de Vinhos (Wine Quality ML Model)

## Contexto do Projeto
Este projeto é o **Tech Challenge da Pós-Tech em Data Analytics (Fase 2)**. O principal objetivo é desenvolver uma solução robusta e reproduzível de Machine Learning para prever a qualidade de vinhos com base em suas características físico-químicas, a partir do [Wine Quality Dataset do Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset).

A solução é desenhada como um pipeline completo que vai desde o carregamento automatizado dos dados até a modelagem, validação e preparação de insights para apresentações executivas.

---

## Objetivo e Definição do Problema
O problema é tratado como uma **classificação binária**, onde:
* **Vinho de Alta Qualidade (Classe 1)**: `quality >= 7`
* **Vinho de Baixa/Média Qualidade (Classe 0)**: `quality < 7`

### Métrica de Sucesso de Machine Learning
* **Métrica Principal**: **F1-Score da Classe 1 (Vinhos de Alta Qualidade)**.
* **Métricas Auxiliares**: Recall da classe 1, Precision da classe 1, ROC-AUC, Acurácia e Matriz de Confusão.

> [!NOTE]
> **Justificativa da Métrica**: Como os vinhos de alta qualidade representam uma parcela minoritária da base (dados desbalanceados), a **Acurácia** isolada pode ser extremamente enganosa (um modelo ingênuo que sempre prevê a classe majoritária teria acurácia alta, mas utilidade nula). O **F1-Score** equilibra a precisão (garantir que vinhos indicados como de alta qualidade realmente o sejam) e o recall (garantir que não estamos perdendo bons lotes de vinho), oferecendo uma métrica robusta e alinhada com o interesse de negócio.

---

## Estrutura do Repositório

O projeto segue uma arquitetura modular, separando a lógica de negócios da execução interativa em notebooks:

```text
wine-quality-classification/
│
├── data/
│   ├── raw/                # Dados brutos originais (baixados via script, ignorados no Git)
│   │   └── .gitkeep        # Arquivo de rastreio de diretório
│   └── processed/          # Dados após split e transformações (ignorados no Git)
│       └── .gitkeep        # Arquivo de rastreio de diretório
│
├── notebooks/
│   ├── 01_eda.ipynb        # Análise Exploratória de Dados (EDA)
│   ├── 02_modelagem.ipynb  # Treinamento e comparação de modelos (etapas futuras)
│   └── 03_interpretabilidade.ipynb # Explicabilidade (SHAP / Feature Importance) (etapas futuras)
│
├── src/
│   ├── __init__.py         # Inicialização do pacote python
│   ├── data_loader.py      # Script de carregamento automatizado do dataset (com fallback)
│   ├── preprocessing.py    # Pré-processamento e split treino/teste estratificado
│   ├── features.py         # Engenharia de atributos (esqueleto para etapas futuras)
│   ├── train.py            # Treinamento de modelos (esqueleto para etapas futuras)
│   ├── evaluate.py         # Avaliação de métricas (esqueleto com foco em F1-score)
│   ├── plots.py            # Biblioteca de geração de gráficos corporativos (300 DPI)
│   ├── eda_generator.py    # Pipeline completa de execução da EDA e geração de relatórios
│   └── validate_bootstrap.py # Validador automático da pipeline inicial de bootstrap
│
├── results/
│   ├── eda_report.md       # Relatório executivo de EDA consolidado com imagens embutidas
│   ├── figures/            # Gráficos em 300 DPI (quality_distribution.png, target_balance.png, etc.)
│   ├── metrics/            # Estatísticas em JSON (eda_summary.json) e CSVs para auditoria
│   └── models/             # Serialização dos modelos treinados (esqueleto)
│
├── presentation/
│   └── README.md           # Planejamento e base para a futura apresentação executiva
│
├── .gitignore              # Configurações de arquivos ignorados pelo git
├── requirements.txt        # Dependências do projeto
└── README.md               # Documentação do projeto (este arquivo)
```

---

## Fase 2 - Análise Exploratória de Dados (EDA) Concluída
A análise exploratória de dados foi totalmente concluída e validada localmente, produzindo relatórios consolidados e tabelas analíticas para auditoria:
* **Relatório Visual Executivo**: [results/eda_report.md](results/eda_report.md) (com análises estatísticas explicadas sob a ótica de negócios, sem afirmações causais).
* **Gráficos Gerados**: localizados em [results/figures/](results/figures/) (gerados a 300 DPI com paletas corporativas elegantes).
* **Tabelas de Auditoria e Resumo**: disponíveis em [results/metrics/](results/metrics/) (incluindo `eda_summary.json` e arquivos CSV analíticos).
* **Script Principal**: [src/eda_generator.py](src/eda_generator.py) (automatiza a extração de métricas e renderização das imagens).


---

## Como Instalar as Dependências

### Pré-requisitos
* Python instalado (versão 3.9 ou superior recomendada).
* Git configurado.

### Configuração do Ambiente Virtual (Recomendado)

1. Clone o repositório ou navegue até a pasta do projeto:
   ```bash
   git clone https://github.com/valo793/Wine-Quality-ML-Model.git
   cd Wine-Quality-ML-Model
   ```

2. Crie um ambiente virtual:
   ```bash
   python -m venv .venv
   ```

3. Ative o ambiente virtual:
   * **Windows (PowerShell)**:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   * **Windows (CMD)**:
     ```cmd
     .venv\Scripts\activate.bat
     ```
   * **Linux/macOS**:
     ```bash
     source .venv/bin/activate
     ```

4. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

---

## Como Executar o Script de Validação

Para certificar-se de que a estrutura e as etapas do kick-start foram configuradas corretamente e rodam sem erros, execute o validador:

```bash
python -m src.validate_bootstrap
```

Esse validador executará o download do dataset usando `kagglehub` (com fallback robusto), transformará a variável alvo em binária, fará o split treino/teste estratificado e imprimirá as estatísticas e distribuições de classe na tela.

---

## Próximos Passos de Desenvolvimento
1. **Fase 3: Pré-processamento e Feature Engineering**: Implementação de variáveis de negócio no arquivo `src/features.py` (ex: `sulfur_ratio`, `acidity_balance`, `alcohol_density_ratio`, `sugar_alcohol_ratio`).
2. **Fase 4: Modelagem**: Treinamento de pelo menos dois modelos de classificação (ex: Regressão Logística como baseline, Random Forest, XGBoost) no notebook `notebooks/02_modelagem.ipynb` e `src/train.py`.
3. **Fase 5: Avaliação Comparativa**: Comparação sistemática de performance entre os modelos usando métricas estruturadas (`src/evaluate.py`).
4. **Fase 6: Interpretação dos Modelos**: Uso de SHAP e análise de importância de atributos para explicabilidade (`notebooks/03_interpretabilidade.ipynb`).
5. **Fase 7: Storytelling Executivo e Apresentação**: Construção do material dinâmico em HTML/JS integrado para a diretoria, simulador de predições e roteiro para o vídeo de pitch de 5 minutos.

