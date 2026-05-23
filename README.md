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
│   └── processed/          # Dados após split e transformações (ignorados no Git)
│
├── notebooks/
│   ├── 01_eda.ipynb        # Análise Exploratória de Dados (EDA)
│   ├── 02_modelagem.ipynb  # Treinamento e comparação de modelos
│   └── 03_interpretabilidade.ipynb # Explicabilidade (SHAP / Feature Importance)
│
├── src/
│   ├── __init__.py         # Inicialização do pacote python
│   ├── data_loader.py      # Script de carregamento automatizado do dataset (com fallback)
│   ├── preprocessing.py    # Pré-processamento e split treino/teste estratificado
│   ├── features.py         # Engenharia de atributos (etapas futuras)
│   ├── train.py            # Treinamento de modelos (etapas futuras)
│   ├── evaluate.py         # Avaliação e geração de relatórios de métricas (etapas futuras)
│   ├── plots.py            # Geração de gráficos corporativos e relatórios visuais (etapas futuras)
│   └── validate_bootstrap.py # Validador automático da pipeline inicial de bootstrap
│
├── results/
│   ├── figures/            # Gráficos e visualizações salvas para a apresentação
│   ├── metrics/            # Arquivos JSON/CSV com resultados comparativos
│   └── models/             # Serialização dos modelos treinados (ex: .pkl, .joblib)
│
├── presentation/
│   └── README.md           # Planejamento e base para a futura apresentação executiva
│
├── .gitignore              # Configurações de arquivos ignorados pelo git
├── requirements.txt        # Dependências do projeto
└── README.md               # Documentação inicial do projeto (este arquivo)
```

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
1. **EDA Completa**: Análise detalhada de outliers, correlações e distribuições no notebook `01_eda.ipynb`.
2. **Engenharia de Atributos**: Implementação de variáveis de negócio no arquivo `src/features.py` (ex: `sulfur_ratio`, `acidity_balance`, `alcohol_density_ratio`).
3. **Modelagem de Algoritmos**: Implementação de classificadores base (Regressão Logística) e avançados (Random Forest, XGBoost) no notebook `02_modelagem.ipynb` e `src/train.py`.
4. **Explicabilidade**: Uso de SHAP para entender as decisões do modelo (`notebooks/03_interpretabilidade.ipynb`).
5. **Apresentação de Storytelling**: Futura interface HTML/JS integrada para comunicação de resultados corporativos.
