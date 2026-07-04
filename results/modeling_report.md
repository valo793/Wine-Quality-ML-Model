# Relatório de Modelagem — Classificação de Qualidade de Vinhos

## Contexto

Este relatório documenta os resultados da **Fase 4–5 (Desenvolvimento e Avaliação de Modelos)** do Tech Challenge.

- **Objetivo**: Classificação binária de vinhos em Alta Qualidade (quality ≥ 7) vs Baixa/Média Qualidade (quality < 7).
- **Métrica Principal**: F1-Score da Classe 1 (vinhos de alta qualidade).
- **Amostras de Treino**: 914
- **Amostras de Teste**: 229
- **Features utilizadas**: 15 (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides...)

---

## Modelos Treinados

Três modelos foram treinados e avaliados no conjunto de teste:

1. **Regressão Logística** — Baseline linear com StandardScaler e `class_weight="balanced"`.
2. **Random Forest** — Ensemble de 200 árvores, sem scaler, com `class_weight="balanced"`.
3. **Gradient Boosting (Hist)** — Boosting sequencial nativo do scikit-learn, com `class_weight="balanced"`.

---

## Resultados Individuais

### Regressão Logística

| Métrica | Valor |
|---|---|
| **F1-Score (Classe 1)** | **0.4667** |
| Precision (Classe 1) | 0.3621 |
| Recall (Classe 1) | 0.6562 |
| ROC-AUC | 0.8628 |
| Acurácia | 0.7904 |

**Matriz de Confusão:**

|  | Pred. 0 | Pred. 1 |
|---|---|---|
| **Real 0** | 160 | 37 |
| **Real 1** | 11 | 21 |

---

### Random Forest

| Métrica | Valor |
|---|---|
| **F1-Score (Classe 1)** | **0.5600** |
| Precision (Classe 1) | 0.7778 |
| Recall (Classe 1) | 0.4375 |
| ROC-AUC | 0.9116 |
| Acurácia | 0.9039 |

**Matriz de Confusão:**

|  | Pred. 0 | Pred. 1 |
|---|---|---|
| **Real 0** | 193 | 4 |
| **Real 1** | 18 | 14 |

---

### Gradient Boosting (Hist)

| Métrica | Valor |
|---|---|
| **F1-Score (Classe 1)** | **0.6774** |
| Precision (Classe 1) | 0.7000 |
| Recall (Classe 1) | 0.6562 |
| ROC-AUC | 0.8967 |
| Acurácia | 0.9127 |

**Matriz de Confusão:**

|  | Pred. 0 | Pred. 1 |
|---|---|---|
| **Real 0** | 188 | 9 |
| **Real 1** | 11 | 21 |


---

## Comparação Consolidada

| Modelo                   |   F1-Score (Classe 1) |   Precision (Classe 1) |   Recall (Classe 1) |   ROC-AUC |   Acurácia |
|:-------------------------|----------------------:|-----------------------:|--------------------:|----------:|-----------:|
| Gradient Boosting (Hist) |              0.677419 |               0.7      |             0.65625 |  0.896732 |   0.912664 |
| Random Forest            |              0.56     |               0.777778 |             0.4375  |  0.911643 |   0.90393  |
| Regressão Logística      |              0.466667 |               0.362069 |             0.65625 |  0.862786 |   0.790393 |

---

## Melhor Modelo Selecionado

> **Gradient Boosting (Hist)** foi selecionado como o melhor modelo com base no F1-Score da Classe 1 = **0.6774**.

A seleção é baseada exclusivamente na métrica principal (F1-Score da Classe 1), que equilibra Precision e Recall para a classe minoritária de vinhos premium — a classe de maior interesse comercial.

---

## Visualizações Geradas

| Gráfico | Arquivo |
|---|---|
| Matrizes de Confusão | `results/figures/confusion_matrices.png` |
| Curvas ROC | `results/figures/roc_curves.png` |
| Comparação de Métricas | `results/figures/model_comparison.png` |
| Feature Importance (RF) | `results/figures/feature_importance_rf.png` |
| Feature Importance (GB) | `results/figures/feature_importance_gb.png` |

---

## Métricas Persistidas

| Arquivo | Descrição |
|---|---|
| `results/metrics/model_comparison.csv` | Tabela comparativa entre modelos |
| `results/metrics/model_comparison.json` | Versão JSON da comparação |
| `results/metrics/best_model_summary.json` | Resumo do modelo vencedor |
| `results/metrics/model_evaluation_*.json` | Métricas detalhadas por modelo |

---

## Modelo Serializado

O melhor modelo foi serializado em `results/models/best_model.pkl` para uso futuro em predições e interpretabilidade (SHAP).

---

## Próximos Passos

- **Fase 6: Interpretação e Storytelling** — Análise de importância de features via SHAP, construção de material executivo e apresentação.
