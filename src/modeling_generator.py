"""
Orquestrador da Fase 4–5: Desenvolvimento e Avaliação de Modelos de Classificação.

Este script executa toda a pipeline de modelagem:
1. Carregamento dos dados e binarização da variável alvo
2. Engenharia de atributos (4 features físico-químicas)
3. Divisão estratificada treino/teste
4. Treinamento de 3 modelos de classificação
5. Avaliação com F1-Score da Classe 1 como métrica principal
6. Comparação consolidada e seleção do melhor modelo
7. Geração de gráficos e relatórios de auditoria
8. Serialização do melhor modelo

Execução:
    .\\.venv\\Scripts\\python.exe -m src.modeling_generator
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

# Adiciona o diretório raiz ao path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.data_loader import load_wine_data
from src.preprocessing import create_target_variable, split_data
from src.features import engineer_features
from src.train import train_model, get_available_models
from src.evaluate import evaluate_classifier, compare_models, save_evaluation_results
from src.plots import (
    plot_confusion_matrices,
    plot_roc_curves,
    plot_model_comparison_bar,
    plot_feature_importance,
)


def generate_modeling_report(
    comparison_df: pd.DataFrame,
    results_dict: dict,
    model_configs: dict,
    feature_names: list,
    train_size: int,
    test_size: int,
    report_path: str,
):
    """
    Gera o relatório executivo de modelagem em Markdown.
    """
    best_row = comparison_df.iloc[0]
    best_name = best_row["Modelo"]

    # Monta seções de métricas individuais
    model_sections = []
    for model_name, metrics in results_dict.items():
        cm = np.array(metrics["confusion_matrix"])
        config = model_configs.get(model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(" ", "_"), {})

        section = f"""### {model_name}

| Métrica | Valor |
|---|---|
| **F1-Score (Classe 1)** | **{metrics['f1_score_class_1']:.4f}** |
| Precision (Classe 1) | {metrics['precision_class_1']:.4f} |
| Recall (Classe 1) | {metrics['recall_class_1']:.4f} |
| ROC-AUC | {f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] is not None else 'N/A'} |
| Acurácia | {metrics['accuracy']:.4f} |

**Matriz de Confusão:**

|  | Pred. 0 | Pred. 1 |
|---|---|---|
| **Real 0** | {cm[0][0]} | {cm[0][1]} |
| **Real 1** | {cm[1][0]} | {cm[1][1]} |
"""
        model_sections.append(section)

    model_sections_str = "\n---\n\n".join(model_sections)

    # Monta tabela comparativa
    comparison_table = comparison_df.to_markdown(index=False)

    report_content = f"""# Relatório de Modelagem — Classificação de Qualidade de Vinhos

## Contexto

Este relatório documenta os resultados da **Fase 4–5 (Desenvolvimento e Avaliação de Modelos)** do Tech Challenge.

- **Objetivo**: Classificação binária de vinhos em Alta Qualidade (quality ≥ 7) vs Baixa/Média Qualidade (quality < 7).
- **Métrica Principal**: F1-Score da Classe 1 (vinhos de alta qualidade).
- **Amostras de Treino**: {train_size}
- **Amostras de Teste**: {test_size}
- **Features utilizadas**: {len(feature_names)} ({', '.join(feature_names[:5])}{'...' if len(feature_names) > 5 else ''})

---

## Modelos Treinados

Três modelos foram treinados e avaliados no conjunto de teste:

1. **Regressão Logística** — Baseline linear com StandardScaler e `class_weight="balanced"`.
2. **Random Forest** — Ensemble de 200 árvores, sem scaler, com `class_weight="balanced"`.
3. **Gradient Boosting (Hist)** — Boosting sequencial nativo do scikit-learn, com `class_weight="balanced"`.

---

## Resultados Individuais

{model_sections_str}

---

## Comparação Consolidada

{comparison_table}

---

## Melhor Modelo Selecionado

> **{best_name}** foi selecionado como o melhor modelo com base no F1-Score da Classe 1 = **{best_row['F1-Score (Classe 1)']:.4f}**.

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
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\n Relatorio de modelagem salvo: {report_path}")


def main():
    print("=" * 65)
    print(" INICIANDO PIPELINE DE MODELAGEM E AVALIACAO (FASES 4-5)")
    print("=" * 65)

    # Caminhos
    root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    figures_dir = os.path.join(root_dir, "results", "figures")
    metrics_dir = os.path.join(root_dir, "results", "metrics")
    models_dir = os.path.join(root_dir, "results", "models")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    try:
        # =====================================================================
        # ETAPA 1: PREPARACAO DOS DADOS (Reutiliza Fases 1-3)
        # =====================================================================
        print("\n[MODELO] Etapa 1: Carregando e preparando dados...")
        df_raw = load_wine_data()
        df_target = create_target_variable(df_raw)
        df_engineered = engineer_features(df_target)

        id_cols = [c for c in ["Id", "id", "ID"] if c in df_engineered.columns]
        X_train, X_test, y_train, y_test = split_data(
            df_engineered, target_col="high_quality", drop_cols=id_cols,
            test_size=0.2, random_state=42,
        )

        feature_names = list(X_train.columns)
        print(f"\n  [OK] Dados preparados: {X_train.shape[0]} treino, {X_test.shape[0]} teste, {len(feature_names)} features.")

        # =====================================================================
        # ETAPA 2: TREINAMENTO DOS MODELOS
        # =====================================================================
        print("\n[MODELO] Etapa 2: Treinando modelos...")

        models_to_train = ["logistic_regression", "random_forest", "gradient_boosting"]
        trained_pipelines = {}   # {display_name: pipeline}
        model_configs = get_available_models()

        for model_name in models_to_train:
            pipeline, display_name = train_model(model_name, X_train, y_train)
            trained_pipelines[display_name] = {
                "pipeline": pipeline,
                "model_key": model_name,
            }

        # =====================================================================
        # ETAPA 3: AVALIAÇÃO DOS MODELOS
        # =====================================================================
        print("\n[MODELO] Etapa 3: Avaliando modelos no conjunto de teste...")

        results_dict = {}  # {display_name: metrics}
        roc_data = {}      # {display_name: {fpr, tpr, auc}}

        for display_name, info in trained_pipelines.items():
            pipeline = info["pipeline"]

            # Predições
            y_pred = pipeline.predict(X_test)

            # Probabilidades (para ROC e AUC)
            if hasattr(pipeline, "predict_proba"):
                y_prob = pipeline.predict_proba(X_test)[:, 1]
            elif hasattr(pipeline, "decision_function"):
                y_prob = pipeline.decision_function(X_test)
            else:
                y_prob = None

            # Avaliação
            metrics = evaluate_classifier(y_test, y_pred, y_prob, model_name=display_name)
            results_dict[display_name] = metrics

            # Dados para curva ROC
            if y_prob is not None:
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_data[display_name] = {
                    "fpr": fpr,
                    "tpr": tpr,
                    "auc": metrics["roc_auc"],
                }

        # =====================================================================
        # ETAPA 4: COMPARAÇÃO CONSOLIDADA
        # =====================================================================
        print("\n[MODELO] Etapa 4: Comparacao consolidada...")
        comparison_df = compare_models(results_dict)

        # =====================================================================
        # ETAPA 5: GERAÇÃO DE GRÁFICOS
        # =====================================================================
        print("\n[MODELO] Etapa 5: Gerando graficos de modelagem...")

        # 5.1 Matrizes de Confusão
        plot_confusion_matrices(
            results_dict,
            os.path.join(figures_dir, "confusion_matrices.png"),
        )

        # 5.2 Curvas ROC
        if roc_data:
            plot_roc_curves(
                roc_data,
                os.path.join(figures_dir, "roc_curves.png"),
            )

        # 5.3 Comparação de métricas
        plot_model_comparison_bar(
            comparison_df,
            os.path.join(figures_dir, "model_comparison.png"),
        )

        # 5.4 Feature Importance (para modelos de árvore)
        for display_name, info in trained_pipelines.items():
            pipeline = info["pipeline"]
            classifier = pipeline.named_steps["classifier"]

            if hasattr(classifier, "feature_importances_"):
                safe_name = info["model_key"].replace("_", " ").title().replace(" ", "")
                short_name = info["model_key"].split("_")[0]  # rf, gradient, etc.
                plot_feature_importance(
                    importances=classifier.feature_importances_,
                    feature_names=feature_names,
                    model_name=display_name,
                    save_path=os.path.join(figures_dir, f"feature_importance_{short_name}.png"),
                )

        # =====================================================================
        # ETAPA 6: PERSISTÊNCIA DE MÉTRICAS
        # =====================================================================
        print("\n[MODELO] Etapa 6: Salvando metricas de avaliacao...")
        save_evaluation_results(results_dict, comparison_df, metrics_dir)

        # =====================================================================
        # ETAPA 7: SERIALIZAÇÃO DO MELHOR MODELO
        # =====================================================================
        print("\n[MODELO] Etapa 7: Serializando o melhor modelo...")
        best_model_name = comparison_df.iloc[0]["Modelo"]
        best_pipeline = trained_pipelines[best_model_name]["pipeline"]
        model_path = os.path.join(models_dir, "best_model.pkl")
        joblib.dump(best_pipeline, model_path)
        print(f"  [OK] Melhor modelo ({best_model_name}) salvo em: {model_path}")

        # =====================================================================
        # ETAPA 8: RELATÓRIO DE MODELAGEM
        # =====================================================================
        print("\n[MODELO] Etapa 8: Gerando relatorio de modelagem...")
        report_path = os.path.join(root_dir, "results", "modeling_report.md")
        generate_modeling_report(
            comparison_df=comparison_df,
            results_dict=results_dict,
            model_configs=model_configs,
            feature_names=feature_names,
            train_size=X_train.shape[0],
            test_size=X_test.shape[0],
            report_path=report_path,
        )

        # =====================================================================
        # RESUMO FINAL
        # =====================================================================
        print("\n" + "=" * 65)
        print(" PIPELINE DE MODELAGEM E AVALIACAO CONCLUIDA COM SUCESSO")
        print("=" * 65)
        print(f"\n  [*] Melhor Modelo: {best_model_name}")
        print(f"  [*] F1-Score (Classe 1): {comparison_df.iloc[0]['F1-Score (Classe 1)']:.4f}")
        print(f"  [*] Modelo serializado: {model_path}")
        print(f"  [*] Relatorio: {report_path}")
        print(f"\n  Graficos em: {figures_dir}")
        print(f"  Metricas em: {metrics_dir}")

    except Exception as e:
        print(f"\n[ERRO CRITICO] Falha durante a pipeline de modelagem: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
