import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)
from typing import Dict, Any, List


def evaluate_classifier(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    model_name: str = "modelo",
) -> Dict[str, Any]:
    """
    Avalia as previsões do classificador utilizando métricas apropriadas para classificação binária.
    A métrica principal de sucesso é o F1-Score da Classe 1 (Vinhos de Alta Qualidade).

    Parameters
    ----------
    y_true : pd.Series
        Valores reais do target (0 ou 1).
    y_pred : np.ndarray
        Previsões do modelo (0 ou 1).
    y_prob : np.ndarray, default=None
        Probabilidade predita da classe positiva (alta qualidade).
    model_name : str, default="modelo"
        Nome do modelo para exibição nos logs.

    Returns
    -------
    Dict[str, Any]
        Dicionário contendo todas as métricas calculadas.
    """
    metrics = {}

    # Métrica Principal: F1-Score da classe positiva
    metrics["f1_score_class_1"] = float(f1_score(y_true, y_pred, pos_label=1))

    # Métricas Auxiliares
    metrics["precision_class_1"] = float(precision_score(y_true, y_pred, pos_label=1))
    metrics["recall_class_1"] = float(recall_score(y_true, y_pred, pos_label=1))
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None

    # Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # Classification Report completo (como dicionário)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics["classification_report"] = report

    # Log estruturado
    print(f"\n--- Relatório de Avaliação: {model_name} ---")
    print(f"  Métrica Principal (F1-Score Classe 1): {metrics['f1_score_class_1']:.4f}")
    print(f"  Precision (Classe 1):                  {metrics['precision_class_1']:.4f}")
    print(f"  Recall (Classe 1):                     {metrics['recall_class_1']:.4f}")
    print(f"  Acurácia Geral:                        {metrics['accuracy']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"  ROC-AUC:                               {metrics['roc_auc']:.4f}")
    print(f"\n  Matriz de Confusão:")
    print(f"    {cm}")

    return metrics


def compare_models(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Compara métricas de múltiplos modelos e identifica o melhor pelo F1-Score da Classe 1.

    Parameters
    ----------
    results_dict : Dict[str, Dict[str, Any]]
        Dicionário {model_name: metrics_dict} retornado por evaluate_classifier.

    Returns
    -------
    pd.DataFrame
        DataFrame comparativo ordenado pelo F1-Score da Classe 1 (decrescente).
    """
    rows = []
    for model_name, metrics in results_dict.items():
        rows.append({
            "Modelo": model_name,
            "F1-Score (Classe 1)": metrics["f1_score_class_1"],
            "Precision (Classe 1)": metrics["precision_class_1"],
            "Recall (Classe 1)": metrics["recall_class_1"],
            "ROC-AUC": metrics.get("roc_auc"),
            "Acurácia": metrics["accuracy"],
        })

    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.sort_values("F1-Score (Classe 1)", ascending=False).reset_index(drop=True)

    # Identifica o melhor modelo
    best_model = comparison_df.iloc[0]["Modelo"]

    print("\n" + "=" * 70)
    print(" COMPARAÇÃO CONSOLIDADA DE MODELOS")
    print("=" * 70)
    print(comparison_df.to_string(index=False))
    print(f"\n [*] Melhor Modelo (por F1-Score Classe 1): {best_model}")
    print("=" * 70)

    return comparison_df


def save_evaluation_results(
    results_dict: Dict[str, Dict[str, Any]],
    comparison_df: pd.DataFrame,
    output_dir: str,
) -> Dict[str, str]:
    """
    Persiste métricas de avaliação em JSON e CSV para auditoria.

    Parameters
    ----------
    results_dict : Dict[str, Dict[str, Any]]
        Métricas individuais por modelo.
    comparison_df : pd.DataFrame
        DataFrame comparativo gerado por compare_models.
    output_dir : str
        Diretório de destino (results/metrics/).

    Returns
    -------
    Dict[str, str]
        Mapeamento de arquivos salvos.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = {}

    # 1. Métricas individuais por modelo (JSON)
    for model_name, metrics in results_dict.items():
        # Cria cópia serializável (sem arrays numpy)
        serializable = {}
        for k, v in metrics.items():
            if k == "classification_report":
                serializable[k] = v
            elif isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            else:
                serializable[k] = v

        safe_name = model_name.lower().replace(" ", "_")
        filepath = os.path.join(output_dir, f"model_evaluation_{safe_name}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=4, ensure_ascii=False)
        saved_files[f"evaluation_{safe_name}"] = filepath
        print(f" - Salvo: {filepath}")

    # 2. Tabela comparativa CSV
    csv_path = os.path.join(output_dir, "model_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    saved_files["comparison_csv"] = csv_path
    print(f" - Salvo: {csv_path}")

    # 3. Tabela comparativa JSON
    json_path = os.path.join(output_dir, "model_comparison.json")
    comparison_df.to_json(json_path, orient="records", indent=4, force_ascii=False)
    saved_files["comparison_json"] = json_path
    print(f" - Salvo: {json_path}")

    # 4. Resumo do melhor modelo
    best_row = comparison_df.iloc[0]
    best_name = best_row["Modelo"]
    best_summary = {
        "best_model": best_name,
        "selection_criterion": "F1-Score da Classe 1 (vinhos de alta qualidade)",
        "f1_score_class_1": float(best_row["F1-Score (Classe 1)"]),
        "precision_class_1": float(best_row["Precision (Classe 1)"]),
        "recall_class_1": float(best_row["Recall (Classe 1)"]),
        "roc_auc": float(best_row["ROC-AUC"]) if best_row["ROC-AUC"] is not None else None,
        "accuracy": float(best_row["Acurácia"]),
    }
    best_path = os.path.join(output_dir, "best_model_summary.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_summary, f, indent=4, ensure_ascii=False)
    saved_files["best_model_summary"] = best_path
    print(f" - Salvo: {best_path}")

    return saved_files
