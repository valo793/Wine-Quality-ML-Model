import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from typing import Dict, Any

def evaluate_classifier(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, Any]:
    """
    Avalia as previsões do classificador utilizando métricas apropriadas para classificação binária.
    A métrica principal de sucesso é o F1-Score da Classe 1 (Vinhos de Alta Qualidade).
    
    Parameters:
    -----------
    y_true : pd.Series
        Valores reais do target (0 ou 1).
    y_pred : np.ndarray
        Previsões do modelo (0 ou 1).
    y_prob : np.ndarray, default=None
        Probabilidade predita da classe positiva (alta qualidade).
        
    Returns:
    --------
    Dict[str, Any]
        Dicionário contendo as métricas calculadas.
    """
    metrics = {}
    
    # Métrica Principal: F1-Score da classe positiva
    metrics["f1_score_class_1"] = f1_score(y_true, y_pred, pos_label=1)
    
    # Métricas Auxiliares
    metrics["precision_class_1"] = precision_score(y_true, y_pred, pos_label=1)
    metrics["recall_class_1"] = recall_score(y_true, y_pred, pos_label=1)
    metrics["accuracy"] = np.mean(y_true == y_pred)
    
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        
    # Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    
    print("\n--- Relatório de Avaliação ---")
    print(f"Métrica Principal (F1-Score Classe 1): {metrics['f1_score_class_1']:.4f}")
    print(f"Precision (Classe 1): {metrics['precision_class_1']:.4f}")
    print(f"Recall (Classe 1): {metrics['recall_class_1']:.4f}")
    print(f"Acurácia Geral: {metrics['accuracy']:.4f}")
    if y_prob is not None:
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        
    print("\nMatriz de Confusão:")
    print(cm)
    
    return metrics
