import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from typing import Dict, Any, Tuple

from src.preprocessing import create_preprocessing_pipeline


# Registro de configurações padrão por modelo
MODEL_CONFIGS = {
    "logistic_regression": {
        "display_name": "Regressão Logística",
        "scaler": "standard",
        "description": "Baseline linear interpretável. Usa StandardScaler e class_weight='balanced'.",
    },
    "random_forest": {
        "display_name": "Random Forest",
        "scaler": None,
        "description": "Ensemble de árvores de decisão. Invariante a escala, usa class_weight='balanced'.",
    },
    "gradient_boosting": {
        "display_name": "Gradient Boosting (Hist)",
        "scaler": None,
        "description": "Boosting sequencial nativo do scikit-learn. Alto desempenho em dados tabulares.",
    },
}


def _create_classifier(model_name: str):
    """
    Instancia o classificador scikit-learn conforme o nome do modelo.

    Parameters
    ----------
    model_name : str
        Identificador do modelo ('logistic_regression', 'random_forest', 'gradient_boosting').

    Returns
    -------
    Estimator scikit-learn
    """
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
            solver="lbfgs",
        )
    elif model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
    elif model_name == "gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=200,
            random_state=42,
            class_weight="balanced",
        )
    else:
        raise ValueError(f"Modelo '{model_name}' não suportado. Opções: {list(MODEL_CONFIGS.keys())}")


def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[Pipeline, str]:
    """
    Treina um modelo de classificação binária dentro de um pipeline scikit-learn
    que inclui imputação e escalonamento adequados ao tipo de modelo.

    O pipeline garante que o fit ocorre apenas nos dados de treino, evitando data leakage.

    Parameters
    ----------
    model_name : str
        Identificador do modelo ('logistic_regression', 'random_forest', 'gradient_boosting').
    X_train : pd.DataFrame
        Features de treino.
    y_train : pd.Series
        Target de treino.

    Returns
    -------
    Tuple[Pipeline, str]
        Pipeline treinado e nome de exibição do modelo.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Modelo '{model_name}' não suportado. Opções: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_name]
    display_name = config["display_name"]

    print(f"\n{'='*60}")
    print(f" Treinando: {display_name}")
    print(f" Scaler: {config['scaler'] or 'Nenhum (invariante a escala)'}")
    print(f"{'='*60}")

    # Cria o pipeline de pré-processamento (imputação + scaler)
    preprocessing_pipeline = create_preprocessing_pipeline(scaler=config["scaler"])

    # Cria o classificador
    classifier = _create_classifier(model_name)

    # Monta o pipeline completo: pre-processamento -> classificador
    full_pipeline = Pipeline([
        ("preprocessing", preprocessing_pipeline),
        ("classifier", classifier),
    ])

    # Treina o pipeline completo (fit ocorre APENAS no treino)
    full_pipeline.fit(X_train, y_train)

    print(f" [OK] {display_name} treinado com sucesso.")
    print(f"   - Amostras de treino: {X_train.shape[0]}")
    print(f"   - Features utilizadas: {X_train.shape[1]}")

    return full_pipeline, display_name


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """Retorna o registro de modelos disponíveis com suas configurações."""
    return MODEL_CONFIGS.copy()
