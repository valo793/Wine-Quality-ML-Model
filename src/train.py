import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any

def train_baseline_model(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """
    Treina o modelo de Baseline (Regressão Logística).
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Features de treino.
    y_train : pd.Series
        Target de treino.
        
    Returns:
    --------
    LogisticRegression
        Modelo treinado de Regressão Logística.
    """
    print("Treinando modelo baseline (Regressão Logística)...")
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    print("Baseline treinado com sucesso.")
    return model

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Treina o modelo de Random Forest para classificação.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Features de treino.
    y_train : pd.Series
        Target de treino.
        
    Returns:
    --------
    RandomForestClassifier
        Modelo treinado de Random Forest.
    """
    print("Treinando modelo Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    print("Random Forest treinado com sucesso.")
    return model
