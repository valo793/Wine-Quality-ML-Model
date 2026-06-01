import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

def create_target_variable(df: pd.DataFrame, source_col: str = "quality", target_col: str = "high_quality") -> pd.DataFrame:
    """
    Cria a variável alvo binária com base no limiar de qualidade especificado:
    - Vinho de Alta Qualidade (1): quality >= 7
    - Vinho de Baixa/Média Qualidade (0): quality < 7
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame original com a coluna de qualidade.
    source_col : str, default="quality"
        Nome da coluna original de qualidade.
    target_col : str, default="high_quality"
        Nome da nova coluna binária de qualidade a ser criada.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame contendo a nova variável alvo.
    """
    if source_col not in df.columns:
        raise KeyError(f"A coluna de qualidade '{source_col}' não foi encontrada no DataFrame.")
        
    # Copia o DataFrame para evitar Side-Effects (Alteraçoes indevidas na base original)
    df_copy = df.copy()
    
    # Cria a variável binária (True/False mapeado para 1/0)
    df_copy[target_col] = (df_copy[source_col] >= 7).astype(int)
    print(f"Variável alvo '{target_col}' criada com sucesso.")
    return df_copy

def split_data(
    df: pd.DataFrame, 
    target_col: str = "high_quality", 
    drop_cols: list = None,
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Separa o DataFrame em features (X) e target (y), e realiza um split estratificado
    em treino e teste para lidar com o desbalanceamento das classes.
    Garante que colunas indesejadas (quality, Id, etc.) sejam removidas das features preditivas.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame contendo as features e a coluna alvo.
    target_col : str, default="high_quality"
        Nome da coluna alvo para classificação.
    drop_cols : list, default=None
        Lista de colunas adicionais para dropar das features (ex: 'quality' original ou IDs).
    test_size : float, default=0.2
        Proporção da base de teste.
    random_state : int, default=42
        Semente aleatória para reprodutibilidade.
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        Retorna X_train, X_test, y_train, y_test.
    """
    if target_col not in df.columns:
        raise KeyError(f"A coluna alvo '{target_col}' não está presente no DataFrame.")
        
    if drop_cols is None:
        drop_cols = []
        
    # Garante que 'quality' (se presente) e variações de 'Id' / 'id' sejam removidas
    default_drops = ["quality", "Id", "id", "ID"]
    cols_to_drop = list(set([target_col] + drop_cols + default_drops))
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    X = df.drop(columns=cols_to_drop)
    y = df[target_col]
    
    print(f"Separando features e alvo. Colunas removidas das features: {cols_to_drop}")
    print(f"Features finais (X): {list(X.columns)}")
    
    # Realiza o split estratificado garantindo que a proporção das classes se mantenha estável
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    print("Divisão treino/teste realizada com amostragem estratificada:")
    print(f" - Treino: {X_train.shape[0]} amostras")
    print(f" - Teste: {X_test.shape[0]} amostras")
    
    return X_train, X_test, y_train, y_test

def create_preprocessing_pipeline(scaler: str = "standard"):
    """
    Cria e retorna um pipeline scikit-learn contendo imputação e escalonamento numérico.
    
    Parameters:
    -----------
    scaler : str, default="standard"
        Tipo de scaler a ser aplicado:
        - "standard": Aplica StandardScaler.
        - "robust": Aplica RobustScaler.
        - None ou "none": Não aplica scaler (apenas imputação).
        
    Returns:
    --------
    Pipeline
        Pipeline scikit-learn estruturado.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, RobustScaler
    
    steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    
    if scaler == "standard":
        steps.append(("scaler", StandardScaler()))
        print("Pipeline de pré-processamento criado com imputador (mediana) e StandardScaler.")
    elif scaler == "robust":
        steps.append(("scaler", RobustScaler()))
        print("Pipeline de pré-processamento criado com imputador (mediana) e RobustScaler.")
    else:
        print("Pipeline de pré-processamento criado com imputador (mediana) e sem Scaler.")
        
    return Pipeline(steps)

