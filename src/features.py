import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Função para engenharia de atributos (Feature Engineering) para o Wine Quality Dataset.
    Implementará variáveis físico-químicas combinadas de acordo com as especificações de negócio.
    
    Novos atributos sugeridos:
    1. sulfur_ratio = free sulfur dioxide / total sulfur dioxide
    2. acidity_balance = fixed acidity - volatile acidity
    3. alcohol_density_ratio = alcohol / density
    4. sugar_alcohol_ratio = residual sugar / alcohol
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame contendo os atributos físico-químicos originais.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame contendo os atributos originais e os novos atributos gerados.
    """
    df_copy = df.copy()
    
    print("Iniciando Engenharia de Atributos...")
    
    # 1. Sulfur Ratio
    if "free sulfur dioxide" in df_copy.columns and "total sulfur dioxide" in df_copy.columns:
        df_copy["sulfur_ratio"] = df_copy["free sulfur dioxide"] / df_copy["total sulfur dioxide"]
        print(" - Novo atributo gerado: 'sulfur_ratio'")
        
    # 2. Acidity Balance
    if "fixed acidity" in df_copy.columns and "volatile acidity" in df_copy.columns:
        df_copy["acidity_balance"] = df_copy["fixed acidity"] - df_copy["volatile acidity"]
        print(" - Novo atributo gerado: 'acidity_balance'")
        
    # 3. Alcohol Density Ratio
    if "alcohol" in df_copy.columns and "density" in df_copy.columns:
        df_copy["alcohol_density_ratio"] = df_copy["alcohol"] / df_copy["density"]
        print(" - Novo atributo gerado: 'alcohol_density_ratio'")
        
    # 4. Sugar Alcohol Ratio
    if "residual sugar" in df_copy.columns and "alcohol" in df_copy.columns:
        df_copy["sugar_alcohol_ratio"] = df_copy["residual sugar"] / df_copy["alcohol"]
        print(" - Novo atributo gerado: 'sugar_alcohol_ratio'")
        
    print("Engenharia de Atributos concluída com sucesso.")
    return df_copy
