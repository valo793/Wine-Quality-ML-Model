import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Função para engenharia de atributos (Feature Engineering) para o Wine Quality Dataset.
    Implementa variáveis físico-químicas combinadas de acordo com as especificações de negócio.
    Utiliza um valor epsilon de 1e-8 para evitar divisões por zero de forma robusta.
    
    Novos atributos gerados:
    1. sulfur_ratio = free sulfur dioxide / (total sulfur dioxide + epsilon)
       Justificativa: Razão de conservação ativa (livre) em relação à conservação total.
    2. acidity_balance = fixed acidity - volatile acidity
       Justificativa: Equilíbrio entre a acidez estrutural agradável e a acidez acética indesejada.
    3. alcohol_density_ratio = alcohol / (density + epsilon)
       Justificativa: Razão entre o teor alcoólico (menos denso) e a densidade geral do vinho.
    4. sugar_alcohol_ratio = residual sugar / (alcohol + epsilon)
       Justificativa: Relação entre o açúcar residual não fermentado e o teor alcoólico do vinho.
    
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
    epsilon = 1e-8
    
    print("Iniciando Engenharia de Atributos...")
    
    # 1. Sulfur Ratio
    if "free sulfur dioxide" in df_copy.columns and "total sulfur dioxide" in df_copy.columns:
        df_copy["sulfur_ratio"] = df_copy["free sulfur dioxide"] / (df_copy["total sulfur dioxide"] + epsilon)
        print(" - Novo atributo gerado: 'sulfur_ratio'")
        
    # 2. Acidity Balance
    if "fixed acidity" in df_copy.columns and "volatile acidity" in df_copy.columns:
        df_copy["acidity_balance"] = df_copy["fixed acidity"] - df_copy["volatile acidity"]
        print(" - Novo atributo gerado: 'acidity_balance'")
        
    # 3. Alcohol Density Ratio
    if "alcohol" in df_copy.columns and "density" in df_copy.columns:
        df_copy["alcohol_density_ratio"] = df_copy["alcohol"] / (df_copy["density"] + epsilon)
        print(" - Novo atributo gerado: 'alcohol_density_ratio'")
        
    # 4. Sugar Alcohol Ratio
    if "residual sugar" in df_copy.columns and "alcohol" in df_copy.columns:
        df_copy["sugar_alcohol_ratio"] = df_copy["residual sugar"] / (df_copy["alcohol"] + epsilon)
        print(" - Novo atributo gerado: 'sugar_alcohol_ratio'")
        
    print("Engenharia de Atributos concluída com sucesso.")
    return df_copy

