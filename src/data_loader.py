import os
import glob
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

def load_wine_data(file_path: str = "") -> pd.DataFrame:
    """
    Carrega o dataset de qualidade de vinhos do Kaggle utilizando kagglehub.
    Contém um fallback robusto caso a chamada direta com file_path falhe ou seja vazia.
    
    Parameters:
    -----------
    file_path : str, default=""
        Caminho do arquivo específico dentro do dataset a ser carregado (opcional).
        
    Returns:
    --------
    pd.DataFrame
        DataFrame carregado com os dados do dataset.
    """
    # Se file_path foi especificado, tentamos carregá-lo diretamente usando o adaptador oficial
    if file_path:
        try:
            print(f"Tentando carregar usando KaggleDatasetAdapter com file_path='{file_path}'...")
            df = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                "yasserh/wine-quality-dataset",
                file_path,
            )
            print("Dataset carregado com sucesso via KaggleDatasetAdapter.")
            return df
        except Exception as e:
            print(f"Falha ao carregar com file_path especificado: {e}. Iniciando fallback...")

    # Fallback robusto: Baixa o diretório completo e procura por arquivos CSV
    try:
        print("Iniciando download do dataset completo via kagglehub...")
        # Baixa a versão mais recente e retorna o path da pasta local
        path = kagglehub.dataset_download("yasserh/wine-quality-dataset")
        print(f"Dataset baixado localmente em: {path}")
        
        # Procura arquivos .csv no diretório baixado
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        if not csv_files:
            # Procura recursivamente se necessário
            csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
            
        if not csv_files:
            raise FileNotFoundError(f"Nenhum arquivo CSV encontrado na pasta {path}")
            
        selected_csv = csv_files[0]
        print(f"Arquivo CSV identificado para carregamento: {selected_csv}")
        df = pd.read_csv(selected_csv)
        print("Dataset carregado com sucesso via pandas (fallback).")
        return df
    except Exception as fallback_error:
        print(f"Erro crítico no fallback do carregamento do dataset: {fallback_error}")
        raise fallback_error
