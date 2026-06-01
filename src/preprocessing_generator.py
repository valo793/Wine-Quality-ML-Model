import os
import sys
import json
import pandas as pd
import numpy as np

# Adiciona o diretório raiz ao path para garantir que imports funcionem robustamente
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.data_loader import load_wine_data
from src.preprocessing import create_target_variable, split_data, create_preprocessing_pipeline
from src.features import engineer_features

def main():
    print("==================================================")
    print(" INICIANDO PIPELINE DE PRÉ-PROCESSAMENTO (FASE 3) ")
    print("==================================================")
    
    # Caminho do diretório raiz
    root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    
    # Caminho das pastas de destino
    metrics_dir = os.path.join(root_dir, "results", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    try:
        # 1. Carregamento dos dados
        print("\n[PREPROC] Passo 1: Carregando dados originais...")
        df_raw = load_wine_data()
        original_features = [c for c in df_raw.columns if c not in ["quality", "Id", "id", "ID"]]
        print(f" - Quantidade de observações: {df_raw.shape[0]}")
        print(f" - Quantidade de features originais: {len(original_features)}")
        
        # 2. Criação da variável alvo
        print("\n[PREPROC] Passo 2: Criando a variável alvo binária...")
        df_target = create_target_variable(df_raw, source_col="quality", target_col="high_quality")
        
        # 3. Engenharia de Atributos
        print("\n[PREPROC] Passo 3: Executando Feature Engineering...")
        df_engineered = engineer_features(df_target)
        created_features = ["sulfur_ratio", "acidity_balance", "alcohol_density_ratio", "sugar_alcohol_ratio"]
        final_features = [c for c in df_engineered.columns if c not in ["quality", "high_quality", "Id", "id", "ID"]]
        
        # 4. Divisão Treino/Teste Estratificada
        print("\n[PREPROC] Passo 4: Realizando divisão de dados (Split Estratificado)...")
        # Identifica ID se existir
        id_cols = [c for c in ["Id", "id", "ID"] if c in df_engineered.columns]
        X_train, X_test, y_train, y_test = split_data(
            df_engineered,
            target_col="high_quality",
            drop_cols=id_cols,
            test_size=0.2,
            random_state=42
        )
        
        # 5. Validação das Proporções de Classe
        print("\n[PREPROC] Passo 5: Validando a proporção de classes nos splits...")
        train_counts = y_train.value_counts()
        train_pcts = y_train.value_counts(normalize=True) * 100
        test_counts = y_test.value_counts()
        test_pcts = y_test.value_counts(normalize=True) * 100
        total_counts = df_engineered["high_quality"].value_counts()
        total_pcts = df_engineered["high_quality"].value_counts(normalize=True) * 100
        
        print(f" - Classe 1 Geral: {total_counts.get(1, 0)} amostras ({total_pcts.get(1, 0.0):.2f}%)")
        print(f" - Classe 1 Treino: {train_counts.get(1, 0)} amostras ({train_pcts.get(1, 0.0):.2f}%)")
        print(f" - Classe 1 Teste: {test_counts.get(1, 0)} amostras ({test_pcts.get(1, 0.0):.2f}%)")
        
        # 6. Validação do Pipeline scikit-learn (Sem data leakage!)
        print("\n[PREPROC] Passo 6: Validando a pipeline de pré-processamento...")
        # Instancia o pipeline padrão (StandardScaler)
        pipeline = create_preprocessing_pipeline(scaler="standard")
        
        # Ajusta (fit) nos dados de TREINO apenas!
        print(" - Ajustando (fit) o pipeline em X_train...")
        X_train_transformed = pipeline.fit_transform(X_train)
        
        # Transforma (transform) apenas nos dados de TESTE!
        print(" - Aplicando (transform) nos dados de X_test...")
        X_test_transformed = pipeline.transform(X_test)
        print(" - Pipeline validada perfeitamente. Nenhuma falha detectada.")
        
        # 7. Salvando arquivos de saída requisitados
        print("\n[PREPROC] Passo 7: Salvando resumos e arquivos de auditoria em results/metrics/...")
        
        # 7.1 preprocessing_summary.json
        removed_cols = ["quality", "high_quality"] + id_cols
        preproc_summary = {
            "total_rows": int(df_raw.shape[0]),
            "num_original_features": int(len(original_features)),
            "num_created_features": int(len(created_features)),
            "num_final_features": int(len(final_features)),
            "train_size": int(X_train.shape[0]),
            "test_size": int(X_test.shape[0]),
            "class_1_proportion_total": float(total_pcts.get(1, 0.0) / 100),
            "class_1_proportion_train": float(train_pcts.get(1, 0.0) / 100),
            "class_1_proportion_test": float(test_pcts.get(1, 0.0) / 100),
            "default_scaler_chosen": "standard",
            "columns_removed": removed_cols,
            "columns_used_in_model": final_features
        }
        summary_json_path = os.path.join(metrics_dir, "preprocessing_summary.json")
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(preproc_summary, f, indent=4, ensure_ascii=False)
        print(f" - Salvo: {summary_json_path}")
        
        # 7.2 feature_engineering_summary.json
        feat_eng_summary = {
            "sulfur_ratio": {
                "formula": "free sulfur dioxide / (total sulfur dioxide + epsilon)",
                "business_justification": "Mede a fração de dióxido de enxofre ativo (livre) que efetivamente age contra a oxidação microbiológica do vinho, em relação ao enxofre total.",
                "technical_care": "Divisão protegida usando constante epsilon (1e-8) para evitar erro ZeroDivisionError caso a variável de total sulfur dioxide seja 0."
            },
            "acidity_balance": {
                "formula": "fixed acidity - volatile acidity",
                "business_justification": "Representa o balanço de acidez no paladar, contrastando a acidez física estrutural desejável com a acidez acética indesejável (sabor de vinagre).",
                "technical_care": "Subtração direta sem riscos de divisão. Mantém a escala original das grandezas físicas."
            },
            "alcohol_density_ratio": {
                "formula": "alcohol / (density + epsilon)",
                "business_justification": "Analisa o teor de álcool em relação à densidade geral do vinho. Como o álcool é menos denso que a água, a razão fornece um indicador indireto de corpo e maturação.",
                "technical_care": "Divisão protegida com constante epsilon (1e-8) para estabilidade numérica e consistência matemática."
            },
            "sugar_alcohol_ratio": {
                "formula": "residual sugar / (alcohol + epsilon)",
                "business_justification": "Avalia o açúcar residual não fermentado em proporção ao álcool produzido, indicando a doçura percebida em relação ao calor alcoólico.",
                "technical_care": "Divisão protegida com constante epsilon (1e-8) para evitar interrupções caso o teor de álcool de algum registro seja zero."
            }
        }
        feat_json_path = os.path.join(metrics_dir, "feature_engineering_summary.json")
        with open(feat_json_path, "w", encoding="utf-8") as f:
            json.dump(feat_eng_summary, f, indent=4, ensure_ascii=False)
        print(f" - Salvo: {feat_json_path}")
        
        # 7.3 train_test_split_summary.csv
        split_summary_data = {
            "split": ["Treino", "Teste", "Total"],
            "class_0_count": [int(train_counts.get(0, 0)), int(test_counts.get(0, 0)), int(total_counts.get(0, 0))],
            "class_0_percentage": [float(train_pcts.get(0, 0.0)), float(test_pcts.get(0, 0.0)), float(total_pcts.get(0, 0.0))],
            "class_1_count": [int(train_counts.get(1, 0)), int(test_counts.get(1, 0)), int(total_counts.get(1, 0))],
            "class_1_percentage": [float(train_pcts.get(1, 0.0)), float(test_pcts.get(1, 0.0)), float(total_pcts.get(1, 0.0))],
            "total_count": [int(y_train.shape[0]), int(y_test.shape[0]), int(df_raw.shape[0])]
        }
        split_summary_df = pd.DataFrame(split_summary_data)
        split_csv_path = os.path.join(metrics_dir, "train_test_split_summary.csv")
        split_summary_df.to_csv(split_csv_path, index=False)
        print(f" - Salvo: {split_csv_path}")
        
        # 7.4 feature_list.csv
        features_data = []
        for feat in final_features:
            feat_type = "criada (feature engineering)" if feat in created_features else "original (physicochemical)"
            features_data.append({"feature_name": feat, "type": feat_type})
        features_df = pd.DataFrame(features_data)
        features_csv_path = os.path.join(metrics_dir, "feature_list.csv")
        features_df.to_csv(features_csv_path, index=False)
        print(f" - Salvo: {features_csv_path}")
        
        print("\n==================================================")
        print(" PIPELINE DE PRÉ-PROCESSAMENTO EXECUTADA COM SUCESSO ")
        print("==================================================")
        
    except Exception as e:
        print(f"\n[ERRO CRÍTICO] Falha durante a execução da pipeline de pré-processamento: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
