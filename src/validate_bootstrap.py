import sys
import os

# Adiciona o diretório raiz ao path para garantir que imports locais funcionem
# mesmo se executado de fora do diretório raiz
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.data_loader import load_wine_data
from src.preprocessing import create_target_variable, split_data

def main():
    print("==================================================")
    print(" INICIANDO VALIDAÇÃO DO BOOTSTRAP DO PROJETO ML   ")
    print("==================================================")
    
    try:
        # 1. Carregar os dados
        print("\nPasso 1: Carregando os dados...")
        df_raw = load_wine_data()
        print(f"Dataset carregado com sucesso. Dimensões: {df_raw.shape[0]} linhas, {df_raw.shape[1]} colunas.")
        print("\nPrimeiros 5 registros:")
        print(df_raw.head())
        
        # 2. Criar a variável alvo binária high_quality
        print("\nPasso 2: Criando a variável alvo binária 'high_quality'...")
        df_processed = create_target_variable(df_raw, source_col="quality", target_col="high_quality")
        
        # Imprime a distribuição original e a nova distribuição
        print("\nDistribuição das notas originais:")
        print(df_processed["quality"].value_counts().sort_index())
        
        print("\nDistribuição da variável alvo binária:")
        counts = df_processed["high_quality"].value_counts()
        percentages = df_processed["high_quality"].value_counts(normalize=True) * 100
        for val, count in counts.items():
            pct = percentages[val]
            label = "Alta Qualidade (>=7)" if val == 1 else "Baixa/Média Qualidade (<7)"
            print(f" - Classe {val} ({label}): {count} amostras ({pct:.2f}%)")
            
        # 3. Separar features e target e fazer Split Treino/Teste Estratificado
        print("\nPasso 3: Realizando a divisão de dados em treino e teste (estratificado)...")
        # Dropamos o 'quality' original para evitar target leakage
        X_train, X_test, y_train, y_test = split_data(
            df_processed, 
            target_col="high_quality", 
            drop_cols=["quality"],
            test_size=0.2, 
            random_state=42
        )
        
        # 4. Validar que a estratificação foi mantida com sucesso
        print("\nPasso 4: Verificando a consistência da amostragem estratificada...")
        train_pct = y_train.value_counts(normalize=True) * 100
        test_pct = y_test.value_counts(normalize=True) * 100
        
        print("Proporção da Classe 1 (Alta Qualidade):")
        print(f" - Base de Treino: {train_pct.get(1, 0.0):.2f}%")
        print(f" - Base de Teste: {test_pct.get(1, 0.0):.2f}%")
        
        print("\n==================================================")
        print(" VALIDAÇÃO CONCLUÍDA: PIPELINE INICIAL FUNCIONANDO!")
        print("==================================================")
        
    except Exception as e:
        print("\n[ERRO CRÍTICO] Falha durante a validação da pipeline inicial:")
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    main()
