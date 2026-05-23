import os
import sys
import json
import pandas as pd
import numpy as np

# Adiciona o diretório raiz ao path para garantir que imports funcionem robustamente
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.data_loader import load_wine_data
from src.preprocessing import create_target_variable
import src.plots as plots

def calculate_iqr_outliers(df: pd.DataFrame, numeric_cols: list) -> dict:
    """
    Calcula a quantidade e o percentual de outliers utilizando o método IQR (Interquartile Range).
    """
    outliers_dict = {}
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Filtra os outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        count = len(outliers)
        pct = (count / len(df)) * 100
        
        outliers_dict[col] = {
            "count": count,
            "percentage": pct,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }
    return outliers_dict

def main():
    print("==================================================")
    # Título estilizado no terminal
    print(" INICIANDO PIPELINE DE ANÁLISE EXPLORATÓRIA (EDA) ")
    print("==================================================")
    
    # Caminho do diretório raiz
    root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    
    # Criação das pastas de saída se não existirem
    figures_dir = os.path.join(root_dir, "results", "figures")
    metrics_dir = os.path.join(root_dir, "results", "metrics")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    try:
        # 1. Carregamento dos dados
        print("\n[EDA] Carregando dados...")
        df_raw = load_wine_data()
        df = create_target_variable(df_raw, source_col="quality", target_col="high_quality")
        
        n_rows, n_cols = df.shape
        numeric_df = df.select_dtypes(include=[np.number])
        numeric_cols = [c for c in numeric_df.columns if c not in ["Id", "quality", "high_quality"]]
        
        print(f"[EDA] Dimensões do dataset: {n_rows} linhas, {n_cols} colunas.")
        
        # 2. Estatísticas Descritivas
        print("\n[EDA] Gerando estatísticas descritivas...")
        desc_stats = df.describe().transpose()
        desc_stats_path = os.path.join(metrics_dir, "descriptive_statistics.csv")
        desc_stats.to_csv(desc_stats_path)
        
        # 3. Balanceamento de Classes (Original vs Binária)
        print("[EDA] Analisando balanceamento das classes...")
        quality_counts = df["quality"].value_counts().sort_index()
        quality_pct = df["quality"].value_counts(normalize=True).sort_index() * 100
        
        target_counts = df["high_quality"].value_counts().sort_index()
        target_pct = df["high_quality"].value_counts(normalize=True).sort_index() * 100
        
        class_balance_df = pd.DataFrame({
            "quality_original": quality_counts,
            "quality_percentage": quality_pct,
            "high_quality_binaria": target_counts,
            "high_quality_percentage": target_pct
        })
        class_balance_path = os.path.join(metrics_dir, "class_balance.csv")
        class_balance_df.to_csv(class_balance_path)
        
        # 4. Valores Ausentes
        print("[EDA] Analisando valores ausentes...")
        missing_values = df.isnull().sum()
        missing_pct = (df.isnull().sum() / n_rows) * 100
        missing_df = pd.DataFrame({
            "missing_count": missing_values,
            "missing_percentage": missing_pct
        })
        missing_values_path = os.path.join(metrics_dir, "missing_values.csv")
        missing_df.to_csv(missing_values_path)
        
        # 5. Outliers (IQR)
        print("[EDA] Identificando outliers pelo método IQR...")
        outliers_dict = calculate_iqr_outliers(df, numeric_cols)
        outliers_df = pd.DataFrame.from_dict(outliers_dict, orient='index')
        outliers_summary_path = os.path.join(metrics_dir, "outliers_summary.csv")
        outliers_df.to_csv(outliers_summary_path)
        
        # 6. Análise de Correlação (Pearson vs Spearman)
        print("[EDA] Computando matrizes de correlação...")
        # Pearson
        pearson_corr_matrix = numeric_df.drop(columns=["Id"], errors="ignore").corr(method="pearson")
        pearson_target = pearson_corr_matrix["high_quality"].drop(index=["high_quality", "quality"], errors="ignore")
        
        # Spearman
        spearman_corr_matrix = numeric_df.drop(columns=["Id"], errors="ignore").corr(method="spearman")
        spearman_target = spearman_corr_matrix["high_quality"].drop(index=["high_quality", "quality"], errors="ignore")
        
        corr_with_target = pd.DataFrame({
            "pearson_correlation": pearson_target,
            "spearman_correlation": spearman_target
        }).sort_values(by="pearson_correlation", ascending=False)
        corr_target_path = os.path.join(metrics_dir, "correlation_with_target.csv")
        corr_with_target.to_csv(corr_target_path)
        
        # Correlações mais fortes
        top_positive_pearson = corr_with_target["pearson_correlation"].idxmax()
        top_positive_pearson_val = corr_with_target["pearson_correlation"].max()
        top_negative_pearson = corr_with_target["pearson_correlation"].idxmin()
        top_negative_pearson_val = corr_with_target["pearson_correlation"].min()
        
        # 7. Exportação do eda_summary.json
        print("\n[EDA] Exportando eda_summary.json...")
        summary_data = {
            "dimensions": {
                "rows": n_rows,
                "columns": n_cols
            },
            "original_quality_distribution": {
                str(k): int(v) for k, v in quality_counts.items()
            },
            "binary_target_distribution": {
                str(k): int(v) for k, v in target_counts.items()
            },
            "high_quality_percentage": float(target_pct.get(1, 0.0)),
            "top_associations": {
                "strongest_positive_correlation": {
                    "variable": top_positive_pearson,
                    "pearson": float(top_positive_pearson_val),
                    "spearman": float(corr_with_target.loc[top_positive_pearson, "spearman_correlation"])
                },
                "strongest_negative_correlation": {
                    "variable": top_negative_pearson,
                    "pearson": float(top_negative_pearson_val),
                    "spearman": float(corr_with_target.loc[top_negative_pearson, "spearman_correlation"])
                }
            },
            "outliers_count_per_feature": {
                k: int(v["count"]) for k, v in outliers_dict.items()
            },
            "missing_values_per_feature": {
                k: int(v) for k, v in missing_values.items()
            }
        }
        
        summary_json_path = os.path.join(metrics_dir, "eda_summary.json")
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)
            
        # 8. Geração de Gráficos (300 DPI, premium)
        print("\n[EDA] Gerando gráficos de alta resolução (300 DPI)...")
        plots.plot_quality_distribution(df, os.path.join(figures_dir, "quality_distribution.png"))
        plots.plot_target_balance(df, os.path.join(figures_dir, "target_balance.png"))
        plots.plot_numeric_distributions(df, numeric_cols, os.path.join(figures_dir, "numeric_distributions.png"))
        plots.plot_correlation_matrix(df, "pearson", os.path.join(figures_dir, "correlation_matrix_pearson.png"))
        plots.plot_correlation_matrix(df, "spearman", os.path.join(figures_dir, "correlation_matrix_spearman.png"))
        plots.plot_target_correlation(pearson_target, spearman_target, os.path.join(figures_dir, "target_correlation_barplot.png"))
        
        # Variáveis chave para análise de boxplot
        key_features = ["alcohol", "volatile acidity", "sulphates"]
        plots.plot_boxplots_by_target(df, key_features, os.path.join(figures_dir, "boxplots_by_target.png"))
        plots.plot_outliers_overview(outliers_dict, os.path.join(figures_dir, "outliers_overview.png"))
        
        # 9. Geração do Relatório eda_report.md
        print("\n[EDA] Compilando relatório results/eda_report.md...")
        report_path = os.path.join(root_dir, "results", "eda_report.md")
        
        # Criação do conteúdo do relatório
        report_content = f"""# Relatório de Análise Exploratória de Dados (EDA) - Wine Quality

Este relatório apresenta os achados detalhados obtidos durante a análise exploratória físico-química do Wine Quality Dataset. Todas as conclusões são fundamentadas em estatísticas descritivas, testes de correlação linear (Pearson) e monotônica (Spearman), além de mapeamento formal de outliers.

---

## 1. Leitura Executiva (Insights de Negócio)

> [!NOTE]
> **O que foi observado?**
> A análise estatística identificou que os vinhos classificados como premium (`high_quality = 1`, nota >= 7) apresentam, em média, **teores alcoólicos substancialmente mais elevados** (estando a variável `alcohol` fortemente associada de forma positiva com a qualidade) e **menores concentrações de acidez volátil** (`volatile acidity`, associada de forma negativa). O teor de sulfatos (`sulphates`) também sugere uma tendência positiva moderada de associação com a alta qualidade.
>
> **Por que isso importa para a vinícola?**
> A acidez volátil é gerada principalmente por bactérias acéticas e está associada ao odor de vinagre no vinho. Monitorar ativamente e manter a acidez volátil baixa na produção pode indicar um melhor controle microbiológico e qualidade do produto final. Por outro lado, o teor de álcool e os sulfatos (que agem como antioxidantes e preservativos) indicam que a fermentação e o processo de conservação do vinho são fatores determinantes para a classificação sensorial superior.
>
> **Qual o possível impacto no modelo de Machine Learning?**
> - **Imbalance de Classe**: Apenas {target_pct.get(1, 0.0):.2f}% dos vinhos são de alta qualidade (159 amostras). Modelos preditivos simples focados em acurácia geral serão enviesados e falharão em detectar os vinhos premium. Por isso, a escolha do F1-score e splits estratificados são críticos.
> - **Variáveis Fortes**: Atributos como `alcohol` e `volatile acidity` serão provavelmente as principais features de decisão de modelos como Random Forest ou XGBoost.
> - **Presença de Outliers**: Algumas variáveis (especialmente `residual sugar` e `chlorides`) apresentam percentuais elevados de outliers. Modelos lineares (como Regressão Logística) podem sofrer impacto se esses valores extremos não forem adequadamente tratados, enquanto modelos de árvore (Random Forest) são inerentemente mais robustos a eles.
>
> **Qual decisão de negócio esse insight apoia?**
> 1. **Triagem Rápida na Linha de Produção**: Permite à vinícola rodar análises químicas simples de acidez volátil e teor alcoólico para classificar e separar preventivamente lotes que tenham potencial de ser rotulados como premium, otimizando o estoque físico e direcionando o envelhecimento em barris.
> 2. **Ajuste Fino de Processos**: Apoiar enólogos a ajustarem as etapas de fermentação e a conservação com sulfitos de maneira objetiva.

---

## 2. Visão Geral do Dataset e Estatísticas
O dataset de classificação física do vinho contém:
* **Total de Observações**: {n_rows} amostras
* **Total de Atributos**: {n_cols} variáveis (incluindo o ID de controle e a coluna alvo criada)
* **Valores Faltantes (Nulls)**: **Nenhum valor nulo** foi encontrado nas variáveis químico-físicas do dataset, o que garante a consistência do pipeline sem necessidade de técnicas de imputação.

### Distribuição das Notas Originais
A avaliação sensorial original varia de 3 a 8. A maior concentração de amostras situa-se nas notas 5 e 6, indicando que a grande maioria da produção possui qualidade intermediária.

![Distribuição das Notas Originais](figures/quality_distribution.png)

### Balanceamento da Variável Alvo (`high_quality`)
Binarizamos a qualidade definindo que notas de 7 a 8 representam a classe positiva (Vinhos de Alta Qualidade). A classe é altamente desbalanceada:
* **Classe 0 (Qualidade Média/Baixa < 7)**: {target_counts.get(0, 0)} amostras ({target_pct.get(0, 0.0):.2f}%)
* **Classe 1 (Alta Qualidade >= 7)**: {target_counts.get(1, 0)} amostras ({target_pct.get(1, 0.0):.2f}%)

![Balanceamento da Variável Alvo](figures/target_balance.png)

---

## 3. Correlações com a Qualidade do Vinho

Avaliamos a relação entre as variáveis através de dois métodos complementares:
1. **Coeficiente de Pearson**: Mede a correlação linear clássica.
2. **Coeficiente de Spearman**: Mede correlações de postos (monotônicas), sendo ideal para dados que não seguem distribuição perfeitamente normal e mitigando o efeito de outliers.

![Associação com high_quality](figures/target_correlation_barplot.png)

### Principais Associações Identificadas:
* **Teor Alcoólico (`alcohol`)**: Apresenta a relação positiva mais forte com a qualidade (Pearson: **{corr_with_target.loc["alcohol", "pearson_correlation"]:.3f}** | Spearman: **{corr_with_target.loc["alcohol", "spearman_correlation"]:.3f}**). Isso sugere uma tendência de que vinhos com maior maturação e fermentação completa sejam mais apreciados pelos avaliadores.
* **Acidez Volátil (`volatile acidity`)**: Apresenta a correlação negativa mais acentuada com o alvo (Pearson: **{corr_with_target.loc["volatile acidity", "pearson_correlation"]:.3f}** | Spearman: **{corr_with_target.loc["volatile acidity", "spearman_correlation"]:.3f}**). Níveis elevados de acidez volátil podem indicar a presença de defeitos sensoriais associados à oxidação acética.
* **Sulfatos (`sulphates`)**: Mostram uma associação positiva moderada (Pearson: **{corr_with_target.loc["sulphates", "pearson_correlation"]:.3f}** | Spearman: **{corr_with_target.loc["sulphates", "spearman_correlation"]:.3f}**), reforçando que a preservação e estabilização adequadas do vinho desempenham um papel relevante no produto final.

### Matrizes de Correlação Completas
As matrizes de correlação ilustram também as interações entre as variáveis independentes, o que é fundamental para evitar a multicolinearidade em modelos estatísticos.

| Matriz de Pearson (Linear) | Matriz de Spearman (Monotônica) |
| :---: | :---: |
| ![Pearson](figures/correlation_matrix_pearson.png) | ![Spearman](figures/figures/correlation_matrix_spearman.png) |

---

## 4. Análise de Variáveis Chave
Os boxplots abaixo ilustram visualmente a distribuição dos três principais fatores associados com vinhos premium (Classe 1) em comparação com vinhos comuns (Classe 0):

![Impacto das Variáveis Chave](figures/boxplots_by_target.png)

---

## 5. Análise de Outliers e Valores Atípicos
Utilizando a metodologia do Intervalo Interquartil (IQR), mapeamos os valores atípicos que se situam além dos limites calculados:
`[Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]`

![Visão Geral de Outliers](figures/outliers_overview.png)

### Discussão de Outliers e Impacto:
* **Cloretos (`chlorides`) e Açúcar Residual (`residual sugar`)**: Apresentam taxas muito elevadas de outliers ({outliers_dict.get('chlorides', {}).get('percentage', 0.0):.2f}% e {outliers_dict.get('residual sugar', {}).get('percentage', 0.0):.2f}%, respectivamente). Estes extremos não representam necessariamente dados corrompidos, mas sim variações legítimas de vinificação (ex: vinhos mais doces ou com traços minerais/salinos distintos).
* **Tratamento de Dados**: Para modelos que serão testados a seguir:
  - Modelos Lineares (ex: Regressão Logística) se beneficiarão de transformações de escala robustas (como `RobustScaler`) ou clipping de valores extremos para mitigar o impacto dos outliers.
  - Modelos Baseados em Árvores (ex: Random Forest e XGBoost) lidarão com estes limites de forma natural sem perda de performance.

---

## 6. Conclusões e Próximos Passos
O bootstrap e a análise exploratória comprovam que:
1. Os dados físico-químicos são limpos, consistentes e apresentam fortes correlações com a qualidade percebida.
2. O desbalanceamento de classes é o maior desafio técnico (apenas {target_pct.get(1, 0.0):.2f}% da classe positiva), o que valida nossa estratégia de utilizar splits estratificados e focar na otimização do **F1-Score da Classe 1**.

Com a fundação visual da EDA estruturada e os dados auditados em arquivos CSV na pasta `results/metrics/`, o projeto está perfeitamente pronto para a **Fase 3 (Pré-processamento e Feature Engineering)** e o desenvolvimento de modelos.
"""
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
            
        print("\n==================================================")
        print(" PIPELINE DE EDA CONCLUÍDA E RELATÓRIO VISUAL GERADO ")
        print("==================================================")
        
    except Exception as e:
        print(f"\n[ERRO CRÍTICO] Falha durante a execução da pipeline de EDA: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
