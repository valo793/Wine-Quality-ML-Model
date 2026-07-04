import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Configurações globais de estilo premium e corporativo
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.facecolor"] = "#f8f9fa"
plt.rcParams["axes.facecolor"] = "#ffffff"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.color"] = "#f1f3f5"
plt.rcParams["grid.linewidth"] = 0.5

# Paleta premium de cores de negócio
PRIMARY_COLOR = "#1a365d"     # Azul escuro executivo
SECONDARY_COLOR = "#4a5568"   # Cinza escuro elegante
ACCENT_COLOR = "#9b2c2c"      # Vermelho borgonha (vinho)
HIGHLIGHT_COLOR = "#2b6cb0"   # Azul brilhante para contrastes

def plot_quality_distribution(df: pd.DataFrame, save_path: str):
    """
    Gera e salva o gráfico de distribuição das notas originais de qualidade (quality).
    DPI = 300
    """
    plt.figure(figsize=(8, 5))
    
    # Contagem das frequências
    ax = sns.countplot(
        x="quality", 
        data=df, 
        color=PRIMARY_COLOR,
        edgecolor="#2d3748",
        linewidth=1
    )
    
    # Adiciona valores sobre as barras para clareza
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}", 
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha='center', va='baseline', 
            fontsize=10, color='#2d3748', 
            xytext=(0, 5), textcoords='offset points',
            weight="bold"
        )
        
    plt.title("Distribuição das Notas Originais de Qualidade (Quality)", fontsize=13, pad=15, weight="bold", color="#1a202c")
    plt.xlabel("Nota Sensorial (3 a 8)", fontsize=10, labelpad=8)
    plt.ylabel("Quantidade de Amostras", fontsize=10, labelpad=8)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo: {save_path}")

def plot_target_balance(df: pd.DataFrame, save_path: str):
    """
    Gera e salva a distribuição da variável alvo binária (high_quality).
    DPI = 300
    """
    plt.figure(figsize=(7, 5))
    
    counts = df["high_quality"].value_counts()
    pcts = df["high_quality"].value_counts(normalize=True) * 100
    
    # Paleta premium de duas cores
    colors = [SECONDARY_COLOR, ACCENT_COLOR]
    
    ax = sns.barplot(
        x=counts.index, 
        y=counts.values, 
        hue=counts.index,
        palette=colors,
        legend=False,
        edgecolor="#2d3748",
        linewidth=1
    )
    
    # Adiciona o texto descritivo sobre as barras (quantidade e %)
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.annotate(
            f"{int(height)} amostras\n({pcts.iloc[i]:.1f}%)", 
            (p.get_x() + p.get_width() / 2., height), 
            ha='center', va='baseline', 
            fontsize=10, color='#2d3748', 
            xytext=(0, 5), textcoords='offset points',
            weight="bold"
        )
        
    plt.title("Balanceamento da Variável Alvo (Classe de Modelagem)", fontsize=13, pad=15, weight="bold", color="#1a202c")
    plt.xlabel("Classificação do Vinho", fontsize=10, labelpad=8)
    plt.ylabel("Quantidade", fontsize=10, labelpad=8)
    plt.xticks([0, 1], ["Baixa/Média Qualidade (< 7)", "Alta Qualidade (>= 7)"])
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo: {save_path}")

def plot_numeric_distributions(df: pd.DataFrame, numeric_cols: list, save_path: str):
    """
    Gera e salva histogramas múltiplos para variáveis numéricas chave.
    DPI = 300
    """
    n_cols = len(numeric_cols)
    rows = (n_cols + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 3.5))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        sns.histplot(
            data=df, 
            x=col, 
            kde=True, 
            color=PRIMARY_COLOR, 
            ax=axes[i],
            edgecolor="none"
        )
        axes[i].set_title(f"Distribuição: {col}", fontsize=11, weight="bold", color="#2d3748")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        axes[i].tick_params(axis="both", which="major", labelsize=9)
        
    # Remove eixos sobressalentes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle("Distribuições das Variáveis Físico-Químicas", fontsize=16, weight="bold", y=0.98, color="#1a202c")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo: {save_path}")

def plot_correlation_matrix(df: pd.DataFrame, method: str, save_path: str):
    """
    Gera e salva mapa de calor de correlação entre as variáveis usando o método especificado.
    DPI = 300
    """
    plt.figure(figsize=(10, 8))
    
    # Exclui variáveis não numéricas ou ID
    numeric_df = df.select_dtypes(include=[np.number])
    if "Id" in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=["Id"])
        
    corr = numeric_df.corr(method=method)
    
    # Máscara para ocultar o triângulo superior (evita redundância visual)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Mapa de calor elegante
    sns.heatmap(
        corr, 
        mask=mask,
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        vmin=-1, 
        vmax=1, 
        center=0,
        square=True, 
        linewidths=0.5, 
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 8}
    )
    
    title_str = f"Matriz de Correlação de {method.capitalize()}"
    plt.title(title_str, fontsize=14, pad=20, weight="bold", color="#1a202c")
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo: {save_path}")

def plot_target_correlation(pearson_corr: pd.Series, spearman_corr: pd.Series, save_path: str):
    """
    Gera um gráfico de barras comparativo das correlações (Pearson vs Spearman) com a variável alvo.
    DPI = 300
    """
    plt.figure(figsize=(10, 6))
    
    # Combina em um DataFrame para plotagem
    corr_df = pd.DataFrame({
        "Pearson": pearson_corr,
        "Spearman": spearman_corr
    }).drop(index=["high_quality", "quality", "Id"], errors="ignore")
    
    # Reshape para formato longo adequado para seaborn
    corr_df = corr_df.reset_index().melt(id_vars="index", var_name="Método", value_name="Correlação")
    corr_df = corr_df.rename(columns={"index": "Atributo"})
    
    # Ordena pelo valor absoluto médio para facilitar a visualização de relevância
    grouped = corr_df.groupby("Atributo")["Correlação"].apply(lambda x: x.abs().mean()).sort_values(ascending=False)
    order = grouped.index
    
    sns.barplot(
        data=corr_df,
        y="Atributo",
        x="Correlação",
        hue="Método",
        palette=[PRIMARY_COLOR, ACCENT_COLOR],
        order=order,
        edgecolor="none"
    )
    
    plt.axvline(x=0, color="#2d3748", linestyle="-", linewidth=0.8)
    plt.title("Associação dos Atributos com a Qualidade do Vinho (high_quality)", fontsize=13, pad=15, weight="bold", color="#1a202c")
    plt.xlabel("Coeficiente de Correlação", fontsize=10, labelpad=8)
    plt.ylabel("")
    plt.legend(title="Método de Análise", frameon=True)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo: {save_path}")

def plot_boxplots_by_target(df: pd.DataFrame, key_features: list, save_path: str):
    """
    Gera boxplots de variáveis chave distribuídas pelo alvo binário para mostrar diferenças visuais.
    DPI = 300
    """
    fig, axes = plt.subplots(1, len(key_features), figsize=(14, 5))
    
    colors = [SECONDARY_COLOR, ACCENT_COLOR]
    
    for i, col in enumerate(key_features):
        sns.boxplot(
            data=df, 
            x="high_quality", 
            y=col, 
            hue="high_quality",
            ax=axes[i], 
            palette=colors,
            legend=False,
            width=0.5,
            linewidth=1.2,
            fliersize=3
        )
        axes[i].set_title(f"Impacto: {col}", fontsize=11, weight="bold", color="#2d3748")
        axes[i].set_xlabel("")
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(["Comum (< 7)", "Premium (>= 7)"], fontsize=9)
        axes[i].set_ylabel("")
        
    plt.suptitle("Diferenças nos Atributos Químicos por Categoria de Qualidade", fontsize=15, weight="bold", y=1.02, color="#1a202c")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo: {save_path}")

def plot_outliers_overview(outliers_dict: dict, save_path: str):
    """
    Gera um gráfico de barras horizontal resumindo a quantidade de outliers detectada por atributo.
    DPI = 300
    """
    plt.figure(figsize=(9, 5))
    
    features = list(outliers_dict.keys())
    counts = [info["count"] for info in outliers_dict.values()]
    percentages = [info["percentage"] for info in outliers_dict.values()]
    
    # Ordena para melhor leitura gráfica
    sorted_idx = np.argsort(counts)
    sorted_features = [features[i] for i in sorted_idx]
    sorted_counts = [counts[i] for i in sorted_idx]
    sorted_pcts = [percentages[i] for i in sorted_idx]
    
    ax = sns.barplot(
        x=sorted_counts, 
        y=sorted_features, 
        color=ACCENT_COLOR,
        edgecolor="none"
    )
    
    # Adiciona rótulos percentuais no final de cada barra
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        if width > 0:
            ax.annotate(
                f" {int(width)} ({sorted_pcts[i]:.1f}%)", 
                (width, p.get_y() + p.get_height() / 2.), 
                ha='left', va='center', 
                fontsize=9, color='#2d3748', 
                weight="bold"
            )
            
    plt.title("Visão Geral de Valores Atípicos (Outliers) por Atributo (Método IQR)", fontsize=13, pad=15, weight="bold", color="#1a202c")
    plt.xlabel("Quantidade de Outliers Identificados", fontsize=10, labelpad=8)
    plt.ylabel("")
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo: {save_path}")


# =============================================================================
# FUNÇÕES DE VISUALIZAÇÃO PARA MODELAGEM (FASE 4-5)
# =============================================================================

def plot_confusion_matrices(results_dict: dict, save_path: str):
    """
    Gera um grid de matrizes de confusão lado a lado para cada modelo avaliado.
    DPI = 300
    
    Parameters
    ----------
    results_dict : dict
        Dicionário {model_display_name: metrics_dict} contendo 'confusion_matrix'.
    save_path : str
        Caminho de saída do gráfico.
    """
    from sklearn.metrics import ConfusionMatrixDisplay
    
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    class_labels = ["Baixa/Média (0)", "Alta (1)"]
    cmap = sns.color_palette("Blues", as_cmap=True)
    
    for i, (model_name, metrics) in enumerate(results_dict.items()):
        cm = np.array(metrics["confusion_matrix"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(ax=axes[i], cmap=cmap, values_format="d", colorbar=False)
        axes[i].set_title(f"{model_name}", fontsize=12, weight="bold", color="#1a202c", pad=10)
        axes[i].set_xlabel("Predição", fontsize=10)
        axes[i].set_ylabel("Real" if i == 0 else "", fontsize=10)
    
    plt.suptitle("Matrizes de Confusão — Comparação entre Modelos",
                 fontsize=15, weight="bold", y=1.03, color="#1a202c")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo: {save_path}")


def plot_roc_curves(roc_data: dict, save_path: str):
    """
    Gera curvas ROC sobrepostas para múltiplos modelos com AUC anotado na legenda.
    DPI = 300
    
    Parameters
    ----------
    roc_data : dict
        Dicionário {model_display_name: {"fpr": array, "tpr": array, "auc": float}}.
    save_path : str
        Caminho de saída do gráfico.
    """
    plt.figure(figsize=(8, 7))
    
    palette = [PRIMARY_COLOR, ACCENT_COLOR, HIGHLIGHT_COLOR, SECONDARY_COLOR, "#38a169", "#d69e2e"]
    
    for i, (model_name, data) in enumerate(roc_data.items()):
        color = palette[i % len(palette)]
        plt.plot(
            data["fpr"], data["tpr"],
            label=f"{model_name} (AUC = {data['auc']:.4f})",
            color=color,
            linewidth=2.2,
        )
    
    # Linha de referência (classificador aleatório)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Classificador Aleatório (AUC = 0.5)")
    
    plt.title("Curvas ROC — Comparação entre Modelos",
              fontsize=14, weight="bold", pad=15, color="#1a202c")
    plt.xlabel("Taxa de Falsos Positivos (FPR)", fontsize=11, labelpad=8)
    plt.ylabel("Taxa de Verdadeiros Positivos (TPR)", fontsize=11, labelpad=8)
    plt.legend(loc="lower right", fontsize=10, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo: {save_path}")


def plot_model_comparison_bar(comparison_df: pd.DataFrame, save_path: str):
    """
    Gera gráfico de barras agrupadas comparando métricas entre modelos.
    DPI = 300
    
    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame com colunas: Modelo, F1-Score (Classe 1), Precision, Recall, ROC-AUC, Acurácia.
    save_path : str
        Caminho de saída do gráfico.
    """
    plt.figure(figsize=(12, 6))
    
    metrics_cols = [c for c in comparison_df.columns if c != "Modelo"]
    
    # Reshape para formato longo
    melted = comparison_df.melt(id_vars="Modelo", value_vars=metrics_cols,
                                 var_name="Métrica", value_name="Valor")
    
    palette = [PRIMARY_COLOR, ACCENT_COLOR, HIGHLIGHT_COLOR, SECONDARY_COLOR, "#38a169"]
    
    ax = sns.barplot(
        data=melted,
        x="Métrica",
        y="Valor",
        hue="Modelo",
        palette=palette[:len(comparison_df)],
        edgecolor="#2d3748",
        linewidth=0.8,
    )
    
    # Adiciona valores sobre as barras
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=3, weight="bold", color="#2d3748")
    
    plt.title("Comparação de Métricas entre Modelos de Classificação",
              fontsize=14, weight="bold", pad=15, color="#1a202c")
    plt.xlabel("", fontsize=10)
    plt.ylabel("Valor da Métrica", fontsize=11, labelpad=8)
    plt.ylim(0, 1.12)
    plt.legend(title="Modelo", frameon=True, fancybox=True, loc="upper right")
    plt.xticks(rotation=15, ha="right")
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo: {save_path}")


def plot_feature_importance(importances: np.ndarray, feature_names: list, model_name: str, save_path: str, top_n: int = 15):
    """
    Gera gráfico de barras horizontal da importância das features para modelos baseados em árvore.
    DPI = 300
    
    Parameters
    ----------
    importances : np.ndarray
        Array de importâncias das features.
    feature_names : list
        Nomes das features correspondentes.
    model_name : str
        Nome do modelo para o título.
    save_path : str
        Caminho de saída do gráfico.
    top_n : int, default=15
        Número máximo de features a exibir.
    """
    plt.figure(figsize=(9, 6))
    
    # Ordena por importância
    indices = np.argsort(importances)[::-1][:top_n]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    # Inverte para exibir a mais importante no topo
    sorted_features = sorted_features[::-1]
    sorted_importances = sorted_importances[::-1]
    
    ax = sns.barplot(
        x=sorted_importances,
        y=sorted_features,
        color=PRIMARY_COLOR,
        edgecolor="none",
    )
    
    # Adiciona valores
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        ax.annotate(
            f" {width:.4f}",
            (width, p.get_y() + p.get_height() / 2.),
            ha="left", va="center",
            fontsize=9, color="#2d3748", weight="bold",
        )
    
    plt.title(f"Importância das Features — {model_name}",
              fontsize=13, weight="bold", pad=15, color="#1a202c")
    plt.xlabel("Importância (Gini / Ganho)", fontsize=10, labelpad=8)
    plt.ylabel("")
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo: {save_path}")
