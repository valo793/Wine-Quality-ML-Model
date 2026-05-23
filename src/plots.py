import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuração de estilo visual corporativo e premium
# Evita as cores padrão "plain red/blue/green" e estabelece paleta premium
PREMIUM_PALETTE = ["#2b2d42", "#8d99ae", "#d90429", "#ef233c", "#f77f00"]
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 120

def plot_confusion_matrix(cm: np.ndarray, save_path: str = None):
    """
    Plota e salva uma matriz de confusão visualmente elegante e premium.
    
    Parameters:
    -----------
    cm : np.ndarray
        Matriz de confusão (2x2).
    save_path : str, default=None
        Caminho onde salvar o gráfico gerado.
    """
    plt.figure(figsize=(6, 5))
    
    # Usando paleta de cores elegantes (tons de azul-escuro/cinza premium)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        cbar=False,
        xticklabels=["Baixa/Média", "Alta"],
        yticklabels=["Baixa/Média", "Alta"],
        annot_kws={"size": 14, "weight": "bold"}
    )
    
    plt.title("Matriz de Confusão (Previsão de Qualidade)", fontsize=14, pad=15, weight="bold", color="#1a1a1a")
    plt.xlabel("Qualidade Predita", fontsize=11, labelpad=10)
    plt.ylabel("Qualidade Real", fontsize=11, labelpad=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Matriz de confusão salva em: {save_path}")
    plt.close()

def plot_feature_importance(importances: list, feature_names: list, save_path: str = None):
    """
    Plota e salva o gráfico de relevância das variáveis com design corporativo premium.
    
    Parameters:
    -----------
    importances : list
        Valores de importância das features.
    feature_names : list
        Nomes correspondentes das features.
    save_path : str, default=None
        Caminho onde salvar o gráfico gerado.
    """
    # Ordena as features por importância
    indices = np.argsort(importances)
    
    plt.figure(figsize=(8, 6))
    
    # Paleta corporativa premium de barras
    colors = sns.color_palette("ch:start=.2,rot=-.3", len(importances))
    
    plt.barh(
        range(len(indices)), 
        [importances[i] for i in indices], 
        color=colors, 
        edgecolor="none"
    )
    
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=10)
    plt.title("Variáveis mais Relevantes para a Qualidade do Vinho", fontsize=14, pad=15, weight="bold", color="#1a1a1a")
    plt.xlabel("Grau de Importância Relativa", fontsize=11, labelpad=10)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Gráfico de feature importance salvo em: {save_path}")
    plt.close()
