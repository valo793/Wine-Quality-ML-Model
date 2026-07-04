"""
Modulo de Interpretabilidade via SHAP.

Funcoes para calcular SHAP values, gerar graficos de explicabilidade
e extrair insights textuais do modelo vencedor.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap


# ---------------------------------------------------------------------------
# Nomes amigaveis para features (pt-BR, linguagem de negocio)
# ---------------------------------------------------------------------------
FEATURE_LABELS = {
    "alcohol": "Teor Alcoolico",
    "volatile acidity": "Acidez Volatil",
    "sulphates": "Sulfatos",
    "citric acid": "Acido Citrico",
    "total sulfur dioxide": "Dioxido de Enxofre Total",
    "free sulfur dioxide": "Dioxido de Enxofre Livre",
    "density": "Densidade",
    "pH": "pH",
    "residual sugar": "Acucar Residual",
    "chlorides": "Cloretos",
    "fixed acidity": "Acidez Fixa",
    "sulfur_ratio": "Razao de Enxofre (Livre/Total)",
    "acidity_balance": "Equilibrio de Acidez",
    "alcohol_density_ratio": "Razao Alcool/Densidade",
    "sugar_alcohol_ratio": "Razao Acucar/Alcool",
}

# Interpretacoes de negocio para as features mais relevantes
FEATURE_INSIGHTS = {
    "alcohol": (
        "O teor alcoolico e o fator mais influente na classificacao de qualidade. "
        "Vinhos com teor alcoolico mais elevado tendem a ser classificados como premium, "
        "o que reflete a relacao entre maturacao da uva, fermentacao completa e corpo do vinho."
    ),
    "volatile acidity": (
        "A acidez volatil (acido acetico) impacta negativamente a qualidade. "
        "Niveis elevados indicam problemas de fermentacao ou contaminacao bacteriana, "
        "resultando em sabores avinagrados que depreciam o produto."
    ),
    "sulphates": (
        "Os sulfatos contribuem positivamente para a qualidade, atuando como "
        "conservantes e antioxidantes. Niveis adequados protegem o aroma e a cor "
        "do vinho durante o armazenamento."
    ),
    "citric acid": (
        "O acido citrico adiciona frescor e complexidade ao sabor. "
        "Vinhos premium geralmente apresentam niveis moderados dessa substancia."
    ),
    "total sulfur dioxide": (
        "O dioxido de enxofre total em excesso pode mascarar aromas delicados. "
        "O modelo identifica que niveis controlados favorecem a qualidade."
    ),
    "density": (
        "A densidade esta inversamente relacionada ao teor alcoolico e diretamente "
        "ao acucar residual. Vinhos mais leves e secos tendem a receber notas superiores."
    ),
    "sulfur_ratio": (
        "A proporcao de enxofre livre sobre o total indica a eficiencia da "
        "protecao antioxidante. Razoes mais altas sugerem melhor conservacao."
    ),
    "alcohol_density_ratio": (
        "Feature engenheirada que captura simultaneamente o corpo e a leveza do vinho. "
        "Valores elevados indicam vinhos com boa estrutura alcoolica e baixa densidade."
    ),
}


def compute_shap_values(pipeline, X_test):
    """
    Calcula SHAP values usando TreeExplainer sobre o classificador
    extraido do pipeline scikit-learn.

    Retorna:
        shap_values: array com os SHAP values
        explainer: objeto TreeExplainer
    """
    # Extrair o classificador do pipeline
    classifier = pipeline.named_steps["classifier"]

    # Pre-processar X_test com os steps anteriores do pipeline
    preprocessing = pipeline[:-1]  # todos os steps exceto o classificador
    X_test_processed = preprocessing.transform(X_test)

    # Garantir que e um array numpy com nomes de features
    if hasattr(X_test_processed, "values"):
        X_processed_array = X_test_processed.values
    elif hasattr(X_test_processed, "toarray"):
        X_processed_array = X_test_processed.toarray()
    else:
        X_processed_array = np.array(X_test_processed)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_processed_array)

    # Para classificacao binaria, pegar os SHAP values da classe 1 (premium)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]

    return shap_values, explainer, X_processed_array


def plot_shap_summary(shap_values, X_processed, feature_names, save_path):
    """Gera o grafico SHAP summary (beeswarm) com ranking de influencia."""
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_processed,
        feature_names=feature_names,
        show=False,
        plot_size=(12, 8),
    )
    plt.title("Impacto das Variaveis na Predicao de Qualidade Premium (SHAP)",
              fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] SHAP Summary salvo: {save_path}")


def plot_shap_bar(shap_values, feature_names, save_path):
    """Gera grafico de barras com importancia media absoluta SHAP."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_idx)))

    bars = ax.barh(
        range(len(sorted_idx)),
        mean_abs[sorted_idx][::-1],
        color=colors[::-1],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx[::-1]], fontsize=11)
    ax.set_xlabel("Importancia Media |SHAP|", fontsize=12, fontweight="bold")
    ax.set_title("Ranking de Importancia das Variaveis (SHAP Values)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] SHAP Bar salvo: {save_path}")


def plot_shap_dependence(shap_values, X_processed, feature_names, feature, save_path):
    """Gera grafico de dependencia SHAP para uma feature especifica."""
    idx = feature_names.index(feature)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.dependence_plot(
        idx,
        shap_values,
        X_processed,
        feature_names=feature_names,
        show=False,
        ax=ax,
    )
    label = FEATURE_LABELS.get(feature, feature)
    ax.set_title(f"Efeito de {label} na Predicao (SHAP Dependence)",
                 fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] SHAP Dependence ({feature}) salvo: {save_path}")


def generate_shap_ranking(shap_values, feature_names):
    """
    Gera ranking de features por importancia SHAP media absoluta.
    Retorna lista de dicts ordenada.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]

    ranking = []
    for rank, idx in enumerate(sorted_idx, start=1):
        fname = feature_names[idx]
        ranking.append({
            "rank": rank,
            "feature": fname,
            "feature_label": FEATURE_LABELS.get(fname, fname),
            "mean_abs_shap": float(mean_abs[idx]),
            "insight": FEATURE_INSIGHTS.get(fname, ""),
        })

    return ranking
