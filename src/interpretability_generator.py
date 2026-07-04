"""
Orquestrador da Fase 6: Interpretacao e Storytelling.

Executa a analise SHAP sobre o modelo vencedor, gera graficos de
explicabilidade, metricas de ranking e o relatorio de interpretabilidade.

Execucao:
    .\\.venv\\Scripts\\python.exe -m src.interpretability_generator
"""

import os
import sys
import json
import traceback

import joblib
import numpy as np

from src.data_loader import load_wine_data
from src.preprocessing import create_target_variable, split_data
from src.features import engineer_features
from src.interpretability import (
    compute_shap_values,
    plot_shap_summary,
    plot_shap_bar,
    plot_shap_dependence,
    generate_shap_ranking,
    FEATURE_LABELS,
)


def generate_interpretability_report(ranking, best_model_name, report_path):
    """Gera o relatorio Markdown de interpretabilidade."""

    top5 = ranking[:5]

    # Tabela de ranking
    table_rows = ""
    for item in ranking:
        table_rows += (
            f"| {item['rank']} | {item['feature_label']} "
            f"(`{item['feature']}`) | {item['mean_abs_shap']:.4f} |\n"
        )

    # Insights das top 5
    insights_section = ""
    for item in top5:
        if item["insight"]:
            insights_section += f"### {item['rank']}. {item['feature_label']}\n\n"
            insights_section += f"{item['insight']}\n\n"

    report_content = f"""# Relatorio de Interpretabilidade - Analise SHAP

## Contexto

Este relatorio documenta a **Fase 6 (Interpretacao dos Resultados)** do Tech Challenge.
O objetivo e identificar quais variaveis tem maior influencia na classificacao de
qualidade do vinho e discutir as implicacoes praticas para o processo de producao.

- **Modelo Analisado**: {best_model_name}
- **Metodo de Explicabilidade**: SHAP (SHapley Additive exPlanations)
- **Tipo de Explainer**: TreeExplainer (otimizado para modelos de arvore)

---

## O que e SHAP?

SHAP e um metodo baseado na teoria de jogos cooperativos (valores de Shapley) que
atribui a cada variavel uma contribuicao justa para a predicao final do modelo.
Diferente da importancia de features por impureza (Gini), o SHAP e mais confiavel
porque considera as interacoes entre variaveis e oferece explicacoes locais
(por amostra) e globais (agregadas).

---

## Ranking de Importancia das Variaveis

| Posicao | Variavel | Importancia Media |SHAP| |
|---|---|---|
{table_rows}

---

## Graficos SHAP Gerados

| Grafico | Arquivo | Descricao |
|---|---|---|
| Summary (Beeswarm) | `results/figures/shap_summary.png` | Distribuicao do impacto de cada variavel |
| Barras de Importancia | `results/figures/shap_bar.png` | Ranking de importancia media absoluta |
| Dependencia - Alcohol | `results/figures/shap_dependence_alcohol.png` | Efeito do teor alcoolico |
| Dependencia - Volatile Acidity | `results/figures/shap_dependence_volatile_acidity.png` | Efeito da acidez volatil |
| Dependencia - Sulphates | `results/figures/shap_dependence_sulphates.png` | Efeito dos sulfatos |

---

## Insights por Variavel (Top 5)

{insights_section}

---

## Implicacoes para o Processo de Producao

Com base na analise SHAP, as seguintes recomendacoes podem ser feitas para a industria vinicola:

1. **Controle do Teor Alcoolico**: Monitorar e otimizar o processo de fermentacao para
   garantir que os niveis de alcool atinjam a faixa associada a vinhos premium.

2. **Reducao da Acidez Volatil**: Implementar controles rigorosos de higiene e temperatura
   durante a fermentacao para minimizar a producao de acido acetico.

3. **Dosagem de Sulfatos**: Ajustar a adicao de sulfitos para proteger o vinho sem
   comprometer o perfil aromatico.

4. **Monitoramento em Tempo Real**: Utilizar os dados fisico-quimicos coletados na
   linha de producao como entrada para o modelo, permitindo triagem automatica de lotes.

5. **Regua de Decisao**: Aplicar o modelo como ferramenta de apoio a decisao,
   classificando lotes por probabilidade de qualidade premium antes do engarrafamento.

---

## Proximos Passos

- Apresentacao executiva disponivel em `presentation/index.html`.
- Video executivo de ate 5 minutos com roteiro baseado nos insights acima.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"  [OK] Relatorio salvo: {report_path}")


def main():
    print("=" * 65)
    print(" INICIANDO PIPELINE DE INTERPRETABILIDADE (FASE 6)")
    print("=" * 65)

    root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    figures_dir = os.path.join(root_dir, "results", "figures")
    metrics_dir = os.path.join(root_dir, "results", "metrics")
    models_dir = os.path.join(root_dir, "results", "models")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    try:
        # =================================================================
        # ETAPA 1: CARREGAR MODELO E RECONSTRUIR DADOS
        # =================================================================
        print("\n[SHAP] Etapa 1: Carregando modelo e reconstruindo dados...")

        model_path = os.path.join(models_dir, "best_model.pkl")
        if not os.path.exists(model_path):
            print(f"  [ERRO] Modelo nao encontrado: {model_path}")
            print("  Execute primeiro: python -m src.modeling_generator")
            sys.exit(1)

        pipeline = joblib.load(model_path)
        print(f"  [OK] Modelo carregado: {model_path}")

        # Ler o nome do modelo do summary
        summary_path = os.path.join(metrics_dir, "best_model_summary.json")
        with open(summary_path, "r", encoding="utf-8") as f:
            best_summary = json.load(f)
        best_model_name = best_summary.get("best_model", "Modelo Desconhecido")

        # Reconstruir dados com os mesmos splits deterministicos
        df_raw = load_wine_data()
        df_target = create_target_variable(df_raw)
        df_engineered = engineer_features(df_target)
        id_cols = [c for c in ["Id", "id", "ID"] if c in df_engineered.columns]
        X_train, X_test, y_train, y_test = split_data(
            df_engineered, target_col="high_quality", drop_cols=id_cols,
            test_size=0.2, random_state=42,
        )
        feature_names = list(X_test.columns)
        print(f"  [OK] Dados reconstruidos: {X_test.shape[0]} amostras de teste, "
              f"{len(feature_names)} features.")

        # =================================================================
        # ETAPA 2: CALCULAR SHAP VALUES
        # =================================================================
        print("\n[SHAP] Etapa 2: Calculando SHAP values...")
        shap_values, explainer, X_processed = compute_shap_values(pipeline, X_test)
        print(f"  [OK] SHAP values calculados. Shape: {shap_values.shape}")

        # =================================================================
        # ETAPA 3: GERAR GRAFICOS SHAP
        # =================================================================
        print("\n[SHAP] Etapa 3: Gerando graficos de explicabilidade...")

        plot_shap_summary(
            shap_values, X_processed, feature_names,
            save_path=os.path.join(figures_dir, "shap_summary.png"),
        )
        plot_shap_bar(
            shap_values, feature_names,
            save_path=os.path.join(figures_dir, "shap_bar.png"),
        )

        # Dependence plots para top 3 features
        top_features = ["alcohol", "volatile acidity", "sulphates"]
        for feat in top_features:
            if feat in feature_names:
                safe_name = feat.replace(" ", "_")
                plot_shap_dependence(
                    shap_values, X_processed, feature_names, feat,
                    save_path=os.path.join(figures_dir, f"shap_dependence_{safe_name}.png"),
                )

        # =================================================================
        # ETAPA 4: GERAR RANKING E METRICAS
        # =================================================================
        print("\n[SHAP] Etapa 4: Gerando ranking e metricas...")
        ranking = generate_shap_ranking(shap_values, feature_names)

        # Salvar ranking JSON
        ranking_path = os.path.join(metrics_dir, "shap_feature_ranking.json")
        with open(ranking_path, "w", encoding="utf-8") as f:
            json.dump(ranking, f, indent=4, ensure_ascii=False)
        print(f"  [OK] Ranking salvo: {ranking_path}")

        # Salvar insights JSON
        insights_path = os.path.join(metrics_dir, "shap_insights.json")
        insights_data = {
            "model": best_model_name,
            "method": "SHAP TreeExplainer",
            "top_5_features": [
                {
                    "feature": r["feature"],
                    "label": r["feature_label"],
                    "importance": r["mean_abs_shap"],
                    "insight": r["insight"],
                }
                for r in ranking[:5]
            ],
        }
        with open(insights_path, "w", encoding="utf-8") as f:
            json.dump(insights_data, f, indent=4, ensure_ascii=False)
        print(f"  [OK] Insights salvo: {insights_path}")

        # =================================================================
        # ETAPA 5: GERAR RELATORIO DE INTERPRETABILIDADE
        # =================================================================
        print("\n[SHAP] Etapa 5: Gerando relatorio de interpretabilidade...")
        report_path = os.path.join(root_dir, "results", "interpretability_report.md")
        generate_interpretability_report(ranking, best_model_name, report_path)

        # =================================================================
        # RESUMO FINAL
        # =================================================================
        print("\n" + "=" * 65)
        print(" PIPELINE DE INTERPRETABILIDADE CONCLUIDA COM SUCESSO")
        print("=" * 65)
        print(f"\n  [*] Modelo analisado: {best_model_name}")
        print(f"  [*] Top feature: {ranking[0]['feature_label']} "
              f"(|SHAP| = {ranking[0]['mean_abs_shap']:.4f})")
        print(f"  [*] Relatorio: {report_path}")
        print(f"\n  Graficos em: {figures_dir}")
        print(f"  Metricas em: {metrics_dir}")

    except Exception as e:
        print(f"\n[ERRO CRITICO] Falha na pipeline de interpretabilidade: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
