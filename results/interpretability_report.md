# Relatorio de Interpretabilidade - Analise SHAP

## Contexto

Este relatorio documenta a **Fase 6 (Interpretacao dos Resultados)** do Tech Challenge.
O objetivo e identificar quais variaveis tem maior influencia na classificacao de
qualidade do vinho e discutir as implicacoes praticas para o processo de producao.

- **Modelo Analisado**: Gradient Boosting (Hist)
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
| 1 | Sulfatos (`sulphates`) | 1.9618 |
| 2 | Razao Alcool/Densidade (`alcohol_density_ratio`) | 1.2337 |
| 3 | Acidez Volatil (`volatile acidity`) | 1.1574 |
| 4 | Teor Alcoolico (`alcohol`) | 0.9214 |
| 5 | Razao de Enxofre (Livre/Total) (`sulfur_ratio`) | 0.8208 |
| 6 | Acido Citrico (`citric acid`) | 0.8044 |
| 7 | Densidade (`density`) | 0.7209 |
| 8 | Dioxido de Enxofre Total (`total sulfur dioxide`) | 0.7105 |
| 9 | Cloretos (`chlorides`) | 0.6281 |
| 10 | Dioxido de Enxofre Livre (`free sulfur dioxide`) | 0.5513 |
| 11 | Razao Acucar/Alcool (`sugar_alcohol_ratio`) | 0.4924 |
| 12 | pH (`pH`) | 0.4154 |
| 13 | Acucar Residual (`residual sugar`) | 0.4069 |
| 14 | Equilibrio de Acidez (`acidity_balance`) | 0.3650 |
| 15 | Acidez Fixa (`fixed acidity`) | 0.2317 |


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

### 1. Sulfatos

Os sulfatos contribuem positivamente para a qualidade, atuando como conservantes e antioxidantes. Niveis adequados protegem o aroma e a cor do vinho durante o armazenamento.

### 2. Razao Alcool/Densidade

Feature engenheirada que captura simultaneamente o corpo e a leveza do vinho. Valores elevados indicam vinhos com boa estrutura alcoolica e baixa densidade.

### 3. Acidez Volatil

A acidez volatil (acido acetico) impacta negativamente a qualidade. Niveis elevados indicam problemas de fermentacao ou contaminacao bacteriana, resultando em sabores avinagrados que depreciam o produto.

### 4. Teor Alcoolico

O teor alcoolico e o fator mais influente na classificacao de qualidade. Vinhos com teor alcoolico mais elevado tendem a ser classificados como premium, o que reflete a relacao entre maturacao da uva, fermentacao completa e corpo do vinho.

### 5. Razao de Enxofre (Livre/Total)

A proporcao de enxofre livre sobre o total indica a eficiencia da protecao antioxidante. Razoes mais altas sugerem melhor conservacao.



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
