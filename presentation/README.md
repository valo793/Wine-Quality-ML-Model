# Planejamento da Apresentação Executiva e Material Visual

Este diretório servirá como repositório para os materiais da futura apresentação executiva (HTML/JS/CSS) e suporte para a gravação do pitch de até 5 minutos.

---

## 1. Storytelling Executivo (Narrativa de Negócio)
Para que a apresentação chame a atenção de diretores e tomadores de decisão, a narrativa deve focar em **valor e resultados**, evitando o jargão puramente matemático de Machine Learning.

A estrutura do storytelling responderá às seguintes perguntas:
1. **Qual problema de negócio estamos resolvendo?**
   * *Resposta*: Otimização do controle de qualidade de lotes de vinho na produção. Garantir previsibilidade e precificação correta de lotes premium antes mesmo do engarrafamento final.
2. **Como os dados físico-químicos ajudam na decisão?**
   * *Resposta*: Em vez de depender apenas de testes sensoriais humanos (subjetivos, lentos e caros), usamos dados objetivos (químicos) obtidos rapidamente na linha de produção para triagem imediata.
3. **Quais características diferenciam vinhos de alta qualidade?**
   * *Resposta*: Identificar a influência direta de acidez volátil, teor alcoólico e dióxido de enxofre na percepção de qualidade do produto.
4. **Qual modelo teve o melhor desempenho e como funciona?**
   * *Resposta*: Traduzir a performance dos modelos (comparativo de acurácia, F1-score e matriz de confusão) em impactos reais (falsos positivos e falsos negativos na produção).
5. **Como a vinícola poderia usar isso na prática?**
   * *Resposta*: Aplicação prática de uma régua de decisão de negócios no processo produtivo.

---

## 2. Régua de Decisão de Negócio (Proposta de Inovação)
Para agregar valor ao modelo de ML, propomos a implementação de uma régua de ação com base na probabilidade do modelo:

| Probabilidade do Modelo | Classificação de Lote | Ação de Negócio Recomendada |
| :--- | :--- | :--- |
| **>= 80%** | **Alto Potencial Premium** | Lote direcionado para linha de envelhecimento especial e precificação elevada. |
| **50% a 79%** | **Qualidade Intermediária** | Encaminhado para revisão pelo enólogo chefe / ajuste fino químico do lote. |
| **< 50%** | **Lote Comum / Baixa Qualidade**| Destinado para marcas de entrada ou vinho de mesa rápido. |

---

## 3. Arquitetura da Futura Apresentação HTML/JS
A futura apresentação será uma página web dinâmica de alta fidelidade visual, com suporte a modo escuro e gráficos interativos.

### Funcionalidades Planejadas:
1. **Dashboard de Resultados**:
   * Gráficos interativos (usando Chart.js ou D3.js) consumindo as métricas exportadas pela pipeline em Python (via arquivos JSON em `results/metrics/`).
   * Visualização clara das matrizes de confusão e curvas ROC-AUC.
2. **Simulador em Tempo Real**:
   * Uma pequena interface interativa em que o executivo pode alterar os valores químicos (ex: `pH`, `alcohol`, `volatile acidity`) usando sliders, simulando o resultado predito pelo modelo em tempo real (consumindo o modelo exportado).
3. **Seção de Explicabilidade**:
   * Representação visual simples da importância das variáveis baseada no SHAP (convertido em gráficos amigáveis).

---

## 4. Estrutura do Vídeo Executivo (Pitch de 5 minutos)
O vídeo de apoio deve ser dinâmico e focado no problema e na solução:
* **Minuto 0:00 - 0:45**: Introdução do problema (o custo de vinhos que não atingem a qualidade esperada e o gargalo do teste subjetivo).
* **Minuto 0:45 - 2:00**: Solução técnica e os dados (explicação rápida de como os atributos químicos impactam a qualidade, sem exagerar nos detalhes técnicos).
* **Minuto 2:00 - 3:30**: Resultados do modelo e a escolha da métrica F1-score de alta qualidade (mostrando que focamos no acerto seguro dos vinhos premium).
* **Minuto 3:30 - 4:30**: Aplicação prática no dia a dia da vinícola (Régua de Decisão e Simulador Web).
* **Minuto 4:30 - 5:00**: Conclusão e chamada para ação de implementação comercial.
