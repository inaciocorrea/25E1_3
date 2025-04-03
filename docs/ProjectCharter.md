# Project Charter - Preditor de Arremessos do Kobe Bryant

## Visão Geral do Negócio

### Problema
Desenvolver um modelo preditivo capaz de analisar as características dos arremessos do Kobe Bryant e prever se um arremesso será convertido ou não, utilizando tanto abordagem de regressão quanto classificação.

### Objetivo
- Criar um modelo de regressão para prever a probabilidade de acerto de um arremesso
- Criar um modelo de classificação para prever se um arremesso será convertido ou não
- Identificar os fatores mais importantes que influenciam o sucesso dos arremessos
- Comparar o desempenho entre as abordagens de regressão e classificação

## Escopo

### Incluído
- Análise dos dados históricos dos arremessos do Kobe Bryant
- Desenvolvimento de dois modelos (regressão e classificação)
- Avaliação comparativa entre os modelos
- API para disponibilização das predições
- Documentação completa do processo

### Excluído
- Análise de dados de outros jogadores
- Análise de vídeos dos arremessos
- Interface gráfica para o usuário final

## Metodologia
- Utilização do framework TDSP (Team Data Science Process)
- Desenvolvimento iterativo dos modelos
- Validação cruzada para avaliação de performance
- Testes A/B para comparação dos modelos

## Métricas de Sucesso

### Métricas de Negócio
- Acurácia mínima de 70% na previsão dos arremessos
- Tempo de resposta da API inferior a 100ms
- Facilidade de interpretação dos resultados

### Métricas Técnicas
#### Regressão
- R² > 0.6
- RMSE < 0.3
- MAE < 0.25

#### Classificação
- Acurácia > 0.7
- F1-Score > 0.7
- AUC-ROC > 0.75

## Planejamento

### Cronograma
1. **Fase 1: Preparação dos Dados** (1 semana)
   - Análise exploratória
   - Pré-processamento
   - Feature engineering

2. **Fase 2: Modelagem** (2 semanas)
   - Desenvolvimento do modelo de regressão
   - Desenvolvimento do modelo de classificação
   - Otimização de hiperparâmetros

3. **Fase 3: Avaliação** (1 semana)
   - Validação dos modelos
   - Comparação de performance
   - Ajustes finais

4. **Fase 4: Implantação** (1 semana)
   - Desenvolvimento da API
   - Documentação
   - Testes de integração

### Recursos Necessários
- Cientista de Dados
- Ambiente de desenvolvimento Python
- Infraestrutura para deploy do modelo
- Acesso aos dados históricos dos arremessos

## Stakeholders
- Equipe de Ciência de Dados
- Analistas de Basketball
- Treinadores
- Entusiastas do esporte

## Riscos e Mitigação

### Riscos
1. **Desbalanceamento dos dados**
   - Mitigação: Utilização de técnicas de balanceamento (SMOTE, undersampling)

2. **Overfitting**
   - Mitigação: Validação cruzada e regularização adequada

3. **Performance insatisfatória**
   - Mitigação: Experimentação com diferentes algoritmos e feature engineering

4. **Problemas de dados**
   - Mitigação: Validação rigorosa e limpeza dos dados

## Aprovações
- [ ] Cientista de Dados Líder
- [ ] Gerente do Projeto
- [ ] Stakeholders principais
