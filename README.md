# Projeto de Ciência de Dados - Preditor de Arremessos do Kobe Bryant

## Visão Geral
Este projeto segue a estrutura do Team Data Science Process (TDSP) da Microsoft e tem como objetivo desenvolver um preditor de arremessos do Kobe Bryant ("Black Mamba") utilizando duas abordagens:

1. Regressão Linear - Para prever a probabilidade de acerto
2. Classificação - Para prever se o arremesso será convertido ou não

## Diagrama do Pipeline

```mermaid
graph TD
    A[Aquisição de Dados] --> B[Preparação dos Dados]
    B --> C[Análise Exploratória]
    
    subgraph Preparação
        B --> B1[Limpeza]
        B1 --> B2[Feature Engineering]
        B2 --> B3[Split Train/Test]
    end
    
    subgraph Modelagem
        C --> D1[Treino Regressão]
        C --> D2[Treino Classificação]
        D1 --> E1[Avaliação Regressão]
        D2 --> E2[Avaliação Classificação]
        E1 --> F[Seleção do Melhor Modelo]
        E2 --> F
    end
    
    subgraph Produção
        F --> G[Deploy do Modelo]
        G --> H[API REST]
        G --> I[Dashboard Streamlit]
        H --> J[Monitoramento]
        I --> J
    end
    
    subgraph MLOps
        J --> K[Análise de Drift]
        K --> L{Necessita Retreino?}
        L -->|Sim| M[Retreinamento]
        M --> G
        L -->|Não| J
    end
```

## Detalhamento das Etapas

### 1. Aquisição e Preparação
- Coleta dos dados históricos dos arremessos
- Limpeza e tratamento de dados ausentes
- Feature engineering
- Divisão treino/teste (80/20)

### 2. Modelagem
- Desenvolvimento paralelo de dois modelos:
  - Regressão Linear (probabilidade)
  - Classificação (decisão binária)
- Avaliação e comparação de performance
- Seleção do modelo final

### 3. Produção
- Deploy do modelo via API REST
- Interface Streamlit para visualização
- Sistema de logs e monitoramento
- Pipeline de retreinamento automático

### 4. MLOps
- Monitoramento contínuo de métricas
- Detecção de data drift
- Retreinamento automático quando necessário
- Versionamento de modelos com MLflow

## Ferramentas e Pipelines

### MLFlow
- **Rastreamento de Experimentos**
  - Registro automático de parâmetros, métricas e artefatos
  - Versionamento de modelos com tags e notas
  - Comparação visual entre diferentes experimentos
  - Organização por experimentos e runs

- **Monitoramento**
  - Métricas de performance em tempo real
  - Logs detalhados de execução
  - Visualização de resultados
  - Registro de dependências e ambiente

- **Atualização de Modelo**
  - Sistema de versionamento de modelos
  - Registro de linhagem dos modelos
  - Rollback facilitado
  - Transição suave entre versões

### PyCaret
- **Funções de Treinamento**
  - Preparação automática de dados
  - Seleção e comparação de modelos
  - Otimização automática de hiperparâmetros
  - Validação cruzada integrada

- **Pipeline de Modelagem**
  - Preprocessamento automático
  - Feature selection integrado
  - Ensemble de modelos
  - Interpretação de resultados

### Scikit-Learn
- **Pipeline de Modelagem**
  - Preprocessamento padronizado de dados
  - Feature engineering consistente
  - Validação cruzada estratificada
  - Métricas de avaliação

- **Funções de Treinamento**
  - Implementação dos algoritmos base
  - Otimização de hiperparâmetros (GridSearchCV)
  - Transformação de features
  - Avaliação de performance

### Streamlit
- **Interface e Deployment**
  - Dashboard interativo em tempo real
  - Visualização de métricas e resultados
  - Interface para novas predições
  - Monitoramento visual do modelo

- **Monitoramento da Saúde**
  - Visualização de drift em features
  - Acompanhamento de métricas
  - Alertas de performance
  - Análise de predições

### Integração das Ferramentas

1. **Pipeline de Desenvolvimento**
   ```mermaid
   graph LR
       A[Dados] --> B[PyCaret]
       B --> C[Scikit-Learn]
       B --> D[MLFlow]
       D --> E[Registro de Experimentos]
       C --> D
   ```

2. **Pipeline de Produção**
   ```mermaid
   graph LR
       A[MLFlow] --> B[Modelo em Produção]
       B --> C[Streamlit]
       B --> D[Monitoramento]
       D --> E[Análise de Drift]
       E --> F{Retreino}
       F -->|Sim| G[PyCaret]
       G --> A
   ```

### Benefícios da Integração

1. **Desenvolvimento Ágil**
   - PyCaret acelera a experimentação
   - Scikit-Learn garante robustez
   - MLFlow mantém rastreabilidade

2. **Produção Robusta**
   - Streamlit facilita visualização
   - MLFlow gerencia versões
   - Monitoramento automatizado

3. **Manutenção Simplificada**
   - Detecção automática de problemas
   - Retreinamento facilitado
   - Rollback rápido quando necessário

## Ferramentas Utilizadas
- MLFlow: Tracking de experimentos
- PyCaret: Desenvolvimento dos modelos
- Scikit-learn: Preprocessamento e avaliação
- Streamlit: Interface de usuário
- FastAPI: API REST

