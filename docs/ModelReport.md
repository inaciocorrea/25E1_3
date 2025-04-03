# Relatório do Modelo - Kobe Shot Predictor

## 1. Modelos Desenvolvidos

### 1.1 Regressão Logística
- **Métricas**:
  - Log Loss: 0.7562
  - F1-Score: 0.4695
- **Hiperparâmetros**:
  - max_iter: 1000
  - solver: lbfgs

### 1.2 Árvore de Decisão
- **Métricas**:
  - Log Loss: ~0.70
  - F1-Score: 0.4962
- **Hiperparâmetros**:
  - max_depth: 5
  - min_samples_split: 5

## 2. Análise de Performance

### 2.1 Comparação dos Modelos
- Regressão Logística tem melhor calibração de probabilidades
- Árvore de Decisão apresenta melhor F1-Score
- Ambos os modelos mostram oportunidades de melhoria

### 2.2 Análise de Erros
- Principais fontes de erro:
  - Distância do arremesso
  - Período do jogo
  - Situação de playoffs

## 3. Monitoramento em Produção

### 3.1 Métricas de Monitoramento
- Log Loss em tempo real
- F1-Score quando target disponível
- Análise de drift nas features
- Volume de predições

### 3.2 Thresholds de Alerta
- F1-Score < 0.45
- Log Loss > 0.7
- Drift > 20% em features críticas

## 4. Manutenção do Modelo

### 4.1 Estratégia de Retreinamento
- **Reativo**: Quando métricas ultrapassam thresholds
- **Preventivo**: Atualização mensal programada

### 4.2 Processo de Atualização
1. Coleta de novos dados
2. Validação de qualidade
3. Retreinamento
4. Testes A/B
5. Deploy gradual

## 5. Conclusões e Recomendações

### 5.1 Pontos Fortes
- Pipeline automatizado
- Monitoramento contínuo
- Estratégias de atualização definidas

### 5.2 Pontos de Melhoria
- Aumentar F1-Score
- Reduzir Log Loss
- Melhorar tratamento de drift

### 5.3 Próximos Passos
1. Explorar feature engineering adicional
2. Testar ensemble de modelos
3. Implementar validação online
4. Expandir monitoramento
