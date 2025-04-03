# Seleção do Modelo

## Métricas Obtidas
- Log Loss: 0.7562
- F1-Score: 0.4695

## Justificativa da Escolha
1. **Métricas de Performance**
   - Log Loss indica calibração das probabilidades
   - F1-Score mostra balanço entre precisão e recall

2. **Características do Modelo**
   - Árvore de Decisão oferece boa interpretabilidade
   - Capaz de capturar relações não-lineares
   - Treinamento e inferência eficientes

3. **Considerações Práticas**
   - Fácil manutenção e atualização
   - Baixo custo computacional
   - Bom para produção

## Monitoramento
1. **Métricas Principais**
   - Acompanhar Log Loss e F1-Score
   - Monitorar drift nas features
   - Avaliar feedback dos usuários

2. **Gatilhos para Retreinamento**
   - Degradação significativa das métricas
   - Mudanças na distribuição dos dados
   - Novos padrões nos dados
