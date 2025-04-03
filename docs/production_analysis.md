# Análise do Modelo em Produção

## Métricas de Teste
- Log Loss: 0.7562
- F1-Score: 0.4695

## Estatísticas da Aplicação
- Total de amostras em produção: 6426
- Amostras com rótulo: 6426
- Porcentagem rotulada: 100.00%

## Análise de Data Drift

### Mudanças por Feature
- lat:
  - Mudança na média: 0.31%
  - Mudança no desvio: 5.97%
- lon:
  - Mudança na média: 0.00%
  - Mudança no desvio: 43.42%
- minutes_remaining:
  - Mudança na média: 15.70%
  - Mudança no desvio: 0.12%
- period:
  - Mudança na média: 7.39%
  - Mudança no desvio: 0.25%
- playoffs:
  - Mudança na média: 5.70%
  - Mudança no desvio: 2.41%
- shot_distance:
  - Mudança na média: 90.72%
  - Mudança no desvio: 56.73%

## Conclusões e Recomendações
1. **Performance do Modelo**
   - O modelo apresenta Log Loss de 0.7562 nos dados de teste
   - F1-Score de 0.4695 indica necessidade de melhorias

2. **Estabilidade dos Dados**
   Detectado drift significativo em algumas features

3. **Próximos Passos**
   Investigar causas da baixa performance e considerar retreinamento
