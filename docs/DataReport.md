# Relatório de Dados - Kobe Bryant Shot Selection

## Visão Geral dos Dados

### Fonte dos Dados
- Dataset de arremessos do Kobe Bryant
- Formato: Parquet
- Localização: Data/raw/

### Dimensões
- Base de Desenvolvimento: {initial_dev_rows} × {initial_dev_cols}
- Base de Produção: {initial_prod_rows} × {initial_prod_cols}
- Total após processamento: {processed_rows} registros

## Features

### Features Selecionadas
1. **Localização**
   - lat: latitude do arremesso
   - lon: longitude do arremesso
   - shot_distance: distância do arremesso

2. **Contexto do Jogo**
   - minutes_remaining: tempo restante
   - period: período do jogo
   - playoffs: indicador de playoffs (0/1)

3. **Target**
   - shot_made_flag: indicador de sucesso do arremesso (0/1)

## Qualidade dos Dados

### Pré-processamento
- Remoção de valores ausentes
- Seleção de features relevantes
- Divisão treino/teste (80/20)

### Estatísticas Descritivas
- Distribuição balanceada de classes
- Features numéricas normalizadas
- Dados temporais preservados

## Análise de Data Drift

### Features Críticas
- shot_distance
- period
- playoffs

### Monitoramento
- Mudanças na média > 20%
- Mudanças no desvio padrão > 20%
- Status de drift por feature

## Recomendações
1. Monitoramento contínuo das features críticas
2. Coleta periódica de novos dados
3. Validação da qualidade dos dados
4. Atualização do modelo quando necessário
