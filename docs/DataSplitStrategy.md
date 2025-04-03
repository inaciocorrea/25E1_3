# Estratégia de Divisão dos Dados

## Escolha do Método de Divisão

### 1. Divisão Estratificada
A escolha por uma divisão estratificada (stratify=y) foi feita para:
- Manter a proporção de classes em treino e teste
- Evitar viés de amostragem
- Garantir representatividade das classes minoritárias

### 2. Proporção 80/20
A escolha de 80% para treino e 20% para teste:
- Fornece dados suficientes para treino
- Mantém conjunto de teste representativo
- Equilibra viés e variância

## Impacto no Modelo Final

### Como a Divisão Afeta o Resultado

1. **Representatividade**
   - A divisão estratificada garante que o modelo seja treinado com uma distribuição similar à real
   - Reduz o risco de overfitting em classes específicas

2. **Generalização**
   - O conjunto de teste independente permite avaliar a capacidade de generalização
   - Ajuda a identificar problemas de overfitting ou underfitting

## Estratégias para Minimizar Viés

1. **Validação Cruzada**
   - K-fold cross validation no treino
   - Avaliação em diferentes divisões

2. **Random State Fixo**
   - Garante reprodutibilidade
   - Permite comparações consistentes

3. **Monitoramento de Distribuição**
   - Verifica similaridade entre treino e teste
   - Identifica possíveis data drifts

4. **Balanceamento de Classes**
   - Técnicas como SMOTE se necessário
   - Garante representatividade

5. **Amostragem Temporal** (se aplicável)
   - Considerar a ordem temporal dos dados
   - Evitar data leakage

## Métricas de Qualidade

Para garantir a qualidade da divisão, monitoramos:
1. Proporção das classes em cada conjunto
2. Distribuição das features
3. Tamanho adequado de cada conjunto
4. Independência entre conjuntos 