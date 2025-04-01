from nodes import (prepare_data, train_models, evaluate_models, 
                  make_predictions, analyze_data_drift)
import yaml
import mlflow
import os
import pandas as pd

def load_config():
    """Carrega as configurações do pipeline"""
    with open("Config/mlflow.yml", "r") as f:
        mlflow_config = yaml.safe_load(f)
    
    with open("Config/catalog.yml", "r") as f:
        data_catalog = yaml.safe_load(f)
    
    return mlflow_config, data_catalog

def create_directories():
    """Cria a estrutura de diretórios necessária"""
    directories = [
        "Data/raw",
        "Data/processed",
        "Code/Model",
        "Docs",
        "mlruns"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def generate_reports(reports_config, metrics, drift_report, predictions, true_values):
    """Gera relatórios de análise"""
    # Usa as métricas do modelo de regressão logística para o relatório
    lr_metrics = metrics['logistic_regression']
    
    with open(reports_config['drift_analysis'], "w") as f:
        f.write("""# Análise do Modelo em Produção

## Métricas de Teste
- Log Loss: {:.4f}
- F1-Score: {:.4f}

## Estatísticas da Aplicação
- Total de amostras em produção: {}
- Amostras com rótulo: {}
- Porcentagem rotulada: {:.2f}%

## Análise de Data Drift

### Mudanças por Feature
{}

## Conclusões e Recomendações
1. **Performance do Modelo**
   - O modelo apresenta Log Loss de {:.4f} nos dados de teste
   - F1-Score de {:.4f} indica {}

2. **Estabilidade dos Dados**
   {}

3. **Próximos Passos**
   {}
""".format(
    lr_metrics['log_loss'],
    lr_metrics['f1_score'],
    len(predictions),
    len(true_values) if true_values is not None else 0,
    (len(true_values) / len(predictions) * 100) if true_values is not None else 0,
    '\n'.join([f"- {feature}:\n  - Mudança na média: {stats['mean_change']:.2f}%\n  - Mudança no desvio: {stats['std_change']:.2f}%" 
               for feature, stats in drift_report['feature_stats'].items()]),
    lr_metrics['log_loss'],
    lr_metrics['f1_score'],
    "performance adequada" if lr_metrics['f1_score'] > 0.5 else "necessidade de melhorias",
    "Dados de produção apresentam distribuição similar aos dados de treino" 
    if all(stats['mean_change'] < 10 for stats in drift_report['feature_stats'].values())
    else "Detectado drift significativo em algumas features",
    "Monitorar performance e coletar mais dados" if lr_metrics['f1_score'] > 0.5
    else "Investigar causas da baixa performance e considerar retreinamento"
))

    # Salva relatório de seleção do modelo
    with open(reports_config['model_selection'], "w") as f:
        f.write("""# Seleção do Modelo

## Métricas Obtidas
- Log Loss: {:.4f}
- F1-Score: {:.4f}

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
""".format(
    metrics['logistic_regression']['log_loss'],
    metrics['logistic_regression']['f1_score']
))

def run_pipeline():
    """Executa o pipeline completo"""
    # Primeiro cria os diretórios
    create_directories()
    
    # Verifica se os arquivos de dados existem
    if not os.path.exists("Data/raw/dataset_kobe_dev.parquet"):
        raise FileNotFoundError(
            "Arquivo 'dataset_kobe_dev.parquet' não encontrado em 'Data/raw/'. "
            "Por favor, coloque os arquivos de dados no diretório correto."
        )
    
    if not os.path.exists("Data/raw/dataset_kobe_prod.parquet"):
        raise FileNotFoundError(
            "Arquivo 'dataset_kobe_prod.parquet' não encontrado em 'Data/raw/'. "
            "Por favor, coloque os arquivos de dados no diretório correto."
        )
    
    # Carrega configurações
    mlflow_config, data_catalog = load_config()
    
    # Configura MLflow
    os.makedirs(mlflow_config['tracking_uri'], exist_ok=True)
    mlflow.set_tracking_uri(f"file:{mlflow_config['tracking_uri']}")
    mlflow.set_experiment(mlflow_config['experiment_name'])
    
    # Executa os nós do pipeline
    with mlflow.start_run(run_name="PreparacaoDados"):
        # Carrega e prepara dados
        train_df, test_df = prepare_data(
            data_catalog['dev_data'],
            data_catalog['prod_data']
        )
        
        # Salva datasets
        train_df.to_parquet("Data/processed/data_filtered_train.parquet")
        test_df.to_parquet("Data/processed/data_filtered_test.parquet")
        
        # Salva também o dataset completo se necessário
        full_df = pd.concat([train_df, test_df])
        full_df.to_parquet("Data/processed/data_filtered.parquet")
        
        # Registra dimensões do dataset filtrado
        mlflow.log_params({
            'filtered_rows': full_df.shape[0],
            'filtered_cols': full_df.shape[1],
            'train_rows': train_df.shape[0],
            'test_rows': test_df.shape[0]
        })
        
        # Treina os modelos
        models = train_models(data_catalog['train_data'], data_catalog['model_paths'])
        
        # Registra o modelo no MLflow
        mlflow.sklearn.log_model(
            models[0],  # modelo de regressão logística
            "logistic_regression",
            registered_model_name="logistic_regression"
        )
        
        # Transiciona o modelo para produção
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name="logistic_regression",
            version=1,  # primeira versão
            stage="Production"
        )
        
        # Avalia os modelos
        metrics, predictions = evaluate_models(models, data_catalog['test_data'])
        
        # Registra métricas de teste para ambos os modelos
        for model_name, model_metrics in metrics.items():
            for metric_name, metric_value in model_metrics.items():
                if isinstance(metric_value, (int, float)):  # apenas métricas numéricas
                    # Adicionar log para debug
                    print(f"Registrando métrica: {model_name}_{metric_name}: {metric_value}")
                    mlflow.log_metric(f"{model_name}_{metric_name}", metric_value)
        
        # Faz predições em produção (usando o modelo de regressão logística)
        prod_predictions, true_values = make_predictions(models[0], data_catalog['prod_data'])
        prod_predictions.to_parquet(data_catalog['predictions_output'])
        
        # Analisa drift e gera relatório
        drift_report = analyze_data_drift(train_df, prod_predictions)
        
        # Salva relatórios
        generate_reports(
            data_catalog['reports'],
            metrics,
            drift_report,
            prod_predictions,
            true_values
        )

        # Registro mais detalhado de parâmetros e métricas
        mlflow.log_params({
            'test_size': 0.2,
            'random_state': 42,
            'stratify': True
        })
        
        mlflow.log_metrics({
            'filtered_rows': full_df.shape[0],
            'filtered_cols': full_df.shape[1],
            'train_rows': train_df.shape[0],
            'test_rows': test_df.shape[0]
        })

if __name__ == "__main__":
    run_pipeline() 