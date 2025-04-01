import pandas as pd
import numpy as np
from pycaret.classification import *
from sklearn.metrics import log_loss, f1_score
from sklearn.model_selection import train_test_split
import mlflow

def prepare_data(dev_data_path, prod_data_path, test_size=0.2):
    """Prepara os dados para treinamento"""
    # Carrega os dados
    dev_data = pd.read_parquet(dev_data_path)
    prod_data = pd.read_parquet(prod_data_path)
    
    # Registra dimensões iniciais
    initial_dims = {
        'dev_data': dev_data.shape,
        'prod_data': prod_data.shape
    }
    
    # Combina os datasets
    df = pd.concat([dev_data, prod_data], axis=0)
    
    # Seleciona as colunas específicas
    selected_columns = [
        'lat', 'lon', 'minutes_remaining', 'period',
        'playoffs', 'shot_distance', 'shot_made_flag'
    ]
    df = df[selected_columns]
    
    # Remove linhas com dados faltantes
    df = df.dropna()
    
    # Registra dimensões após pré-processamento
    processed_dims = df.shape
    
    # Divide os dados
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df['shot_made_flag']
    )
    
    train_df.to_parquet("Data/processed/base_train.parquet")
    test_df.to_parquet("Data/processed/base_test.parquet")
    
    # Registra dimensões finais
    final_dims = {
        'train': train_df.shape,
        'test': test_df.shape
    }
    
    # Registra todas as dimensões no MLflow
    mlflow.log_params({
        'initial_dev_rows': initial_dims['dev_data'][0],
        'initial_dev_cols': initial_dims['dev_data'][1],
        'initial_prod_rows': initial_dims['prod_data'][0],
        'initial_prod_cols': initial_dims['prod_data'][1],
        'processed_rows': processed_dims[0],
        'processed_cols': processed_dims[1],
        'train_rows': final_dims['train'][0],
        'train_cols': final_dims['train'][1],
        'test_rows': final_dims['test'][0],
        'test_cols': final_dims['test'][1]
    })
    
    return train_df, test_df

def train_models(train_data_path, model_paths):
    """Treina os modelos de regressão logística e árvore de decisão"""
    train_data = pd.read_parquet(train_data_path)
    
    # Setup inicial do PyCaret
    setup(
        data=train_data,
        target='shot_made_flag',
        session_id=42,
        verbose=False,
        log_experiment=False
    )
    
    # Treina regressão logística
    lr_model = create_model('lr', verbose=False)
    save_model(lr_model, model_paths['logistic'])
    
    # Treina árvore de decisão
    dt_model = create_model('dt', verbose=False)
    save_model(dt_model, model_paths['decision_tree'])
    
    return lr_model, dt_model

def evaluate_models(models, test_data_path):
    """Avalia ambos os modelos com métricas expandidas"""
    test_data = pd.read_parquet(test_data_path)
    lr_model, dt_model = models
    
    metrics = {
        'logistic_regression': {},
        'decision_tree': {}
    }
    
    predictions = {}  # Dicionário para armazenar as predições de cada modelo
    
    for model_name, model in zip(['logistic_regression', 'decision_tree'], [lr_model, dt_model]):
        model_predictions = predict_model(model, data=test_data)
        predictions[model_name] = model_predictions
        
        # Métricas básicas
        metric_value = log_loss(test_data['shot_made_flag'], model_predictions['prediction_score'])
        metrics[model_name] = {
            'log_loss': metric_value,
            'f1_score': f1_score(test_data['shot_made_flag'], model_predictions['prediction_label']),
            'timestamp': pd.Timestamp.now().isoformat(),
            'samples_evaluated': len(test_data),
            'needs_retraining': False
        }
        
        # Verifica necessidade de retreinamento
        if (metrics[model_name]['f1_score'] < 0.45 or 
            metrics[model_name]['log_loss'] > 0.7):
            metrics[model_name]['needs_retraining'] = True
            metrics[model_name]['retraining_reason'] = 'performance_degradation'
        
        # Registra métricas no MLflow
        mlflow.log_metric(f"{model_name}_log_loss", metric_value)
    
    return metrics, predictions

def make_predictions(model, prod_data_path):
    """Faz predições em produção"""
    prod_data = pd.read_parquet(prod_data_path)
    
    # Remove a coluna target se existir
    if 'shot_made_flag' in prod_data.columns:
        true_values = prod_data['shot_made_flag'].copy()
        prod_data = prod_data.drop('shot_made_flag', axis=1)
    else:
        true_values = None
    
    predictions = predict_model(model, data=prod_data)
    
    return predictions, true_values

def analyze_data_drift(train_data, prod_data):
    """Analisa mudanças entre dados de treino e produção"""
    drift_report = {
        'feature_stats': {},
        'distribution_changes': {},
        'drift_status': 'stable'
    }
    
    features = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance']
    critical_features = ['shot_distance', 'period', 'playoffs']  # features mais importantes
    
    drift_detected = False
    for feature in features:
        train_stats = train_data[feature].describe()
        prod_stats = prod_data[feature].describe()
        
        mean_change = abs((prod_stats['mean'] - train_stats['mean']) / train_stats['mean']) * 100
        std_change = abs((prod_stats['std'] - train_stats['std']) / train_stats['std']) * 100
        
        drift_report['feature_stats'][feature] = {
            'mean_change': mean_change,
            'std_change': std_change,
            'is_critical': feature in critical_features,
            'needs_attention': mean_change > 20 or std_change > 20
        }
        
        # Verifica drift em features críticas
        if feature in critical_features and (mean_change > 20 or std_change > 20):
            drift_detected = True
    
    drift_report['drift_status'] = 'warning' if drift_detected else 'stable'
    drift_report['retraining_recommended'] = drift_detected
    
    return drift_report 