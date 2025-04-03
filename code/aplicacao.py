import mlflow
import pandas as pd
from sklearn.metrics import log_loss, f1_score
from nodes import analyze_data_drift
from config import data_processed_path, data_raw_path

def run_application_pipeline():
    """Pipeline de aplicação do modelo em produção"""
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    with mlflow.start_run(run_name="PipelineAplicacao"):
        # Carrega dados de produção
        prod_data = pd.read_parquet(data_raw_path + "/dataset_kobe_prod.parquet")
        
        # Seleciona apenas as features necessárias
        selected_features = [
            'lat', 'lon', 'minutes_remaining', 'period',
            'playoffs', 'shot_distance', 'shot_made_flag'
        ]
        prod_data = prod_data[selected_features]
        
        # Remove linhas com dados faltantes
        prod_data = prod_data.dropna()
        
        # Separa o target
        target = prod_data['shot_made_flag']
        features = prod_data.drop('shot_made_flag', axis=1)
        
        # Carrega o modelo
        model = mlflow.sklearn.load_model("models:/logistic_regression/Production")
        
        # Faz predições
        predictions = model.predict(features)
        
        # Cria DataFrame com resultados - incluindo todas as features
        results_df = prod_data.copy()  # Mantém todas as features
        results_df['prediction'] = predictions
        results_df['prediction_score'] = predictions  # Para compatibilidade com o dashboard
        results_df['prediction_label'] = [1 if p > 0.5 else 0 for p in predictions]
        
        # Salva resultados
        results_df.to_parquet(data_processed_path + "/production_predictions.parquet")
        mlflow.log_artifact(data_processed_path + "/production_predictions.parquet")

        # Calcula e registra métricas
        metrics = {
            'prod_log_loss': log_loss(target, predictions),
            'prod_f1_score': f1_score(target, [1 if p > 0.5 else 0 for p in predictions])
        }
        
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        # Carrega dados de treino para comparação
        train_data = pd.read_parquet(data_processed_path + "/data_filtered_train.parquet")
        
        # Analisa drift
        drift_report = analyze_data_drift(train_data, results_df)  # Usando results_df aqui
        
        # Analisa aderência
        adherence_report = analyze_model_adherence(train_data, results_df, drift_report)
        mlflow.log_dict(adherence_report, "adherence_report.json")

def analyze_model_adherence(train_data, prod_data, drift_report):
    """Analisa aderência do modelo aos dados de produção"""
    adherence_report = {
        'significant_changes': [],
        'stability_status': 'stable',
        'recommendations': []
    }
    
    # Analisa mudanças significativas
    for feature, stats in drift_report['feature_stats'].items():
        if stats['needs_attention']:
            adherence_report['significant_changes'].append({
                'feature': feature,
                'mean_change': stats['mean_change'],
                'std_change': stats['std_change']
            })
    
    # Avalia estabilidade geral
    if len(adherence_report['significant_changes']) > 2:
        adherence_report['stability_status'] = 'unstable'
        adherence_report['recommendations'].append('Consider retraining')
    
    return adherence_report

if __name__ == "__main__":
    run_application_pipeline() 