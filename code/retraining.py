def define_retraining_strategies():
    """Define estratégias de retreinamento"""
    return {
        'reactive': {
            'triggers': [
                'f1_score < 0.45',
                'log_loss > 0.7',
                'drift > 20% in critical features'
            ],
            'actions': [
                'collect_new_data',
                'retrain_model',
                'validate_performance',
                'gradual_deployment'
            ]
        },
        'predictive': {
            'monitoring': [
                'trend_analysis',
                'early_drift_detection',
                'seasonality_patterns'
            ],
            'actions': [
                'scheduled_retraining',
                'incremental_updates',
                'a_b_testing',
                'continuous_validation'
            ]
        }
    } 