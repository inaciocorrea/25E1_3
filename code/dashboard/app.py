import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import mlflow
import os
from datetime import datetime
from PIL import Image

# Configuração da página
st.set_page_config(
    page_title="Kobe Shots Monitor",
    page_icon="🏀",
    layout="wide"
)

# Configuração do MLflow
mlflow.set_tracking_uri("file:./mlruns")

# Caminho para a imagem da quadra
COURT_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "assets", "half_court.jpg")

def load_data():
    """Carrega os dados necessários"""
    train_data = pd.read_parquet("Data/processed/base_train.parquet")
    prod_predictions = pd.read_parquet("Data/processed/production_predictions.parquet")
    return train_data, prod_predictions

def get_mlflow_metrics():
    """Recupera métricas do MLflow"""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Treinamento")  # Nome atualizado do experimento
    
    if experiment:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        if runs:
            latest_run = runs[0]
            # Adicionar log para debug
            print("Métricas encontradas:", latest_run.data.metrics)
            return {
                'test_log_loss': latest_run.data.metrics.get('logistic_regression_log_loss', 0),
                'test_f1_score': latest_run.data.metrics.get('logistic_regression_f1_score', 0)
            }
    return {'test_log_loss': 0, 'test_f1_score': 0}

def plot_metric_history(metric_name):
    """Plota histórico de uma métrica"""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Treinamento")
    
    if experiment:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time"]
        )
        
        dates = [datetime.fromtimestamp(run.info.start_time/1000) for run in runs]
        values = [run.data.metrics.get(f"logistic_regression_{metric_name}", 0) for run in runs]
        
        fig = px.line(
            x=dates, 
            y=values,
            title=f"Histórico de {metric_name}",
            labels={"x": "Data", "y": metric_name}
        )
        return fig
    return None

def plot_feature_distribution(train_data, prod_data, feature):
    """Plota distribuição de uma feature"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=train_data[feature],
        name='Treino',
        opacity=0.75
    ))
    
    fig.add_trace(go.Histogram(
        x=prod_data[feature],
        name='Produção',
        opacity=0.75
    ))
    
    fig.update_layout(
        title=f"Distribuição de {feature}",
        xaxis_title=feature,
        yaxis_title="Contagem",
        barmode='overlay'
    )
    
    return fig

def plot_shots_simple(data, title="Mapa de Arremessos"):
    """Plota apenas os pontos de arremesso"""
    fig = go.Figure()
    
    # Define as cores para cada tipo de arremesso
    if 'prediction_score' in data.columns:
        # Dados de produção - usa probabilidade predita
        colors = data['prediction_score'].map(lambda x: 'rgba(0,0,0,0.7)' if pd.isna(x) else 
                                                      'rgba(0,200,0,0.7)' if x >= 0.5 else 
                                                      'rgba(200,0,0,0.7)')
    else:
        # Dados de treino - usa resultado real
        colors = data['shot_made_flag'].map(lambda x: 'rgba(0,0,0,0.7)' if pd.isna(x) else 
                                                     'rgba(0,200,0,0.7)' if x == 1 else 
                                                     'rgba(200,0,0,0.7)')
    
    # Adiciona os pontos (sem nome para não aparecer na legenda)
    fig.add_trace(go.Scatter(
        x=data['lon'],
        y=data['lat'],
        mode='markers',
        name="",  # Remove da legenda
        showlegend=False,  # Garante que não aparece na legenda
        marker=dict(
            size=6,
            color=colors,
            showscale=False,
        ),
        hovertemplate=(
            'Distância: %{customdata[0]:.1f} pés<br>' +
            'Resultado: %{customdata[1]}<br>' +
            'Posição: (%{x:.1f}, %{y:.1f})<br>' +
            '<extra></extra>'
        ),
        customdata=data.apply(lambda row: [
            row['shot_distance'],
            'Desconhecido' if pd.isna(row.get('shot_made_flag', None)) else
            'Cesta' if row.get('shot_made_flag', 0) == 1 else 'Erro'
        ], axis=1).tolist()
    ))
    
    # Adiciona apenas os itens da legenda
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=8, color='rgba(0,200,0,1)'),
        name='Convertido',  # Mudado de "Cesta" para "Convertido"
        showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=8, color='rgba(200,0,0,1)'),
        name='Errado',  # Mudado de "Erro" para "Errado"
        showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=8, color='rgba(0,0,0,1)'),
        name='Sem Informação',  # Mudado de "Desconhecido" para "Sem Informação"
        showlegend=True
    ))
    
    # Configuração do layout
    fig.update_layout(
        title=title,
        width=500,
        height=470,
        showlegend=True,
        legend=dict(
            yanchor="bottom",  # Ancora no bottom
            y=0.01,           # Posição vertical próxima à base
            xanchor="left",   # Ancora à esquerda
            x=0.01,           # Posição horizontal próxima à borda esquerda
            bgcolor='rgba(255,255,255,0.95)',  # Fundo quase sólido
            bordercolor='black',  # Borda preta
            borderwidth=1,        # Espessura da borda
            font=dict(
                size=12,
                color='black'     # Texto preto
            )
        ),
        xaxis=dict(
            range=[-30, 30],
            showgrid=True,
            zeroline=True,
            title="Distância Lateral (pés)",
            titlefont=dict(size=14, color='black'),
            tickfont=dict(size=12, color='black'),
            autorange=True  # Força autorange aqui
        ),
        yaxis=dict(
            range=[-5, 45],
            showgrid=True,
            zeroline=True,
            title="Distância da Cesta (pés)",
            titlefont=dict(size=14, color='black'),
            tickfont=dict(size=12, color='black'),
            autorange=True  # Força autorange aqui
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(color='black'),
    )
    
    return fig

def main():
    st.title("🏀 Monitor de Predições - Kobe Shots")
    
    # Carrega dados
    train_data, prod_predictions = load_data()
    metrics = get_mlflow_metrics()
    
    # Layout em três colunas
    col1, col2, col3 = st.columns(3)
    
    # Métricas principais com formatação melhorada
    with col1:
        st.metric(
            "Log Loss", 
            f"{metrics.get('test_log_loss', 0):.4f}",
            help="Log Loss do modelo em dados de teste"
        )
    with col2:
        st.metric(
            "F1-Score", 
            f"{metrics.get('test_f1_score', 0):.4f}",
            help="F1-Score do modelo em dados de teste"
        )
    with col3:
        st.metric(
            "Amostras Válidas", 
            int(len(prod_predictions)),
            help="Número total de predições em produção"
        )
    
    # Tabs para diferentes visualizações
    tab1, tab2, tab3 = st.tabs(["📈 Métricas", "🔄 Data Drift", "🎯 Predições"])
    
    # Tab de Métricas
    with tab1:
        st.subheader("Evolução das Métricas")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_logloss = plot_metric_history('log_loss')
            if fig_logloss:
                st.plotly_chart(fig_logloss, use_container_width=True)
        
        with col2:
            fig_f1 = plot_metric_history('f1_score')
            if fig_f1:
                st.plotly_chart(fig_f1, use_container_width=True)
    
    # Tab de Data Drift
    with tab2:
        st.subheader("Análise de Data Drift")
        feature = st.selectbox(
            "Selecione a feature para análise:",
            ['shot_distance', 'lat', 'lon', 'minutes_remaining', 'period', 'playoffs']
        )
        
        fig_dist = plot_feature_distribution(train_data, prod_predictions, feature)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Estatísticas básicas
        col1, col2 = st.columns(2)
        with col1:
            st.write("Estatísticas de Treino")
            st.dataframe(train_data[feature].describe())
        with col2:
            st.write("Estatísticas de Produção")
            st.dataframe(prod_predictions[feature].describe())
    
    # Tab de Predições
    with tab3:
        st.subheader("Últimas Predições")
        
        # Visualização da quadra
        st.subheader("Mapa de Arremessos")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Arremessos de Treino")
            court_train = plot_shots_simple(train_data, "Arremessos de Treino")
            st.plotly_chart(court_train, use_container_width=True)
        
        with col2:
            st.write("Arremessos em Produção")
            court_prod = plot_shots_simple(prod_predictions, "Arremessos em Produção")
            st.plotly_chart(court_prod, use_container_width=True)
        
        # Amostra das últimas predições
        st.subheader("Últimas Predições")
        st.dataframe(
            prod_predictions.tail(10)[['shot_distance', 'prediction_label', 'prediction_score']]
        )
        
        # Distribuição das probabilidades preditas
        fig_pred = px.histogram(
            prod_predictions,
            x='prediction_score',
            title="Distribuição das Probabilidades Preditas"
        )
        st.plotly_chart(fig_pred, use_container_width=True)

if __name__ == "__main__":
    main() 