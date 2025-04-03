.PHONY: all mlflow pipeline application streamlit clean

all: mlflow pipeline application streamlit

mlflow:
	@echo "Iniciando servidor MLflow..."
	@mlflow ui --backend-store-uri file:./mlruns --host 0.0.0.0 --port 5000 &
	@sleep 5  # Aguarda o servidor iniciar


pipeline:
	@echo "Executando pipeline..."
	@python code/pipeline.py


application:
	@echo "Executando aplicação..."
	@python code/aplicacao.py


streamlit:
	@echo "Iniciando dashboard Streamlit..."
	@streamlit run code/dashboard/app.py


clean:
	@echo "Limpando arquivos temporários..."
	@rm -f logs.log
	@rm -rf mlartifacts/*
	@rm -rf mlruns/.trash/* 