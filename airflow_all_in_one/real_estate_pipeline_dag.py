import sys
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging

# Konfiguracja loggingu
logging.basicConfig(level=logging.INFO)

# Importowanie funkcji z plików w tym samym folderze
from urls import main as run_urls
from scraper import main as run_scraper
from preprocessing import main as run_preprocessing
from random_forest_model import main as run_random_forest

# Domyślne argumenty DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 12, 22),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Tworzenie instancji DAG
with DAG(
    'real_estate_pipeline',
    default_args=default_args,
    description='Pipelines for scraping, preprocessing, and training the real estate model',
    schedule_interval='@daily',  # Zmiana na schedule_interval
) as dag:

    # Zadanie 1: Uruchomienie skryptu do generowania URL-i
    task_urls = PythonOperator(
        task_id='run_urls',
        python_callable=run_urls,
    )

    # Zadanie 2: Uruchomienie skryptu do scrapowania danych
    task_scraper = PythonOperator(
        task_id='run_scraper',
        python_callable=run_scraper,
    )

    # Zadanie 3: Uruchomienie skryptu do przetwarzania danych
    task_preprocessing = PythonOperator(
        task_id='run_preprocessing',
        python_callable=run_preprocessing,
    )

    # Zadanie 4: Uruchomienie skryptu do trenowania modelu
    task_random_forest = PythonOperator(
        task_id='run_random_forest',
        python_callable=run_random_forest,
    )

    # Określanie zależności między zadaniami (kolejność wykonywania)
    task_urls >> task_scraper >> task_preprocessing >> task_random_forest
