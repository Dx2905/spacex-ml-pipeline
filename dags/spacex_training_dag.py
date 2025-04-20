from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'gaurav',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'spacex_ml_training_pipeline',
    default_args=default_args,
    description='Train and track SpaceX landing prediction model',
    schedule_interval='@weekly',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'spacex'],
) as dag:

    train_model = BashOperator(
        task_id='train_model',
        bash_command='python /opt/airflow/project/train.py'
    )
