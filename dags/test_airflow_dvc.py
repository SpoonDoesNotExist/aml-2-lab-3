"""ЛР 3. Версионирование Titanic и модели через DVC и MinIO."""
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

REPO = Path('/opt/airflow/repo')
FEATURES = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']

# Значения читаются из Admin -> Variables при выполнении задачи.
# append_env сохраняет остальное окружение; PATH передаётся явно.
DVC_ENV = {
    'AWS_ACCESS_KEY_ID': '{{ var.value.minio_access_key }}',
    'AWS_SECRET_ACCESS_KEY': '{{ var.value.minio_secret_key }}',
    'AWS_ENDPOINT_URL': '{{ var.value.minio_endpoint }}',
    'AWS_DEFAULT_REGION': 'us-east-1',
    'PATH': os.environ['PATH'],
}


def create_dataset():
    import pandas as pd
    from catboost.datasets import titanic

    try:
        train, test = titanic()
        print('Dataset source: catboost.datasets.titanic')
    except (RuntimeError, OSError) as error:
        print(f'CatBoost download unavailable: {error}; using local Titanic snapshot')
        train = pd.read_csv(REPO / 'raw' / 'train.csv')
        test = pd.read_csv(REPO / 'raw' / 'test.csv')
    mean_age = train['Age'].mean()
    for frame in (train, test):
        frame['Sex'] = frame['Sex'].map({'male': 0, 'female': 1})
        frame['Age'] = frame['Age'].fillna(mean_age)
    train[FEATURES + ['Survived']].to_csv(REPO / 'data_train.csv', index=False)
    test[FEATURES].to_csv(REPO / 'data_test.csv', index=False)
    print(f'Saved data_train.csv: {len(train)} rows; data_test.csv: {len(test)} rows')


def make_prediction():
    import pandas as pd

    with (REPO / 'model.pkl').open('rb') as stream:
        model = pickle.load(stream)
    test = pd.read_csv(REPO / 'data_test.csv')
    predictions = model.predict(test[FEATURES].values)
    print(f'Predictions: {len(predictions)}; first passenger: {int(predictions[0])}')
    print(f'First 10 predictions: {predictions[:10].tolist()}')


with DAG(
    dag_id='test_airflow_dvc',
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args={'owner': 'student', 'retries': 1, 'retry_delay': timedelta(seconds=15)},
    tags=['lab-3', 'dvc', 'titanic'],
) as dag:
    dataset = PythonOperator(task_id='create_dataset', python_callable=create_dataset)
    version = BashOperator(
        task_id='version_data', cwd=str(REPO), env=DVC_ENV, append_env=True,
        bash_command='set -euo pipefail; dvc add data_train.csv data_test.csv; dvc push',
    )
    train = BashOperator(
        task_id='train_model', cwd=str(REPO), env=DVC_ENV, append_env=True,
        bash_command='set -euo pipefail; dvc pull; python train.py; dvc add model.pkl; dvc push',
    )
    prediction = PythonOperator(task_id='make_prediction', python_callable=make_prediction)
    dataset >> version >> train >> prediction
