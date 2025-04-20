FROM apache/airflow:2.7.1

USER airflow

RUN pip install --user --no-cache-dir \
    scikit-learn \
    pandas \
    numpy \
    mlflow \
    joblib
