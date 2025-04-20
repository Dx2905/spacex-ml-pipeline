import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
import mlflow
# mlflow.set_tracking_uri("file:///opt/airflow/project/mlruns")
import os

# Detect if running in Airflow (Docker) or locally
if os.environ.get("AIRFLOW_CTX_DAG_ID"):
    mlflow.set_tracking_uri("file:///opt/airflow/project/mlruns")  # for Airflow/Docker
else:
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))  # for local runs

import mlflow.sklearn
import joblib

# Load data
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv")
X = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv")
Y = pd.Series(data['Class'].to_numpy())

# Standardize X
transform = preprocessing.StandardScaler()
X = transform.fit(X).transform(X)

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train multiple models
models = {
    "LogisticRegression": (LogisticRegression(), {"C": [0.01, 0.1, 1], "penalty": ["l2"], "solver": ["lbfgs"]}),
    "SVM": (SVC(), {"kernel": ["linear", "rbf", "poly", "sigmoid"], "C": np.logspace(-3, 3, 5), "gamma": np.logspace(-3, 3, 5)}),
    "DecisionTree": (DecisionTreeClassifier(), {
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_depth": [2*n for n in range(1,10)],
        "max_features": ["sqrt"],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 5, 10]
    }),
    "KNN": (KNeighborsClassifier(), {
        "n_neighbors": list(range(1, 11)),
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "p": [1, 2]
    }),
}

results = {}
mlflow.set_experiment("SpaceX Landing Prediction")

for name, (model, params) in models.items():
    print(f"Training {name}...")
    with mlflow.start_run(run_name=name):
        clf = GridSearchCV(model, params, cv=10)
        clf.fit(X_train, Y_train)
        yhat = clf.predict(X_test)

        acc = accuracy_score(Y_test, yhat)
        f1 = f1_score(Y_test, yhat)
        jac = jaccard_score(Y_test, yhat)

        mlflow.log_param("model_name", name)
        mlflow.log_params(clf.best_params_)
        mlflow.log_metrics({"accuracy": acc, "f1_score": f1, "jaccard_score": jac})
        # mlflow.sklearn.log_model(clf.best_estimator_, "model")
        # mlflow.sklearn.log_model(clf.best_estimator_, artifact_path="model")
        mlflow.sklearn.log_model(
            clf.best_estimator_,
            artifact_path="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE
        )


        results[name] = {
            "best_params": clf.best_params_,
            "test_accuracy": acc,
            "f1_score": f1,
            "jaccard_score": jac,
            "model": clf.best_estimator_
        }
        print(f"→ {name} logged to MLflow.")

# Save best model
best_model_name = max(results, key=lambda k: results[k]["test_accuracy"])
joblib.dump(results[best_model_name]["model"], "model.pkl")
print(f"Best model saved: {best_model_name}")