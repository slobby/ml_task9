import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from typing import Any, Callable, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from .config import space_knn, space_random_forest
from .constants import MODELS, TARGET
from .pipeline import create_pipeline


def get_dataset(input_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    dataset = pd.read_csv(input_path)
    return (dataset.drop(TARGET, axis=1), dataset[TARGET])


def get_cat_num(features_train: pd.DataFrame) -> tuple[list, list]:
    categoricals = []
    numericals = []
    for col in features_train.columns:
        if col.startswith("Soil_Type") or col.startswith("Wilderness_Area"):
            categoricals.append(col)
        else:
            numericals.append(col)
    return (categoricals, numericals)


def get_model_space(model: str, random_state: int) -> tuple[Union[RandomForestClassifier, KNeighborsClassifier], dict[str, list]]:
    space = dict()
    if model == MODELS[0]:
        clr = RandomForestClassifier(random_state=random_state)
        space = space_random_forest
    if model == MODELS[1]:
        clr = KNeighborsClassifier()
        space = space_knn
    return (clr, space)


def cv_validate_metrics(
    pipe: Pipeline, features_train: pd.DataFrame, target_train: pd.Series
) -> tuple[float, float, float]:
    scoring = ["accuracy", "f1_weighted", "roc_auc_ovr_weighted"]
    scores = cross_validate(pipe, features_train, target_train, scoring=scoring)
    return (
        scores["test_accuracy"].mean(),
        scores["test_f1_weighted"].mean(),
        scores["test_roc_auc_ovr_weighted"].mean(),
    )


def validate_metrics(
    estimator: Any, features_train: pd.DataFrame, target_train: pd.Series
) -> tuple[float, float, float]:
    yhat = estimator.predict(features_train)
    y_pred = estimator.predict_proba(features_train)
    return (
        accuracy_score(target_train, yhat),
        f1_score(target_train, yhat, average='weighted'),
        roc_auc_score(target_train, y_pred, average='weighted', multi_class='ovr')
    )


def write_to_mlflow(estimator: Any,
                    validator: Callable,
                    features_train: pd.DataFrame,
                    target_train: pd.Series,
                    params: dict) -> None:

    accuracy, f1, roc_auc = validator(
        estimator, features_train, target_train
    )

    model_t = (
        "RandomForestClassifier" if params['model'] == MODELS[0] else "KNeighborsClassifier"
    )
    mlflow.sklearn.log_model(estimator, "models")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_param("use_scaler", params['use_scaler'])
    mlflow.log_param("model", model_t)

    if params['model'] == MODELS[0]:
        mlflow.log_param('n_estimators', params['n_estimators'])
        mlflow.log_param("max_depth", params['max_depth'])
        mlflow.log_param("max_features", params['max_features'])
        mlflow.log_param("min_samples_leaf", params['min_samples_leaf'])
    if params['model'] == MODELS[1]:
        mlflow.log_param("neighbors", params['n_neighbors'])
        mlflow.log_param("weights", params['weights'])
