from pathlib import Path
from typing import Any, Union
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, cross_validate
import click
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import RobustScaler

from forest_cover.constants import (
    DATA_DIR,
    DATA_PATH,
    MODEL_PATH,
    MODELS,
    OUTPUT_DIR,
    TARGET,
    WEIGTHS,
)
from .pipeline import create_pipeline


def get_dataset(input_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    dataset = pd.read_csv(input_path)
    return (dataset.drop(TARGET, axis=1), dataset[TARGET])


def get_model_space(model: str, random_state: int, categoricals: list[str], numericals: list[str]) -> tuple[Union[RandomForestClassifier, KNeighborsClassifier], dict[str, list]]:
    space = dict()
    if model == MODELS[0]:
        clr = RandomForestClassifier(random_state=random_state)
        space['n_estimators'] = [2, 5]  # [10, 50, 100]
        space['max_depth'] = [2, 10]  # [2, 10, 20, 50]
        space['max_features'] = [0.2, 0.5]  # [0.2, 0.5, 0.7]
        space['min_samples_leaf'] = [1, 5]  # [1, 2, 5, 10]
    if model == MODELS[1]:
        clr = KNeighborsClassifier()
        space['n_neighbors'] = [2, 10]  # [2, 5, 10, 15]
        space['weights'] = WEIGTHS
    return (clr, space)


def cv_validate_metrics(
    estimator: Any, features_train: pd.DataFrame, target_train: pd.Series
) -> tuple[float, float, float]:
    scoring = ["accuracy", "f1_weighted", "roc_auc_ovr_weighted"]
    scores = cross_validate(estimator, features_train, target_train, scoring=scoring)
    return (
        scores["test_accuracy"].mean(),
        scores["test_f1_weighted"].mean(),
        scores["test_roc_auc_ovr_weighted"].mean(),
    )


@click.command()
@click.option(
    "-i",
    "--input-path",
    default=Path(DATA_DIR).joinpath(DATA_PATH),
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-o",
    "--output-path",
    default=Path(OUTPUT_DIR).joinpath(MODEL_PATH),
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option("--random-state", default=42, type=int, show_default=True)
@click.option(
    "--model",
    default=MODELS[0],
    type=click.Choice(MODELS, case_sensitive=False),
    show_default=True,
)
def nested(
    input_path: Path,
    output_path: Path,
    random_state: int,
    model: str
) -> None:
    features_train, target_train = get_dataset(input_path)
    categoricals = []
    numericals = []
    for col in features_train.columns:
        if col.startswith("Soil_Type") or col.startswith("Wilderness_Area"):
            categoricals.append(col)
        else:
            numericals.append(col)
    pipeline = Pipeline(steps=[("scaller", ColumnTransformer(transformers=[("rs", RobustScaler(), numericals)], remainder="passthrough"))])
    features_train = pipeline.fit_transform(features_train)
    with mlflow.start_run():
        cv_outer = KFold(n_splits=10, shuffle=True, random_state=random_state)
        # features_train = features_train.to_numpy()
        target_train = target_train.to_numpy()
        for train_ix, test_ix in cv_outer.split(features_train):
            X_train, X_test = features_train[train_ix, :], features_train[test_ix, :]
            y_train, y_test = target_train[train_ix], target_train[test_ix]
            cv_inner = KFold(n_splits=3, shuffle=True, random_state=random_state)
            clr, space = get_model_space(model, random_state, categoricals, numericals)
            search = GridSearchCV(clr, space, scoring='accuracy', cv=cv_inner, refit=True)
            result = search.fit(X_train, y_train)
            best_model = result.best_estimator_
            accuracy, f1, roc_auc = cv_validate_metrics(
                best_model, X_test, y_test)
            print(result.best_params_)

        # model_t = (
        #     "RandomForestClassifier" if model == MODELS[0] else "KNeighborsClassifier"
        # )
        # mlflow.log_metric("accuracy", accuracy)
        # mlflow.log_metric("f1", f1)
        # mlflow.log_metric("roc_auc", roc_auc)
        # mlflow.log_param("use_scaler", True)
        # mlflow.log_param("model", model_t)
        # if model == MODELS[0]:
        #     mlflow.log_param("max_depth", max_depth)
        #     mlflow.log_param("max_features", max_features)
        #     mlflow.log_param("min_samples_leaf", min_samples_leaf)
        # if model == MODELS[1]:
        #     mlflow.log_param("neighbors", neighbors)
        #     mlflow.log_param("weights", weights)
        # mlflow.sklearn.log_model(pipeline, "models")
            click.echo(f"Accuracy: {accuracy}.\nF1: {f1}.\nROC_AUC: {roc_auc}.")
        # dump(pipeline, output_path)
        # click.echo(f"Model is saved to {output_path}.")
