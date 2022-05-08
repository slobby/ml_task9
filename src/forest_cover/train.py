from pathlib import Path
import pandas as pd
from joblib import dump
from sklearn.model_selection import cross_validate
import click
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn

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
@click.option("--use-scaler/--no-scaler", default=False, type=bool, show_default=True)
@click.option("--max-depth", default=100, type=int, show_default=True)
@click.option(
    "--max-features",
    default=1,
    type=click.FloatRange(0, 1, min_open=True, max_open=False),
    show_default=True,
)
@click.option("--min-samples-leaf", default=1, type=int, show_default=True)
@click.option("--neighbors", default=5, type=int, show_default=True)
@click.option(
    "--weights",
    default=WEIGTHS[0],
    type=click.Choice(WEIGTHS, case_sensitive=False),
    show_default=True,
)
def train(
    input_path: Path,
    output_path: Path,
    random_state: int,
    model: str,
    use_scaler: bool,
    max_depth: int,
    max_features: float,
    min_samples_leaf: int,
    neighbors: int,
    weights: str,
) -> None:
    features_train, target_train = get_dataset(input_path)
    categoricals = []
    numericals = []
    for col in features_train.columns:
        if col.startswith("Soil_Type") or col.startswith("Wilderness_Area"):
            categoricals.append(col)
        else:
            numericals.append(col)

    with mlflow.start_run():
        pipeline = create_pipeline(
            model,
            use_scaler,
            max_depth,
            max_features,
            min_samples_leaf,
            random_state,
            neighbors,
            weights,
            categoricals,
            numericals,
        )
        accuracy, f1, roc_auc = cv_validate_metrics(
            pipeline, features_train, target_train
        )
        model_t = (
            "DecisionTreeClassifier" if model == MODELS[0] else "KNeighborsClassifier"
        )
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("model", model_t)
        if model == MODELS[0]:
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("max_features", max_features)
            mlflow.log_param("min_samples_leaf", min_samples_leaf)
        if model == MODELS[1]:
            mlflow.log_param("neighbors", neighbors)
            mlflow.log_param("weights", weights)
        mlflow.sklearn.log_model(pipeline, "models")
        click.echo(f"Accuracy: {accuracy}.\nF1: {f1}.\nROC_AUC: {roc_auc}.")
        dump(pipeline, output_path)
        click.echo(f"Model is saved to {output_path}.")
