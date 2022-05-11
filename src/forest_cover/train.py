import click
from joblib import dump
import mlflow
import pandas as pd
from pathlib import Path

from forest_cover.constants import (
    DATA_DIR,
    DATA_PATH,
    MODEL_PATH,
    MODELS,
    OUTPUT_DIR,
    WEIGTHS,
)
from .utils import cv_validate_metrics, get_cat_num, get_dataset, write_to_mlflow
from .pipeline import create_pipeline


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
@click.option("--estimators", default=100, type=int, show_default=True)
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
    estimators: int,
    max_depth: int,
    max_features: float,
    min_samples_leaf: int,
    neighbors: int,
    weights: str,
) -> None:
    params = {'model': model,
              'use_scaler': use_scaler,
              'n_estimators': estimators,
              'max_depth': max_depth,
              'max_features': max_features,
              'min_samples_leaf': min_samples_leaf,
              'n_neighbors': neighbors,
              'weights': weights}
    features_train, target_train = get_dataset(input_path)
    categoricals, numericals = get_cat_num(features_train)

    pipeline = create_pipeline(
        model,
        use_scaler,
        estimators,
        max_depth,
        max_features,
        min_samples_leaf,
        random_state,
        neighbors,
        weights,
        categoricals,
        numericals,
    )

    with mlflow.start_run():
        write_to_mlflow(pipeline,
                        cv_validate_metrics,
                        features_train,
                        target_train,
                        params)
        dump(pipeline, output_path)
        click.echo(f"Model is saved to {output_path}.")
