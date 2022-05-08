from typing import Union
from pathlib import Path
import pandas as pd
from joblib import dump
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import click

from forest_cover.constants import DATA_DIR, DATA_PATH, MODEL_PATH, OUTPUT_DIR, TARGET


def get_dataset(input_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    dataset = pd.read_csv(input_path)
    return (dataset.drop(TARGET, axis=1), dataset[TARGET])


def create_pipeline(
        use_scaler: bool,
        max_depth: Union[int, None],
        max_features: Union[int, str, float],
        min_samples_leaf: Union[int, float],
        random_state: int) -> Pipeline:

    pipeline_steps = []
    pipeline_steps.append(
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=max_depth,
                max_features=max_features,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state)
        )
    )

    return Pipeline(steps=pipeline_steps)


def cv_validate_metrics(pipe: Pipeline, features_train: pd.DataFrame, target_train: pd.Series) -> tuple[float, float, float]:
    scoring = ['accuracy', 'f1_weighted', 'roc_auc_ovr_weighted']
    scores = cross_validate(pipe, features_train,
                            target_train, scoring=scoring)
    return scores['test_accuracy'].mean(), scores['test_f1_weighted'].mean(), scores['test_roc_auc_ovr_weighted'].mean()


@click.command()
@click.option(
    "-i",
    "--input-path",
    default=Path(DATA_DIR).joinpath(DATA_PATH),
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True
)
@click.option(
    "-o",
    "--output-path",
    default=Path(OUTPUT_DIR).joinpath(MODEL_PATH),
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True
)
@click.option(
    "--max-depth",
    default=100,
    type=int,
    show_default=True
)
@click.option(
    "--max-features",
    default=1,
    type=click.FloatRange(
        0, 1, min_open=True, max_open=False),
    show_default=True
)
@click.option(
    "--min-samples-leaf",
    default=1,
    type=int,
    show_default=True
)
def train(
    input_path: Path,
    output_path: Path,
    random_state: int,
    use_scaler: bool,
    max_depth: int,
    max_features: float,
    min_samples_leaf: int
) -> None:
    features_train, target_train = get_dataset(input_path)

    pipeline = create_pipeline(
        use_scaler, max_depth, max_features, min_samples_leaf, random_state)
    pipeline.fit(features_train, target_train)
    accuracy, f1, roc_auc = cv_validate_metrics(
        pipeline, features_train, target_train)
    click.echo(f"Accuracy: {accuracy}.\nF1: {f1}.\nROC_AUC: {roc_auc}.")
    dump(pipeline, output_path)
    click.echo(f"Model is saved to {output_path}.")
