import mlflow
from pathlib import Path
from joblib import dump
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import GridSearchCV, KFold
import click
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import RobustScaler

from .constants import DATA_DIR, DATA_PATH, MODEL_PATH, MODELS, OUTPUT_DIR
from .utils import (
    get_cat_num,
    get_dataset,
    get_model_space,
    validate_metrics,
    write_to_mlflow,
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
def nested(input_path: Path, output_path: Path, random_state: int, model: str) -> None:
    features_train, target_train = get_dataset(input_path)
    categoricals, numericals = get_cat_num(features_train)

    pipeline = Pipeline(
        steps=[
            (
                "scaller",
                ColumnTransformer(
                    transformers=[("rs", RobustScaler(), numericals)],
                    remainder="passthrough",
                ),
            )
        ]
    )
    features_train = pipeline.fit_transform(features_train)

    cv_outer = KFold(n_splits=10, shuffle=True, random_state=random_state)
    target_train = target_train.to_numpy()

    for i, ix in enumerate(cv_outer.split(features_train), start=1):
        train_ix, test_ix = ix
        with mlflow.start_run():
            X_train, X_test = features_train[train_ix, :], features_train[test_ix, :]
            y_train, y_test = target_train[train_ix], target_train[test_ix]
            cv_inner = KFold(n_splits=3, shuffle=True, random_state=random_state)
            clr, space = get_model_space(model, random_state)
            search = GridSearchCV(
                clr, space, scoring="accuracy", cv=cv_inner, refit=True
            )
            result = search.fit(X_train, y_train)
            best_model = result.best_estimator_
            params = result.best_params_
            params["model"] = model
            params["use_scaler"] = True
            write_to_mlflow(best_model, validate_metrics, X_test, y_test, params)
            dump(pipeline, output_path)
            click.echo(f"Model is saved to {output_path}.")
