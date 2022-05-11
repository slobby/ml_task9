from typing import Union
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler

from forest_cover.constants import MODELS


def create_pipeline(
    model: str,
    use_scaler: bool = True,
    n_estimators: int = 100,
    max_depth: Union[int, None] = None,
    max_features: Union[float, str] = "auto",
    min_samples_leaf: int = 1,
    random_state: Union[int, None] = None,
    neighbors: int = 5,
    weights: str = "uniform",
    categoricals: list[str] = list(),
    numericals: list[str] = list()
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        transformer = ColumnTransformer(
            transformers=[("rs", RobustScaler(), numericals)], remainder="passthrough"
        )
        pipeline_steps.append(("scaller", transformer))

    if model == MODELS[0]:
        clr = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
    elif model == MODELS[1]:
        clr = KNeighborsClassifier(n_neighbors=neighbors, weights=weights)

    pipeline_steps.append(("classifier", clr))

    return Pipeline(steps=pipeline_steps)
