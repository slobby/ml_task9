from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler

from forest_cover.constants import MODELS


def create_pipeline(
    model: str,
    use_scaler: bool,
    max_depth: int,
    max_features: float,
    min_samples_leaf: int,
    random_state: int,
    neighbors: int,
    weights: str,
    categoricals: list[str],
    numericals: list[str],
) -> Pipeline:

    pipeline_steps = []
    if use_scaler:
        transformer = ColumnTransformer(
            transformers=[("rs", RobustScaler(), numericals)], remainder="passthrough"
        )
        pipeline_steps.append(("scaller", transformer))

    if model == MODELS[0]:
        clr = DecisionTreeClassifier(
            max_depth=max_depth,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
    elif model == MODELS[1]:
        clr = KNeighborsClassifier(n_neighbors=neighbors, weights=weights)

    pipeline_steps.append(("classifier", clr))

    return Pipeline(steps=pipeline_steps)
