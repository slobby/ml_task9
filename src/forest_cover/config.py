from typing import Any
from forest_cover.constants import WEIGTHS

space_random_forest: dict[str, list[Any]] = dict()
space_random_forest["n_estimators"] = [10, 50]  # [10, 50, 100]
space_random_forest["max_depth"] = [2, 10, 20]  # [2, 10, 20, 50]
space_random_forest["max_features"] = [0.2, 0.7]  # [0.2, 0.5, 0.7, 1.0]
space_random_forest["min_samples_leaf"] = [1, 5]  # [1, 2, 5, 10]


space_knn: dict[str, list[Any]] = dict()
space_knn["n_neighbors"] = [1, 2, 3, 4, 20]
space_knn["weights"] = WEIGTHS
