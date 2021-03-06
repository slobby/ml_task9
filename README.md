## Task9.

#### What wasn`t done:

Task11\
Task14\
Task15

## Usage

1. Clone this repository to your machine.
2. Download [Forest train dataset](https://www.kaggle.com/competitions/forest-cover-type-prediction) dataset, save csv locally (default path is _data/train.csv_ in repository's root).
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.1.11).
4. Install the project dependencies (_run this and following commands in a terminal, from the root of a cloned repository_):

```
poetry install --no-dev
```

5. Run train with the following command:

```
poetry run train [OPTIONS]
```

You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:

```
poetry run train --help
```

```
Usage: train [OPTIONS]

Options:
  -i, --input-path FILE         [default: data\train.csv]
  -o, --output-path FILE        [default: output\model.joblib]
  --random-state INTEGER        [default: 42]
  --model [tree|knn]            [default: tree]
  --use-scaler / --no-scaler    [default: no-scaler]
  --estimators INTEGER          [default: 100]
  --max-depth INTEGER           [default: 100]
  --max-features FLOAT RANGE    [default: 1; 0<x<=1]
  --min-samples-leaf INTEGER    [default: 1]
  --neighbors INTEGER           [default: 5]
  --weights [uniform|distance]  [default: uniform]
  --help                        Show this message and exit.
```

6. Run MLflow UI to see the information about experiments you conducted:

```
poetry run mlflow ui
```

7. For creating EDA run the following command:

```
poetry run eda
```

then open in dir /data/report.html

8. Run nestedCD with the following command:

```
poetry run nested [OPTIONS]
```

You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:

```
poetry run nested --help
```

```
Usage: nested [OPTIONS]

Options:
  -i, --input-path FILE   [default: data\train.csv]
  -o, --output-path FILE  [default: output\model.joblib]
  --random-state INTEGER  [default: 42]
  --model [tree|knn]      [default: tree]
  --help                  Show this message and exit.
```

## Development

The code in this repository must be tested, formatted with black, and pass mypy typechecking before being commited to the repository.

Install all requirements (including dev requirements) to poetry environment:

```
poetry install
```

Screenshort from MLFlow

![MLFlow](https://github.com/slobby/ml_task9/blob/master/pic/ml.png)

Screenshort from MLFlow

![MLFlow](https://github.com/slobby/ml_task9/blob/master/pic/ml_2.png)

Screenshort from flake and black

To run black run the following command:

```
poetry run black .
```

To run flake8 run the following command:

```
poetry run flake8
```

![flake and black](https://github.com/slobby/ml_task9/blob/master/pic/flake.png)

To run mypy run the following command:

```
poetry run mypy .
```

Screenshort from mypy

![mypy](https://github.com/slobby/ml_task9/blob/master/pic/mypy.png)
