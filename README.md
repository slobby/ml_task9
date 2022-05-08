## Homework for RS School Machine Learning course. Task9.

#### What wasn`t done:
Task9 \
Task11\
Task14\
task15


## Usage
1. Clone this repository to your machine.
2. Download [Forest train dataset](https://www.kaggle.com/competitions/forest-cover-type-prediction) dataset, save csv locally (default path is *data/train.csv* in repository's root).
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.1.11).
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
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
6. Run MLflow UI to see the information about experiments you conducted:
```
poetry run mlflow ui
```

7. For creating EDA run the following command:
```
poetry run eda
```

then open in dir /data/report.html 

## Development

The code in this repository must be tested, formatted with black, and pass mypy typechecking before being commited to the repository.

Install all requirements (including dev requirements) to poetry environment:
```
poetry install
```

Screenshort from MLFlow
![MLFlow](.\pic\ml.png)

Screenshort from flake and black
![flake and black](.\pic\flake.png)

Screenshort from mypy
![mypy](.\pic\mypy.png)


