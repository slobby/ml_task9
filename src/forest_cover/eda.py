from pathlib import Path, PurePath
import click
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport

from forest_cover.constants import DATA_DIR, DATA_PATH, EDA_PATH, EDA_TITLE, OUTPUT_DIR
from forest_cover.exceptions import WrongFileException


@click.command()
@click.option(
    '-i',
    '--input',
    default=Path(DATA_DIR).joinpath(DATA_PATH),
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-o",
    "--output",
    default=Path(OUTPUT_DIR).joinpath(EDA_PATH),
    type=click.Path(dir_okay=False, path_type=Path),
    show_default=True,
)
def eda(input: Path = None, output: Path = None) -> None:
    if input is None:
        input = Path(DATA_DIR).joinpath(DATA_PATH)
    if input.exists():
        output = output or Path(OUTPUT_DIR).joinpath(EDA_PATH)
        df = pd.read_csv(input)
        profile = ProfileReport(df, title=EDA_TITLE, explorative=True)
        profile.to_file(output)
    else:
        raise WrongFileException(f'File [path] doesn`t exist')
