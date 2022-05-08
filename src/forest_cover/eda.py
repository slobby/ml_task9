from pathlib import Path
from typing import Union
import click
import pandas as pd
from pandas_profiling import ProfileReport  # type: ignore

from forest_cover.constants import DATA_DIR, DATA_PATH
from forest_cover.constants import EDA_PATH, EDA_TITLE, OUTPUT_DIR


@click.command()
@click.option(
    "-i",
    "--input",
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
def eda(input: Union[Path, None] = None, output: Union[Path, None] = None) -> None:
    if input is None:
        input = Path(DATA_DIR).joinpath(DATA_PATH)
    if input.exists():
        output = output or Path(OUTPUT_DIR).joinpath(EDA_PATH)
        df = pd.read_csv(input)
        profile = ProfileReport(df, title=EDA_TITLE, explorative=True)
        profile.to_file(output)
    else:
        raise ValueError()
