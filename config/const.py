import os
from pathlib import Path

import datetime
from typing import Literal

today = datetime.datetime.now().date()
yesterday = today - datetime.timedelta(days=1)

# CONST
PROJECT_ROOT = Path(__file__).parent.parent
DIR_DATA = PROJECT_ROOT / "data"
DIR_RAW_DATA = DIR_DATA / "raw"
DIR_PROCESSED_DATA = DIR_DATA / "processed"
DIR_RESULTS_DATA = DIR_DATA / "results"
DIR_FIGURES = PROJECT_ROOT / "figures"
DIR_RESULTS = PROJECT_ROOT / "results"