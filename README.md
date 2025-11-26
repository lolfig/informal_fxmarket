informal_fxmarket
==================

Project Overview
----------------
This repository collects, processes, and analyses data about informal foreign exchange (FX) market ("blue dollar", P2P USDT/ARS, USDCUP, NIG, etc.).
It includes scrapers to collect intraday data, scripts to prepare time series, and analysis scripts that run Hidden Markov Models, random-walk tests, and other forms of empirical decomposition.

Key features
------------
- Scrapers that collect P2P prices and other FX-related series.
- Scripts to prepare time series from raw data and produce pickled dataframes for analysis.
- Analysis code using tools such as HMM (`hmmlearn`), SciPy, NumPy, statsmodels, and plotting utilities.

Requirements
------------
- Python 3.10+ (3.11 recommended; some packages may depend on specific versions).
- Use the included `requirements.txt` to install dependencies:

  PowerShell examples
  -------------------
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt

Project layout
--------------
- `config/` — configuration constants and paths used across scripts.
- `data/`
  - `data/raw/` — raw data stored here (IDEA: track these files if they are source-of-truth).
  - `data/processed/` — derived/processed data (typically ignored in git; used for intermediate pickles and CSVs).
- `src/` — main scripts for scraping, preparing time series, and analyses. Key scripts:
    - `scraping_blue.py` — collects USDT–ARS data from Binance P2P and writes `usdt_ars_intraday.csv`.
    - `scraping_usd_eur.py` — downloads daily FX series with `yfinance` (note: currently writes to an absolute path; update as needed).
    - `preparing_time_series.py` — consolidates and interpolates raw series and writes pickles to `data/processed/`.
    - `hidden_markov_model.py` — example HMM-based analysis script; uses `DIR_PROCESSED_DATA` to read pickles.
    - `emp_decom.py`, `intraday_looking.py`, `random_walk_test.py` — additional analysis scripts.
- `models/` — optional model outputs (ignored by default, recommended to use Git LFS for big artifacts).
- `Plots/` — generated plot images (ignored by default).

Running scripts
---------------
Run scripts directly with Python. Example:

  PowerShell
  ----------------
  cd 'c:\Users\agarc\Projects\informal_fxmarket'
  .\.venv\Scripts\Activate.ps1
  python src/scraping_blue.py
  python src/preparing_time_series.py
  python src/hidden_markov_model.py

Important notes
---------------
- `scraping_usd_eur.py` currently uses an absolute path in the script to save the series. Update it to use `config.const` constants or a relative path if you want it to save under this repository's `data/` directories.
- `data/processed/` is ignored by default (see `.gitignore`). If you want to track processed data in git, remove the `data/processed/` line from `.gitignore` or move processed data to a different tracked location.

Removing already-tracked files from Git
--------------------------------------
If you previously committed files that are now in `.gitignore` and you want to untrack them without deleting them locally, run something like:

  PowerShell
  ----------------
  cd 'c:\Users\agarc\Projects\informal_fxmarket'
  git rm -r --cached data/processed/
  git rm -r --cached results/
  git rm -r --cached models/
  git commit -m "Untrack data, results and model artifacts"

This will remove them from the repository while keeping them on your filesystem.

Optional/Advanced
-----------------
- For large model artifacts, consider using Git LFS.
- Consider adding `pre-commit` hooks for linters/formatters (black, isort, flake8) and unit tests.
- For reliable development, use `pip install -r requirements.txt` or manage environments via `conda`.

Contributing
------------
- Feel free to open issues or pull requests.
- If adding scripts or datasets, try to keep raw data as source-of-truth in `data/raw/` and avoid checking generated outputs into the repository.

