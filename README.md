# Movie Recommendation System Using Machine Learning

A Flask web application that recommends movies from the MovieLens 100K
dataset using k-Nearest Neighbors collaborative filtering, in two modes:

- **User-based**: given a user ID from the dataset, find the most similar
  users (by cosine similarity of their rating patterns) and recommend
  movies those users rated highly that the target user hasn't seen.
- **Item-based**: given a movie title, find the most similar movies by
  comparing how users rated them.

Only movies with at least 100 ratings (configurable) are recommendable,
so recommendations are backed by enough data to be meaningful.

Built as the practical component of MCSP-232, IGNOU MCA. See
`PROJECT_SPEC.md` for the full build contract this codebase follows.

## Dataset

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)
(`ml-100k`) — 100,000 ratings from 943 users on 1,682 movies. The dataset
is downloaded by `scripts/fetch_data.py`; it is not committed to this
repository.

## Requirements

- Python 3.12
- git

## Getting started

Clone the repository, set up a virtual environment, install
dependencies, then fetch the data, build the database, and train the
model:

### Windows (PowerShell)

```powershell
git clone <repository-url> movie-recommendation-system
cd movie-recommendation-system
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python scripts\initialise.py
flask --app wsgi run
```

### Linux / macOS

```bash
git clone <repository-url> movie-recommendation-system
cd movie-recommendation-system
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python scripts/initialise.py
flask --app wsgi run
```

Then open http://127.0.0.1:5000 in a browser.

`scripts/initialise.py` runs the three setup steps in order —
`fetch_data.py` (download the dataset), `load_database.py` (load it into
SQLite), `train_model.py` (build and persist the KNN models) — and stops
with a clear error if any step fails. They can also be run individually
in that same order if you need to redo just one step.

## Project structure

```
app/
    api/            Flask blueprint: routes, input validation, error handling
    ml/             Framework-agnostic ML layer: data loading, preprocessing,
                     feature construction, the KNN engine, evaluation metrics
    models/         SQLAlchemy ORM models (users, movies, ratings)
    repositories/   Read-only data access, the only layer that queries the DB
    services/       Recommendation service, model lifecycle, chart generation
    static/         Hand-written CSS (see note below)
    templates/      Jinja templates
    utils/          Logging setup
    config.py       Environment-driven configuration, read once at startup
    extensions.py   The shared SQLAlchemy instance
    __init__.py     Application factory
scripts/
    fetch_data.py       Downloads and extracts the MovieLens 100K archive
    load_database.py    Loads the raw files into SQLite
    train_model.py      Builds the feature matrices and fits the KNN models
    evaluate.py         Runs app/ml/evaluation.py and prints the results
    initialise.py       Runs the three setup scripts above in sequence
data/       Raw and processed data (gitignored)
models/     Persisted model artefacts (gitignored)
reports/    Generated evaluation figures (gitignored)
wsgi.py     Entry point used by `flask run` / a production WSGI server
```

Dependency direction is strictly one-way:
`Presentation -> Application -> Service -> ML -> Data`. Routes hold no
business logic; services never import Flask; the ML layer takes plain
DataFrames and is testable without a running app.

## Running the evaluation

`scripts/evaluate.py` builds a reproducible held-out split of the
ratings, evaluates the user-based recommender at several neighbour
counts, and prints Precision@10, Recall@10, catalogue coverage, and the
sparsity of the user-movie matrix:

```bash
python scripts/evaluate.py
```

This requires the database to already be populated (run
`scripts/load_database.py`, or `scripts/initialise.py`, first if it
isn't). It fits its own models on the held-out train split and does not
need `scripts/train_model.py` to have been run.

## Running the tests

The automated test suite (`tests/`) uses only Python's standard-library
`unittest` — no pytest, no coverage tool, no extra dependency:

```bash
python -m unittest discover -s tests -v
```

The tests are read-only: they never write to the database or to
`models/`, and the one test that renders a chart does so into a
temporary directory, never `app/static/generated/`. Most of them do
need the database loaded and the model trained first
(`scripts/load_database.py` then `scripts/train_model.py`); a test
class that needs either and can't find it raises an error from
`setUpClass` naming both scripts, so a missing precondition reports
`FAILED`, never `OK` — it can't be mistaken for a pass. Results and
worked examples from a real run are recorded in `docs/test_cases.md`.

## Deployment

This project deploys to [Render](https://render.com) as a free-tier
Python web service, configured declaratively in `render.yaml` at the
project root (a Render "Blueprint").

- **Build**: `pip install -r requirements.txt && python scripts/load_database.py && python scripts/train_model.py`
  — installs dependencies, loads `data/raw/ml-100k/` into SQLite, then
  builds and persists the KNN models to `models/`.
- **Start**: `gunicorn wsgi:app --bind 0.0.0.0:$PORT` — serves the app
  with gunicorn, binding to the port Render assigns via the `PORT`
  environment variable. Nothing in the codebase hardcodes a port;
  local development is unaffected and continues to use
  `flask --app wsgi run` on Flask's default port.
- **Health check**: `/health`, so Render can detect the service is up.

The raw MovieLens dataset (`data/raw/ml-100k/`) is committed to this
repository rather than fetched during the build, because GroupLens's TLS
certificate is currently expired and Render's build environment cannot
download it. `scripts/fetch_data.py` still skips the download whenever
the extracted files are already present, so this only matters for an
environment that doesn't have them yet — the zip archive itself,
`data/processed/`, the SQLite database, and `models/*.joblib` remain
gitignored and are rebuilt by the build step.

`render.yaml` sets `APP_ENV=production` (so debug mode stays off) and
has Render generate a random `SECRET_KEY` at deploy time — the
application factory refuses to start under the production config
without one.

To deploy: push this repository to GitHub (or GitLab) and create a new
Blueprint in the Render dashboard pointing at it; Render reads
`render.yaml` and provisions the service automatically.

## Note on styling

All CSS in `app/static/style.css` is original and hand-written for this
project. The BootstrapMade "Tempo" theme (or any other third-party theme
or CSS framework) is not used anywhere in this codebase.

## Credits

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
History and Context. ACM Transactions on Interactive Intelligent
Systems 5, 4, Article 19.

Home page photo by [Felix Mooneeram](https://unsplash.com/@felixmooneeram)
on [Unsplash](https://unsplash.com).
