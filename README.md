# Reelmate

A Flask web application that recommends movies from the MovieLens 100K
dataset using k-Nearest Neighbors collaborative filtering, in two modes:

- **User-based**: given a viewer ID from the dataset, find the most
  similar viewers (by cosine similarity of their rating patterns) and
  recommend films those viewers rated highly that the target viewer
  hasn't seen.
- **Item-based**: given a film title, find the most similar films by
  comparing how viewers rated them.

Only films with at least 100 ratings (configurable) are recommendable,
so every recommendation is backed by enough rating history to be
meaningful. A Browse page lets you explore the catalogue by genre, and
an About page explains the method and reports its measured accuracy.

Built as the practical component of MCSP-232, IGNOU MCA. See
`PROJECT_SPEC.md` for the full build contract this codebase follows.

## Requirements

- Python 3.12
- git

## Setup

Clone the repository, create a virtual environment, and install
dependencies:

### Windows (PowerShell)

```powershell
git clone <repository-url> movie-recommendation-system
cd movie-recommendation-system
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

### Linux / macOS

```bash
git clone <repository-url> movie-recommendation-system
cd movie-recommendation-system
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Building the database and model

Three steps, in order, turn a clean checkout into a runnable app:

```bash
python scripts/fetch_data.py
python scripts/load_database.py
python scripts/train_model.py
```

- **`fetch_data.py`** downloads `ml-100k.zip` from GroupLens, verifies
  it against the published MD5 checksum, extracts only the three files
  the next step needs (`u.user`, `u.item`, `u.data`) into
  `data/raw/ml-100k/`, and skips the download entirely if those files
  are already present. It fails with a clear error if the download
  fails or the checksum doesn't match.
- **`load_database.py`** loads those files into a SQLite database at
  `data/movie_recommendation.db`.
- **`train_model.py`** builds the user–movie feature matrices, fits the
  KNN models, persists them to `models/`, and runs the evaluation
  suite against the freshly trained model, persisting its results
  alongside the model artifacts.

`python scripts/initialise.py` runs all three in sequence and stops
with a clear error if any step fails; the three can also be run
individually, in that same order, if you need to redo just one.

## Running the app

```bash
flask --app wsgi run
```

Then open http://127.0.0.1:5000 in a browser.

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

## Project structure

Mapped to the numbered modules in `PROJECT_SPEC.md` section 4:

```
app/
    ml/                          13.1 Data Collection / 13.2 Preprocessing /
                                  13.3 Feature Construction
        data_loader.py           13.1 Data Collection
        preprocessing.py         13.2 Data Preprocessing
        feature_builder.py       13.3 Feature Construction
        knn_engine.py            13.4 Recommendation Engine
        evaluation.py            13.4 Recommendation Engine (metrics)
    services/
        recommendation_service.py  13.4 Recommendation Engine,
                                    13.6 Model Integration
        model_trainer.py           13.6 Model Integration
        visualization_service.py   13.6 Model Integration
    api/
        routes.py                13.5 User Interface
        validators.py             13.6 Model Integration (Input Validation Entity)
        errors.py                  13.6 Model Integration (Error Handling Entity)
        page_context.py            13.5 User Interface
    templates/, static/            13.5 User Interface
    repositories/                  Data layer (spec section 3), the only
                                    code that queries the database
    models/                        SQLAlchemy ORM models (users, movies,
                                    ratings — spec section 3.3)
    config.py                      Environment-driven configuration, read
                                    once at startup
    extensions.py                  The shared SQLAlchemy instance
    __init__.py                    Application factory
scripts/
    fetch_data.py       13.1 Data Collection — downloads and verifies ml-100k
    load_database.py    13.1 Data Collection — loads the raw files into SQLite
    train_model.py       13.3/13.4 — builds features, fits and persists the
                          KNN models, runs and persists the evaluation
    evaluate.py           13.4 Recommendation Engine (metrics) — reruns the
                          evaluation suite and prints the report
    initialise.py         Runs fetch_data.py, load_database.py and
                          train_model.py in sequence
tests/          Automated test suite (see "Running the tests" above)
docs/           Test case records and other project documentation
data/           Raw and processed data (gitignored; see "Dataset" below)
models/         Persisted model artefacts (gitignored)
reports/        Generated evaluation figures (gitignored)
wsgi.py         Entry point used by `flask run` / a production WSGI server
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

## Deployment

This project deploys to [Render](https://render.com) as a free-tier
Python web service, configured declaratively in `render.yaml` at the
project root (a Render "Blueprint").

- **Build**: `pip install -r requirements.txt && python scripts/fetch_data.py && python scripts/load_database.py && python scripts/train_model.py`
  — installs dependencies, downloads and verifies the dataset, loads it
  into SQLite, then builds and persists the KNN models to `models/`.
- **Start**: `gunicorn wsgi:app --bind 0.0.0.0:$PORT` — serves the app
  with gunicorn, binding to the port Render assigns via the `PORT`
  environment variable. Nothing in the codebase hardcodes a port;
  local development is unaffected and continues to use
  `flask --app wsgi run` on Flask's default port.
- **Health check**: `/health`, so Render can detect the service is up.

`render.yaml` sets `APP_ENV=production` (so debug mode stays off) and
has Render generate a random `SECRET_KEY` at deploy time — the
application factory refuses to start under the production config
without one.

To deploy: push this repository to GitHub (or GitLab) and create a new
Blueprint in the Render dashboard pointing at it; Render reads
`render.yaml` and provisions the service automatically.

If the service already exists and was **not** originally created
through Render's Blueprint flow (New → Blueprint), `render.yaml` has no
effect on it — Render only reads the file when instantiating a service
from a Blueprint, and otherwise uses whatever Build Command is set on
the service's own Settings page. Check the service's Settings page: if
its Build Command doesn't already match `render.yaml`, paste this in:

```
pip install -r requirements.txt && python scripts/fetch_data.py && python scripts/load_database.py && python scripts/train_model.py
```

## Building a submission copy

`scripts/build_submission.py` writes a dated ZIP of the project to
`dist/`, for handing in or archiving a snapshot separate from the git
history:

```bash
python scripts/build_submission.py
```

It always excludes `.venv/`, `.git/`, `__pycache__/`, `dist/`, log
files, IDE folders, `app/static/generated/`, and developer-tool
configuration such as `CLAUDE.md`. The database, trained model
artefacts, and raw MovieLens files are excluded by default — pass
`--include-data` to include them — because the dataset licence
restricts redistribution (see "Dataset" below).

## Dataset

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)
(`ml-100k`) — 100,000 ratings from 943 users on 1,682 movies, collected
by GroupLens Research at the University of Minnesota. It is downloaded
by `scripts/fetch_data.py` from `files.grouplens.org` at build time;
**it is not committed to this repository**, and `data/raw/` is
gitignored.

The dataset's own licence terms (`data/raw/ml-100k/README`, present
locally after `fetch_data.py` runs) require that:

- It be used for research purposes only, with results made publicly
  available.
- It **not be redistributed** without separate permission from
  GroupLens.
- It not be used for commercial or revenue-bearing purposes without
  permission.
- Any published or public work using it cite the dataset.

This is why it isn't committed here, and why `build_submission.py`
excludes it by default.

**Citation:**

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
History and Context. ACM Transactions on Interactive Intelligent
Systems 5, 4, Article 19.

## Note on styling

All CSS in `app/static/style.css` is original and hand-written for this
project. No third-party theme or CSS framework is used anywhere in this
codebase.

## Credits

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
History and Context. ACM Transactions on Interactive Intelligent
Systems 5, 4, Article 19.

Home page photo by [Felix Mooneeram](https://unsplash.com/@felixmooneeram)
on [Unsplash](https://unsplash.com).
