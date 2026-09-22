# Test cases

Automated test cases for the Testing chapter, run with Python's
standard-library `unittest`. Every ID below maps to one test method in
`tests/`; a row with several sampled inputs (e.g. a set of viewer IDs)
corresponds to a `subTest` loop inside that one method.

## Validation (`tests/test_validators.py`)

| ID | Layer | What is tested | Input | Expected result | Actual result |
|---|---|---|---|---|---|
| TC-VAL-01 | Validation | Viewer ID: valid ID is accepted | `"10"` | Returns `10` | Pass |
| TC-VAL-02 | Validation | Viewer ID: 0 is rejected | `"0"` | `ValidationError` | Pass |
| TC-VAL-03 | Validation | Viewer ID: highest valid ID is accepted | `str(max(known_user_ids))` | Returns that ID | Pass |
| TC-VAL-04 | Validation | Viewer ID: one above the highest is rejected | `str(max+1)` | `ValidationError` | Pass |
| TC-VAL-05 | Validation | Viewer ID: negative is rejected | `"-1"` | `ValidationError` | Pass |
| TC-VAL-06 | Validation | Viewer ID: non-numeric is rejected | `"abc"` | `ValidationError` | Pass |
| TC-VAL-07 | Validation | Viewer ID: empty is rejected | `""` | `ValidationError` | Pass |
| TC-VAL-08 | Validation | Film title: valid title is accepted | `"Toy Story (1995)"` | Returns the title | Pass |
| TC-VAL-09 | Validation | Film title: unknown title is rejected | `"Not A Real Film (2099)"` | `ValidationError` | Pass |
| TC-VAL-10 | Validation | Film title: empty is rejected | `""` | `ValidationError` | Pass |
| TC-VAL-11 | Validation | Film title: title with an apostrophe is accepted | `"Schindler's List (1993)"` | Returns the title | Pass |
| TC-VAL-12 | Validation | Film title: title with a comma and a colon is accepted | `"Godfather: Part II, The (1974)"` | Returns the title | Pass |
| TC-VAL-13 | Validation | Genre: valid genre is accepted | `"Drama"` | Returns the genre | Pass |
| TC-VAL-14 | Validation | Genre: unknown genre is rejected | `"NotAGenre"` | `ValidationError` | Pass |
| TC-VAL-15 | Validation | Genre: "Children's" is accepted | `"Children's"` | Returns the genre | Pass |
| TC-VAL-16 | Validation | Genre: "Film-Noir" is accepted | `"Film-Noir"` | Returns the genre | Pass |

## ML engine (`tests/test_knn_engine.py`)

| ID | Layer | What is tested | Input | Expected result | Actual result |
|---|---|---|---|---|---|
| TC-ML-01 | ML engine | Every neighbour similarity lies between 0 and 1 | Viewers 1, 10, 57, 100, 405, 500, 750, 900 | All similarities in `[0, 1]` | Pass |
| TC-ML-02 | ML engine | Every contributing rating meets the high-rating threshold | Same viewer sample | All contributing ratings `>= 4` | Pass |
| TC-ML-03 | ML engine | Rank score equals Σ(similarity × rating) over contributing neighbours (worked example) | Viewer 10, "Nightmare Before Christmas, The (1993)" | Equal to within `1e-9` | Pass |
| TC-ML-04 | ML engine | Neighbour top-rated titles ordered by rating, then rating count | Viewer 10's 5 neighbours | Each neighbour's `top_rated_movies` non-increasing on (rating, rating_count) | Pass |
| TC-ML-05 | ML engine | Neighbour top-rated title order is identical across two calls | Viewer 10, called twice | Both calls return identical tuples | Pass |
| TC-ML-06 | ML engine | Recommended films never include a film the viewer already rated | Same viewer sample | Recommended and already-rated movie ID sets disjoint | Pass |
| TC-ML-07 | ML engine | Item-based results never include the query film | "Toy Story (1995)", "Boot, Das (1981)", "Schindler's List (1993)" | Query title absent from its own results | Pass |

## Service layer (`tests/test_service.py`)

| ID | Layer | What is tested | Input | Expected result | Actual result |
|---|---|---|---|---|---|
| TC-SVC-01 | Service | Raw dataset statistics | — | 943 viewers, 1,682 films, 100,000 ratings | Pass |
| TC-SVC-02 | Service | Modelled dataset statistics | — | 338 films, 64,819 ratings | Pass |
| TC-SVC-03 | Service | Raw genre count | — | 18 genres | Pass |
| TC-SVC-04 | Service | Modelled matrix sparsity | — | Rounds to 79.66% | Pass |
| TC-SVC-05 | Service | Age band: under 18 | 17 | "Under 18" | Pass |
| TC-SVC-06 | Service | Age band: low edge of 18-24 | 18 | "18-24" | Pass |
| TC-SVC-07 | Service | Age band: high edge of 18-24 | 24 | "18-24" | Pass |
| TC-SVC-08 | Service | Age band: low edge of 25-34 | 25 | "25-34" | Pass |
| TC-SVC-09 | Service | Age band: high edge of 25-34 | 34 | "25-34" | Pass |
| TC-SVC-10 | Service | Age band: low edge of 35-44 | 35 | "35-44" | Pass |
| TC-SVC-11 | Service | Age band: high edge of 35-44 | 44 | "35-44" | Pass |
| TC-SVC-12 | Service | Age band: low edge of 45-54 | 45 | "45-54" | Pass |
| TC-SVC-13 | Service | Age band: high edge of 45-54 | 54 | "45-54" | Pass |
| TC-SVC-14 | Service | Age band: edge of 55-and-over | 55 | "55 and over" | Pass |
| TC-SVC-15 | Service | Age band: well above the 55-and-over edge | 90 | "55 and over" | Pass |
| TC-SVC-16 | Service | Viewer profile renders without error when occupation is unrecorded | Viewer 57 (occupation "none" in the source data) | `profile.occupation is None`, no exception | Pass |
| TC-SVC-17 | Service | Browse tile counts match `browse_by_genre()`, capped at the page size | All 18 genres | `len(films) == min(tile.film_count, 12)` | Pass |
| TC-SVC-18 | Service | Browse results sorted by mean rating, then rating count, then title | All 18 genres | List equals its own sort by `(-mean_rating, -rating_count, title)` | Pass |
| TC-SVC-19 | Service | Persisted evaluation results exist | — | `performance_report()` is not `None` | Pass |
| TC-SVC-20 | Service | Every persisted evaluation metric lies between 0 and 1 | Sparsity and every k's precision/recall/coverage | All in `[0, 1]` | Pass |

## Visualisation (`tests/test_visualization.py`)

| ID | Layer | What is tested | Input | Expected result | Actual result |
|---|---|---|---|---|---|
| TC-VIZ-01 | Visualisation | Genre affinity chart writes to a temporary directory and a second call reuses the file | Viewer 10's recommendations, a `tempfile.TemporaryDirectory()` | File under the temp dir; second call returns the same path with an unchanged mtime (no re-render) | Pass |

## Routes (`tests/test_routes.py`)

| ID | Layer | What is tested | Input | Expected result | Actual result |
|---|---|---|---|---|---|
| TC-RT-01 | Routes | Home page | `GET /` | 200 | Pass |
| TC-RT-02 | Routes | Browse index | `GET /browse` | 200 | Pass |
| TC-RT-03 | Routes | Browse a genre | `GET /browse/Drama` | 200 | Pass |
| TC-RT-04 | Routes | Browse a genre with an apostrophe | `GET /browse/Children's` | 200 | Pass |
| TC-RT-05 | Routes | About page | `GET /about` | 200 | Pass |
| TC-RT-06 | Routes | Health check | `GET /health` | 200 | Pass |
| TC-RT-07 | Routes | User-based recommendation, known viewer | `POST /recommend/user user_id=10` | 200 | Pass |
| TC-RT-08 | Routes | User-based recommendation, unknown viewer | `POST /recommend/user user_id=9999` | 400 | Pass |
| TC-RT-09 | Routes | User-based recommendation, viewer 0 | `POST /recommend/user user_id=0` | 400 | Pass |
| TC-RT-10 | Routes | User-based recommendation, non-numeric viewer | `POST /recommend/user user_id=abc` | 400 | Pass |
| TC-RT-11 | Routes | User-based recommendation, empty viewer | `POST /recommend/user user_id=` | 400 | Pass |
| TC-RT-12 | Routes | Item-based recommendation, known title | `POST /recommend/movie "Toy Story (1995)"` | 200 | Pass |
| TC-RT-13 | Routes | Item-based recommendation, unknown title | `POST /recommend/movie "Not A Real Film (2099)"` | 400 | Pass |
| TC-RT-14 | Routes | Item-based page by URL, comma title | `GET /movie/Boot, Das (1981)` | 200 | Pass |
| TC-RT-15 | Routes | Item-based page by URL, apostrophe title | `GET /movie/Schindler's List (1993)` | 200 | Pass |
| TC-RT-16 | Routes | Unknown genre on Browse | `GET /browse/NotAGenre` | 400 | Pass |
| TC-RT-17 | Routes | Unknown route | `GET /does-not-exist` | 404 | Pass |
| TC-RT-18 | Routes | Unhandled exception renders the error template and exposes no exception details | `GET /health`, with the view function replaced in memory by one that raises `RuntimeError` for the duration of the one request | 500; body contains "Something went wrong on our side." and neither the exception class name nor a traceback | Pass |

## Run record

- **Command:** `python -m unittest discover -s tests -v`
- **Date:** 2026-09-22
- **Python version:** 3.12.10
- **Total tests:** 62
- **Pass:** 62
- **Fail:** 0
- **Error:** 0

Database and model artifacts were present for this run. A checkout
without either does not skip: every test class that needs them raises
`tests.support.MissingPreconditionsError` from `setUpClass`, naming
`scripts/load_database.py` and `scripts/train_model.py`, and the suite
reports `FAILED (errors=N)` — never `OK` — so a missing precondition
cannot be mistaken for a pass. Confirmed by temporarily renaming
`models/` and re-running: `Ran 11 tests ... FAILED (errors=20)` (the 11
are `AgeBandBoundaryTests`, the only class with no database or model
dependency; every other class's `setUpClass` counted as one error).
Restoring `models/` and re-running returned to `Ran 62 tests ... OK`.
