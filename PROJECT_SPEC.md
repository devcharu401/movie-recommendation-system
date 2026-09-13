# PROJECT_SPEC.md

Authoritative build contract for **Movie Recommendation System Using Machine Learning** (IGNOU MCSP-232).
Derived from the approved synopsis. The synopsis is fixed; code conforms to it, not the reverse.
Anything not listed under "In Scope" is out of scope.

---

## 1. Identity

| Field | Value |
|---|---|
| Course code | MCSP-232 |
| Title | Movie Recommendation System Using Machine Learning |
| Candidate | Charulatha M S |
| Enrollment | 2450585749 |
| Guide | Anto Sunny |
| Programme | MCA_NEW, School of Computer and Information Sciences, IGNOU |
| Category | AI/ML — Recommender Systems (data-driven intelligent system) |

---

## 2. Locked technical decisions

Non-negotiable — each is named explicitly in the approved synopsis.

| Concern | Locked value |
|---|---|
| Language | Python 3.x |
| Web framework | Flask |
| Frontend | HTML + CSS (server-rendered Jinja templates) |
| ML library | scikit-learn |
| Data handling | NumPy, Pandas |
| Visualisation | Matplotlib, Seaborn |
| Dataset | MovieLens 100K (`ml-100k`) |
| Algorithm | k-Nearest Neighbors, `metric='cosine'`, `algorithm='brute'` |
| Filtering modes | User-based CF **and** item-based CF (both mandatory) |
| Candidate filter | Movies with **at least 100 ratings** |
| Result size | **Top 10** recommendations |
| Exclusion rule | Never recommend a movie the target user has already rated |
| User inputs | User ID (user-based) **or** movie title (item-based) |
| API style | RESTful Flask routes |
| VCS | Git + GitHub |

**Explicitly absent from the synopsis — do not build:** user registration, login, authentication, admin panel, rating submission by end users, React/Vue/JS SPA, external movie APIs (TMDB/OMDb), Docker, matrix factorisation, deep learning.

---

## 3. Data layer

### 3.1 Source
MovieLens 100K: `u.user`, `u.item`, `u.data` (pipe/tab delimited, ISO-8859-1 encoding).

### 3.2 Persistence decision
Synopsis §12 specifies three relational tables with SQL data types. Flat CSV-only handling would contradict the approved document.

**Decision: SQLite via SQLAlchemy ORM**, populated by a one-time loader script from the raw MovieLens files.

- Satisfies §12 and the ER diagram literally
- Zero-cost, file-based, portable, survives free-tier hosting with no external DB service
- `DATABASE_URL` env var allows a swap to MySQL/Postgres without code change

### 3.3 Schema (exact, per synopsis §12)

**users**
| Column | Type | Key |
|---|---|---|
| user_id | INT | PK |
| age | INT | |
| gender | VARCHAR(10) | |
| occupation | VARCHAR(100) | |
| zip_code | VARCHAR(10) | |

**movies**
| Column | Type | Key |
|---|---|---|
| movie_id | INT | PK |
| movie_title | VARCHAR(255) | |
| release_date | DATE | |
| genre | VARCHAR(100) | |

**ratings**
| Column | Type | Key |
|---|---|---|
| rating_id | INT | PK, autoincrement |
| user_id | INT | FK → users.user_id |
| movie_id | INT | FK → movies.movie_id |
| rating | INT (1–5) | |
| timestamp | DATETIME | |

### 3.4 Known transforms required at load
- `u.item` carries 19 binary genre flags; collapse to a pipe-delimited string to fit `genre VARCHAR(100)`
- `u.item` release dates are `dd-MMM-yyyy`; parse to `DATE`, allow NULL (a small number of rows are blank)
- `u.data` has no rating identifier; `rating_id` is a generated surrogate key
- Raw files are ISO-8859-1, not UTF-8 — explicit encoding on every read

---

## 4. Module map (synopsis §13 → code)

Report traceability depends on this mapping. Preserve it.

| § | Synopsis module | Code location |
|---|---|---|
| 13.1 | Data Collection | `scripts/load_database.py`, `app/ml/data_loader.py` |
| 13.2 | Data Preprocessing | `app/ml/preprocessing.py` |
| 13.3 | Feature Construction | `app/ml/feature_builder.py` |
| 13.4 | Recommendation Engine | `app/ml/knn_engine.py`, `app/services/recommendation_service.py` |
| 13.5 | User Interface | `app/templates/`, `app/static/`, `app/api/routes.py` |
| 13.6 | Model Integration | `app/api/`, `app/services/`, `app/api/errors.py` |

### 4.1 Required internal entities
Each must exist as a named class or function — the synopsis lists them individually and the report will reference them.

- Data Collection: User Data Loader, Movie Data Loader, Ratings Data Loader, Data Format Validator
- Preprocessing: Missing Value Handler, Duplicate Record Remover, Data Filtering Entity (≥100 ratings), Dataset Merger
- Feature Construction: User–Movie Matrix Generator, Feature Vector Builder, Sparse Matrix Handler (`scipy.sparse.csr_matrix`)
- Recommendation Engine: Similarity Calculator, KNN Neighbor Selector, User-Based Recommendation Entity, Item-Based Recommendation Entity, Recommendation Ranking Engine
- UI: Input Handler, Input Validation Entity, Request Dispatcher, Result Display Entity
- Model Integration: Controller Entity, Model Invocation Handler, Response Handler, Error Handling Entity

### 4.2 Class-diagram fidelity (synopsis §11.5)
Public method names on the recommendation service must match the approved class diagram:

- `Recommendation`: `visualize()`, `train_model()`, `recommend()`, `model()`; attributes `user_feature_df`, `movie_feature_df`
- `Application`: `recommend_user_based()`, `recommend_item_based()`

Snake_case is the Python-correct rendering of the diagram's names; the report notes this convention once.

---

## 5. Architecture (synopsis §14)

Four layers, strict one-way dependency:

```
Presentation  — Jinja templates + CSS
Application   — Flask blueprints, request validation, error handling
Service       — recommendation orchestration, model lifecycle
ML            — preprocessing, feature construction, KNN engine, evaluation
Data          — SQLAlchemy models, repositories, SQLite
```

Rules:
- Routes contain no business logic and no direct DB or DataFrame access
- Services never import Flask request/response objects
- ML layer is framework-agnostic, importable and testable standalone
- All data access goes through repositories
- Configuration read once, from environment, in `app/config.py`

---

## 6. Repository layout

```
movie-recommender/
├── app/
│   ├── __init__.py              application factory
│   ├── config.py
│   ├── extensions.py
│   ├── api/
│   │   ├── routes.py
│   │   ├── validators.py
│   │   └── errors.py
│   ├── services/
│   │   ├── recommendation_service.py
│   │   ├── model_trainer.py
│   │   └── visualization_service.py
│   ├── ml/
│   │   ├── data_loader.py
│   │   ├── preprocessing.py
│   │   ├── feature_builder.py
│   │   ├── knn_engine.py
│   │   └── evaluation.py
│   ├── repositories/
│   │   ├── user_repository.py
│   │   ├── movie_repository.py
│   │   └── rating_repository.py
│   ├── models/
│   │   ├── user.py
│   │   ├── movie.py
│   │   └── rating.py
│   ├── templates/
│   ├── static/
│   └── utils/
│       └── logging.py
├── scripts/
│   ├── fetch_data.py            downloads ml-100k
│   ├── load_database.py         raw files → SQLite
│   └── train_model.py           builds and persists KNN artefacts
├── tests/
├── data/                        gitignored
├── models/                      gitignored
├── docs/
├── .env.example
├── .gitignore
├── requirements.txt
├── wsgi.py
└── README.md
```

---

## 7. Routes

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Input form (user ID / movie title) |
| POST | `/recommend/user` | User-based top-10 |
| POST | `/recommend/movie` | Item-based top-10 |
| GET | `/api/movies` | Movie titles for autocomplete |
| GET | `/health` | Liveness check for the host platform |

---

## 8. Coding standard (global rule)

- Human-authored style. No comment restating what the next line does. No AI-boilerplate, no emoji, no decorative banners.
- Comments only for non-obvious intent: algorithm rationale, dataset quirks, deliberate trade-offs.
- Layered separation enforced as in §5. No business logic in routes.
- All configuration via environment variables; no hardcoded paths, no secrets in the repo.
- Paths resolved from `pathlib.Path(__file__).resolve().parents[n]` — never absolute.
- Input validation at the application boundary. Centralised error handling. Structured logging, no `print()`.
- Type hints on public functions; docstrings on public interfaces only.
- No dead code, no commented-out blocks, no unused imports.
- Tests live under `tests/`, mirroring the package structure.

---

## 9. Security (synopsis §15) — must be visibly implemented

- UI level: validate user ID is an integer within the known range; validate movie title against the movie table
- Application level: sanitise all inputs before service dispatch; explicit method restrictions per route
- Data level: recommendation path opens the database read-only; no write path exposed to users
- Version control: all changes tracked in Git with meaningful commit messages

---

## 10. Evaluation

The synopsis objectives require effectiveness evaluation but name no metric. Implemented in `app/ml/evaluation.py`:

- Precision@10 and Recall@10 on a held-out split
- Coverage (share of catalogue recommendable)
- Sparsity statistic of the user–movie matrix
- k sensitivity table across candidate k values

These feed the report's testing chapter directly.

---

## 11. Deployment

- Target: free-tier Flask host (Render free web service, or Hugging Face Spaces)
- Model artefacts built at deploy time by `scripts/train_model.py`, not committed
- Dataset fetched by `scripts/fetch_data.py`, not committed
- `/health` endpoint for platform probes
- Cold-start latency is acceptable for the free tier; note it in the report rather than engineering around it

---

## 12. Portability contract

A clean machine must reach a running app with only:

```bash
git clone <url> && cd movie-recommender
cp .env.example .env
pip install -r requirements.txt
python scripts/fetch_data.py
python scripts/load_database.py
python scripts/train_model.py
flask --app wsgi run
```

Anything breaking this chain is a defect, fixed at the gate where it appears.

---

## 13. Open items

| # | Item | Resolution point |
|---|---|---|
| 1 | Guide's DFD comment on the synopsis | G6 — after code is final |
| 2 | Synopsis states year of submission 2025; confirm the actual submission cycle for front matter | Before report writing |
| 3 | Baseline codebase gap list | G1 — after zip audit |
