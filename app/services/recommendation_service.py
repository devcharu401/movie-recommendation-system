from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.neighbors import NearestNeighbors

from app.ml.evaluation import EvaluationReport
from app.ml.knn_engine import HIGH_RATING_THRESHOLD, Neighbor, NeighborRating, recommend_item_based, recommend_user_based
from app.repositories.rating_repository import get_dataset_statistics, get_user_genre_distribution
from app.repositories.user_repository import get_user_profile
from app.services.model_trainer import ModelArtifacts, ModelTrainer
from app.services.visualization_service import generate_all_figures, plot_genre_affinity

NO_OCCUPATION_RECORDED = "none"
TOP_GENRE_COUNT = 3
BROWSE_TOP_N = 12
AFFINITY_TOP_VIEWER_GENRES = 6


def _age_band(age: int) -> str:
    if age < 18:
        return "Under 18"
    if age <= 24:
        return "18-24"
    if age <= 34:
        return "25-34"
    if age <= 44:
        return "35-44"
    if age <= 54:
        return "45-54"
    return "55 and over"


@dataclass(frozen=True)
class DatasetOverview:
    """Landing-page dataset overview (spec 13.5, Result Display Entity).
    raw_* comes straight from the database; modelled_* is what actually
    reaches the KNN models after preprocessing's filters (spec 13.2)."""

    raw_viewers: int
    raw_films: int
    raw_ratings: int
    raw_genres: int
    modelled_viewers: int
    modelled_films: int
    modelled_ratings: int
    modelled_genres: int
    earliest_rating: datetime
    latest_rating: datetime


@dataclass(frozen=True)
class GenreShare:
    genre: str
    percentage: int


@dataclass(frozen=True)
class GenreTile:
    """One tile on the Browse page (spec 13.5): a genre and how many
    recommendable films carry it."""

    name: str
    film_count: int


@dataclass(frozen=True)
class BrowseFilm:
    """One film card on the Browse page (spec 13.5)."""

    movie_id: int
    movie_title: str
    genre: str
    release_year: int | None
    mean_rating: float
    rating_count: int


@dataclass(frozen=True)
class ViewerProfile:
    """Viewer profile panel above the user-based results (spec 13.5)."""

    age_band: str
    occupation: str | None
    films_rated: int
    mean_rating: float
    top_genres: list[GenreShare]


@dataclass(frozen=True)
class UserBasedRecommendation:
    """Service-layer return value for the user-based path (spec: neighbour
    evidence must reach the caller instead of being discarded after
    ranking). recommendations keeps its existing record shape; neighbors,
    neighbor_ratings and contributing_neighbor_ratings are the KNN engine's
    supporting evidence for it. neighbor_ratings is every neighbour who
    rated the movie at all; contributing_neighbor_ratings is the subset that
    met the scoring threshold and so actually shaped the ranking — kept
    separate so a caller cannot mix the two up."""

    recommendations: list[dict]
    neighbors: list[Neighbor]
    neighbor_ratings: dict[int, list[NeighborRating]]
    contributing_neighbor_ratings: dict[int, list[NeighborRating]]


class Recommendation:
    """Recommendation (class diagram, spec 11.5): holds the trained feature
    matrices and KNN models for the process lifetime. Artifacts are loaded
    once here, at construction, not per recommend() call."""

    def __init__(
        self,
        trainer: ModelTrainer,
        similar_users_count: int,
        top_n_recommendations: int,
        figures_dir: Path,
        generated_dir: Path,
    ) -> None:
        self._trainer = trainer
        self._similar_users_count = similar_users_count
        self._top_n_recommendations = top_n_recommendations
        self._figures_dir = figures_dir
        self._generated_dir = generated_dir

        self._apply(self._load_artifacts())

    def _load_artifacts(self) -> ModelArtifacts:
        try:
            return self._trainer.load()
        except FileNotFoundError as exc:
            raise RuntimeError(
                "no trained model artifacts found; run scripts/train_model.py before starting the app"
            ) from exc

    def _apply(self, artifacts: ModelArtifacts) -> None:
        self.user_feature_df = artifacts.user_feature_df
        self.movie_feature_df = artifacts.movie_feature_df
        self._movie_catalog = artifacts.movie_catalog
        self._merged_dataset = artifacts.merged_dataset
        self._user_model = artifacts.user_model
        self._movie_model = artifacts.movie_model

    def train_model(self) -> None:
        """Rebuilds and persists the matrices and models, then reloads this
        instance's state from the new artifacts."""
        self._apply(self._trainer.train())

    def model(self) -> tuple[NearestNeighbors, NearestNeighbors]:
        """Returns the fitted (user_model, movie_model) pair."""
        return self._user_model, self._movie_model

    def recommend(
        self, *, user_id: int | None = None, movie_title: str | None = None
    ) -> UserBasedRecommendation | list[dict]:
        """Dispatches to the user-based or item-based path depending on which
        identifier is supplied. User-based carries neighbour evidence
        alongside the ranking; item-based keeps its existing record-list
        shape, since neighbour evidence has no meaning there."""
        if user_id is None and movie_title is None:
            raise ValueError("provide exactly one of user_id or movie_title")
        if user_id is not None and movie_title is not None:
            raise ValueError("provide exactly one of user_id or movie_title")
        if user_id is not None:
            recommendations, neighbors, neighbor_ratings, contributing_neighbor_ratings = recommend_user_based(
                user_id,
                self.user_feature_df,
                self._user_model,
                self._movie_catalog,
                self._similar_users_count,
                self._top_n_recommendations,
            )
            return UserBasedRecommendation(
                recommendations, neighbors, neighbor_ratings, contributing_neighbor_ratings
            )
        return recommend_item_based(
            movie_title,
            self.movie_feature_df,
            self._movie_model,
            self._movie_catalog,
            self._top_n_recommendations,
        )

    def visualize(self) -> list[Path]:
        """Generates the report figures (spec 13.6) from the currently
        loaded dataset."""
        return generate_all_figures(self._merged_dataset, self._figures_dir, self._top_n_recommendations)

    def known_user_ids(self) -> list[int]:
        """User IDs present in the trained user-movie matrix, for input
        validation (spec 13.5)."""
        return list(self.user_feature_df.index)

    def recommendable_movie_titles(self) -> list[str]:
        """The filtered, recommendable movie set (spec 13.5), for input
        validation and the /api/movies autocomplete source."""
        return sorted(self._movie_catalog.index)

    def dataset_statistics(self) -> DatasetOverview:
        """Raw counts come from the database via the repository; modelled
        counts come from the merged, filtered frame already held in memory
        from artifact load, so this never re-queries or re-derives what
        train_model() already produced. merged has one row per surviving
        rating, so user_id/movie_id nunique() and len() give viewers, films
        and ratings on the right axis; genre still needs splitting on '|'
        for the same reason as the raw count."""
        raw = get_dataset_statistics()
        merged = self._merged_dataset
        modelled_genre_tokens = {
            genre for combination in merged["genre"].dropna().unique() if combination for genre in combination.split("|")
        }
        return DatasetOverview(
            raw_viewers=raw["total_viewers"],
            raw_films=raw["total_films"],
            raw_ratings=raw["total_ratings"],
            raw_genres=raw["distinct_genres"],
            modelled_viewers=int(merged["user_id"].nunique()),
            modelled_films=int(merged["movie_id"].nunique()),
            modelled_ratings=int(len(merged)),
            modelled_genres=len(modelled_genre_tokens),
            earliest_rating=raw["earliest_rating"],
            latest_rating=raw["latest_rating"],
        )

    def performance_report(self) -> EvaluationReport | None:
        """Persisted evaluation results for the About page (spec 10), or
        None if scripts/train_model.py has not produced one yet — the page
        then shows metrics as unavailable rather than erroring."""
        return self._trainer.load_evaluation()

    def relevance_threshold(self) -> int:
        """Minimum held-out rating that counts as 'relevant' in the
        evaluation metrics (spec 10) — the same threshold the recommender
        itself uses for a 'highly rated' movie."""
        return HIGH_RATING_THRESHOLD

    def browse_genres(self) -> list[GenreTile]:
        """Genre tiles for the Browse page (spec 13.5), one per genre found
        in the recommendable movie catalog — the same filtered set the KNN
        models are trained on, so a tile's count always matches what
        browse_by_genre() lists for it, with no separate SQL re-filter."""
        counts: dict[str, int] = {}
        for combination in self._movie_catalog["genre"]:
            if not combination:
                continue
            for genre in combination.split("|"):
                counts[genre] = counts.get(genre, 0) + 1
        return sorted(
            (GenreTile(name=genre, film_count=count) for genre, count in counts.items()),
            key=lambda tile: tile.name,
        )

    def browse_by_genre(self, genre: str) -> list[BrowseFilm]:
        """Top BROWSE_TOP_N films carrying `genre` (spec 13.5), ranked by
        mean rating, then by how many viewers rated it, then title, so the
        order is deterministic when films tie on mean rating — common at
        this catalogue size. Mean rating is computed from the same merged,
        filtered dataset the models were trained on, not a fresh query."""
        catalog = self._movie_catalog
        carries_genre = catalog["genre"].apply(lambda combination: bool(combination) and genre in combination.split("|"))
        subset = catalog[carries_genre]

        mean_ratings = self._merged_dataset.groupby("movie_id")["rating"].mean()
        ranked = subset.assign(mean_rating=subset["movie_id"].map(mean_ratings)).sort_values(
            by=["mean_rating", "rating_count", "movie_title"], ascending=[False, False, True]
        )

        return [
            BrowseFilm(
                movie_id=int(row.movie_id),
                movie_title=row.movie_title,
                genre=row.genre,
                release_year=row.release_date.year if row.release_date else None,
                mean_rating=round(float(row.mean_rating), 2),
                rating_count=int(row.rating_count),
            )
            for row in ranked.head(BROWSE_TOP_N).itertuples()
        ]

    def viewer_profile(self, user_id: int) -> ViewerProfile:
        """Profile panel shown above one user's recommendations (spec 13.5).
        Age banding happens here, not in the repository or the template, so
        the band boundaries live in exactly one place."""
        facts = get_user_profile(user_id)
        if facts is None:
            raise ValueError(f"unknown user_id: {user_id}")

        occupation = facts["occupation"]
        if not occupation or occupation.strip().lower() == NO_OCCUPATION_RECORDED:
            occupation = None

        top_genres = [
            GenreShare(genre=entry["genre"], percentage=round(entry["share"] * 100))
            for entry in get_user_genre_distribution(user_id)[:TOP_GENRE_COUNT]
        ]

        return ViewerProfile(
            age_band=_age_band(facts["age"]),
            occupation=occupation,
            films_rated=facts["films_rated"],
            mean_rating=round(facts["mean_rating"], 1),
            top_genres=top_genres,
        )

    def genre_affinity_chart(self, user_id: int, recommendations: list[dict]) -> Path:
        """Genre affinity chart for the user-based results page (spec
        13.5): the viewer's genre mix against their recommendations' genre
        mix. One PNG per user id, cached under GENERATED_DIR — a user's
        recommendations only change when the model is retrained, so
        re-rendering on every request would be wasted work. Written to a
        temporary file and renamed into place so a concurrent request
        never reads a half-written image."""
        path = self._generated_dir / f"genre_affinity_{user_id}.png"
        if path.exists():
            return path

        viewer_distribution = get_user_genre_distribution(user_id)
        viewer_share = {entry["genre"]: entry["share"] for entry in viewer_distribution}
        top_viewer_genres = {entry["genre"] for entry in viewer_distribution[:AFFINITY_TOP_VIEWER_GENRES]}

        recommendation_genres = [
            genre for movie in recommendations for genre in movie["genre"].split("|") if genre
        ]
        recommendation_share = {
            genre: recommendation_genres.count(genre) / len(recommendations)
            for genre in set(recommendation_genres)
        }

        genres = sorted(
            top_viewer_genres | set(recommendation_share),
            key=lambda genre: (-viewer_share.get(genre, 0.0), genre),
        )
        affinity = pd.DataFrame(
            {
                "genre": genres,
                "Your ratings": [viewer_share.get(genre, 0.0) * 100 for genre in genres],
                "Your recommendations": [recommendation_share.get(genre, 0.0) * 100 for genre in genres],
            }
        )

        self._generated_dir.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".png.tmp")
        plot_genre_affinity(affinity, temp_path)
        temp_path.replace(path)
        return path
