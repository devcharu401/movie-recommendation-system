from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_rating_distribution(merged: pd.DataFrame, figures_dir: Path) -> Path:
    """Rating distribution across the 1-5 scale (spec 13.6, evaluation ch.)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="rating", data=merged, order=sorted(merged["rating"].unique()), ax=ax)
    ax.set_title("Rating Distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of Ratings")
    return _save(fig, figures_dir / "rating_distribution.png")


def plot_ratings_per_user(merged: pd.DataFrame, figures_dir: Path) -> Path:
    """How many ratings each user contributed, to show engagement spread."""
    counts = merged.groupby("user_id").size()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(counts, bins=30, ax=ax)
    ax.set_title("Ratings per User")
    ax.set_xlabel("Number of Ratings")
    ax.set_ylabel("Number of Users")
    return _save(fig, figures_dir / "ratings_per_user.png")


def plot_top_rated_movies(merged: pd.DataFrame, figures_dir: Path, top_n: int) -> Path:
    """Highest average-rated movies among the filtered candidate set."""
    stats = merged.groupby("movie_title")["rating"].mean().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=stats.values, y=stats.index, ax=ax, orient="h")
    ax.set_title(f"Top {top_n} Rated Movies (avg rating)")
    ax.set_xlabel("Average Rating")
    ax.set_ylabel("Movie")
    fig.tight_layout()
    return _save(fig, figures_dir / "top_rated_movies.png")


def generate_all_figures(merged: pd.DataFrame, figures_dir: Path, top_n: int) -> list[Path]:
    """Visualization entity behind Recommendation.visualize() (spec 11.5,
    13.6): produces the report's figures from the current dataset."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_rating_distribution(merged, figures_dir),
        plot_ratings_per_user(merged, figures_dir),
        plot_top_rated_movies(merged, figures_dir, top_n),
    ]


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
