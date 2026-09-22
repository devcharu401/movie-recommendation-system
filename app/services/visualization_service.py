from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

sns.set_theme(style="whitegrid")

AFFINITY_TEXT_COLOR = "#f3ece1"
AFFINITY_GRID_COLOR = "#b4a89a"
AFFINITY_VIEWER_COLOR = "#7a93a8"
AFFINITY_RECOMMENDATION_COLOR = "#e63950"


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


def plot_genre_affinity(affinity: pd.DataFrame, output_path: Path) -> Path:
    """Viewer genre mix vs. recommendation genre mix (spec 13.5, user-based
    results page). affinity has one row per genre and columns "genre",
    "Your ratings", "Your recommendations" (percentages, prepared by the
    caller). Styled for the dark results page rather than the report's
    default seaborn theme: transparent background, light text, no title —
    the page's own heading covers that."""
    melted = affinity.melt(id_vars="genre", var_name="series", value_name="share")
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(affinity) + 1.5))
    sns.barplot(
        data=melted,
        y="genre",
        x="share",
        hue="series",
        hue_order=["Your ratings", "Your recommendations"],
        palette={"Your ratings": AFFINITY_VIEWER_COLOR, "Your recommendations": AFFINITY_RECOMMENDATION_COLOR},
        ax=ax,
    )

    ax.set_xlabel("Share of films")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.tick_params(colors=AFFINITY_TEXT_COLOR)
    ax.xaxis.label.set_color(AFFINITY_TEXT_COLOR)
    ax.grid(False)
    ax.xaxis.grid(True, color=AFFINITY_GRID_COLOR, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    legend.set_title(None)
    for text in legend.get_texts():
        text.set_color(AFFINITY_TEXT_COLOR)

    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    fig.tight_layout()
    return _save(fig, output_path, transparent=True)


def generate_all_figures(merged: pd.DataFrame, figures_dir: Path, top_n: int) -> list[Path]:
    """Visualization entity behind Recommendation.visualize() (spec 11.5,
    13.6): produces the report's figures from the current dataset."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_rating_distribution(merged, figures_dir),
        plot_ratings_per_user(merged, figures_dir),
        plot_top_rated_movies(merged, figures_dir, top_n),
    ]


def _save(fig: plt.Figure, path: Path, *, transparent: bool = False) -> Path:
    # format is explicit, not inferred from path's suffix: the caching
    # temp-file-then-rename pattern saves to a ".png.tmp" path, whose
    # trailing extension isn't a format matplotlib recognises.
    fig.savefig(path, format="png", bbox_inches="tight", transparent=transparent)
    plt.close(fig)
    return path
