from __future__ import annotations

from flask import current_app


def landing_page_context(service) -> dict:
    return {
        "movie_titles": service.recommendable_movie_titles(),
        "stats": service.dataset_statistics(),
        "max_user_id": max(service.known_user_ids()),
        "min_ratings": current_app.config["MIN_RATINGS_PER_MOVIE"],
    }
