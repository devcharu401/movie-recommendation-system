from __future__ import annotations


def landing_page_context(service) -> dict:
    return {
        "movie_titles": service.recommendable_movie_titles(),
        "stats": service.dataset_statistics(),
    }
