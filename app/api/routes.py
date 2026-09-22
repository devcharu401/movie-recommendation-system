from __future__ import annotations

from flask import Blueprint, current_app, jsonify, render_template, request

from app.api.page_context import landing_page_context
from app.api.validators import validate_movie_title, validate_user_id

bp = Blueprint("api", __name__)


def _service():
    return current_app.extensions["recommendation_service"]


@bp.route("/", methods=["GET"])
def index():
    service = _service()
    return render_template("index.html", **landing_page_context(service))


@bp.route("/recommend/user", methods=["POST"])
def recommend_user():
    service = _service()
    user_id = validate_user_id(request.form.get("user_id"), service.known_user_ids())
    result = service.recommend(user_id=user_id)
    profile = service.viewer_profile(user_id)
    return render_template(
        "user_recommendations.html",
        user_id=user_id,
        results=result.recommendations,
        profile=profile,
        neighbors=result.neighbors,
        contributing_ratings=result.contributing_neighbor_ratings,
    )


@bp.route("/recommend/movie", methods=["POST"])
def recommend_movie():
    service = _service()
    movie_title = validate_movie_title(request.form.get("movie_title"), service.recommendable_movie_titles())
    return _render_movie_recommendations(service, movie_title)


@bp.route("/movie/<path:movie_title>", methods=["GET"])
def recommend_movie_by_title(movie_title):
    service = _service()
    movie_title = validate_movie_title(movie_title, service.recommendable_movie_titles())
    return _render_movie_recommendations(service, movie_title)


def _render_movie_recommendations(service, movie_title):
    results = service.recommend(movie_title=movie_title)
    return render_template("movie_recommendations.html", movie_title=movie_title, results=results)


@bp.route("/api/movies", methods=["GET"])
def api_movies():
    service = _service()
    return jsonify(service.recommendable_movie_titles())


@bp.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok")
