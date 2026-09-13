from __future__ import annotations

import logging

from flask import Flask, current_app, jsonify, render_template, request

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised by the Input Validation Entity when request data fails validation."""


def register_error_handlers(app: Flask) -> None:
    """Error Handling Entity (spec 13.5/13.6): centralises 404, 500 and
    validation-error handling so no route carries its own error handling,
    and no stack trace ever reaches the client."""

    @app.errorhandler(ValidationError)
    def handle_validation_error(error: ValidationError):
        if request.path.startswith("/api/"):
            return jsonify(error=str(error)), 400
        service = current_app.extensions["recommendation_service"]
        return (
            render_template("index.html", movie_titles=service.recommendable_movie_titles(), error=str(error)),
            400,
        )

    @app.errorhandler(404)
    def handle_not_found(error):
        if request.path.startswith("/api/"):
            return jsonify(error="not found"), 404
        return "<h1>404</h1><p>The page you requested does not exist.</p>", 404

    @app.errorhandler(500)
    def handle_server_error(error):
        logger.exception("unhandled server error")
        if request.path.startswith("/api/"):
            return jsonify(error="internal server error"), 500
        return "<h1>500</h1><p>Something went wrong. Please try again.</p>", 500
