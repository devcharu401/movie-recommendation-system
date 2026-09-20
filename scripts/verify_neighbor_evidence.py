import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from app import create_app
from app.services.recommendation_service import UserBasedRecommendation

TARGET_USER_ID = 10
TOP_MOVIES_TO_INSPECT = 3


def _print_neighbors(result: UserBasedRecommendation) -> None:
    print(f"Neighbours of user {TARGET_USER_ID} (ordered by similarity descending):")
    for neighbor in result.neighbors:
        print(f"  user {neighbor.user_id:>4}  similarity={neighbor.similarity:.4f}")


def _print_movie_evidence(result: UserBasedRecommendation) -> None:
    print(f"\nNeighbour ratings behind the top {TOP_MOVIES_TO_INSPECT} recommendations:")
    for record in result.recommendations[:TOP_MOVIES_TO_INSPECT]:
        movie_id = record["movie_id"]
        print(f"  {record['movie_title']} (movie_id={movie_id}, rank_score={record['rank_score']:.4f})")
        ratings = result.neighbor_ratings.get(movie_id, [])
        if not ratings:
            print("    no neighbour in the k-set rated this movie")
            continue
        for neighbor_rating in ratings:
            print(f"    user {neighbor_rating.neighbor_id:>4} rated it {neighbor_rating.rating}")


def _print_similarity_bounds(result: UserBasedRecommendation) -> None:
    similarities = [neighbor.similarity for neighbor in result.neighbors]
    print(f"\nSimilarity bounds across all neighbours: min={min(similarities):.4f}, max={max(similarities):.4f}")


def main() -> None:
    app = create_app()
    with app.app_context():
        service = app.extensions["recommendation_service"]
        result = service.recommend(user_id=TARGET_USER_ID)

    _print_neighbors(result)
    _print_movie_evidence(result)
    _print_similarity_bounds(result)


if __name__ == "__main__":
    main()
