import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import create_app
from app.ml.evaluation import DEFAULT_NEIGHBOR_COUNTS, EvaluationReport, run_evaluation
from app.ml.knn_engine import HIGH_RATING_THRESHOLD
from app.ml.preprocessing import build_preprocessed_dataset


def _print_report(report: EvaluationReport) -> None:
    print("Movie Recommendation System - Evaluation Report")
    print("=" * 64)
    print(f"User-movie matrix: {report.matrix_shape[0]} users x {report.matrix_shape[1]} movies")
    print(f"Sparsity: {report.sparsity:.2%} of matrix cells are unrated")
    print(f"Held-out split: {report.train_rows} train ratings, {report.test_rows} test ratings")
    print(f"Users evaluated per neighbour count: {report.sample_size}")
    print()
    print("Precision@N and Recall@N: N = top_n recommendations shown to a user.")
    print("A recommended movie counts as relevant if the user rated it")
    print(f">= {HIGH_RATING_THRESHOLD} in the held-out test ratings (the same threshold the")
    print("recommender itself uses for a 'highly rated' movie).")
    print("Coverage: share of the 338-movie candidate set that appeared in")
    print("at least one sampled user's recommendation list.")
    print()

    top_n = report.top_n
    header = (
        f"{'Neighbours (k)':>14} | {f'Precision@{top_n}':>14} | {f'Recall@{top_n}':>12} "
        f"| {'Coverage':>9} | {'Users w/ ground truth':>22}"
    )
    print(header)
    print("-" * len(header))
    for result in report.results:
        print(
            f"{result.neighbor_count:>14} | "
            f"{result.precision_at_k:>14.4f} | "
            f"{result.recall_at_k:>12.4f} | "
            f"{result.coverage:>9.2%} | "
            f"{result.evaluated_users:>22}"
        )


def main() -> None:
    app = create_app()
    with app.app_context():
        merged = build_preprocessed_dataset(app.config["MIN_RATINGS_PER_MOVIE"])
        report = run_evaluation(
            merged,
            neighbor_counts=DEFAULT_NEIGHBOR_COUNTS,
            top_n=app.config["TOP_N_RECOMMENDATIONS"],
        )
    _print_report(report)


if __name__ == "__main__":
    main()
