import logging
import sys


def configure_logging(log_level: str) -> None:
    """Configure root logging handlers once, from the app's LOG_LEVEL setting."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
