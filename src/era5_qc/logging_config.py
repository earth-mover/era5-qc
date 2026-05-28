from __future__ import annotations

import logging

from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """Emit log records through tqdm.write so progress bars stay intact."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s",
                                           datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
