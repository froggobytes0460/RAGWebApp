# pyright: reportAny=false
# pyright: reportExplicitAny=false

import logging
import logging.handlers
import queue
import sys
from typing import Any

from backend.core.config import LogSettings

_NOISY_LOGGERS: set[str] = {
    "uvicorn",
    "uvicorn.access",
    "uvicorn.error",
    "fastapi",
    "sqlalchemy.engine",
    "sqlalchemy.pool",
    "qdrant_client",
    "httpx",
    "httpcore",
    "huggingface_hub",
    "transformers",
    "tokenizers",
    "fastembed",
    "docling",
    "apscheduler",
    "slowapi",
    "langchain",
    "langchain_core",
    "langchain_qdrant",
}

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s [%(threadName)s] %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"


class AppLogger:
    _root: logging.Logger

    def __init__(self) -> None:
        self._listener: logging.handlers.QueueListener | None = None
        self._root = logging.getLogger()

    def setup(self, log_settings: LogSettings) -> None:
        formatter = logging.Formatter(fmt=_FORMAT, datefmt=_DATEFMT)
        log_queue: queue.Queue[logging.LogRecord] = queue.Queue(maxsize=-1)

        handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
        if log_settings.file_path is not None:
            handlers.append(
                logging.handlers.RotatingFileHandler(
                    filename=log_settings.file_path,
                    maxBytes=log_settings.max_bytes,
                    backupCount=log_settings.backup_count,
                    encoding="utf-8",
                )
            )
        for h in handlers:
            h.setFormatter(fmt=formatter)

        self._root.handlers.clear()
        self._root.setLevel(
            level=logging.getLevelNamesMapping().get(log_settings.level, logging.INFO)
        )
        self._root.addHandler(hdlr=logging.handlers.QueueHandler(queue=log_queue))

        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)

        self._listener = logging.handlers.QueueListener(
            log_queue, *handlers, respect_handler_level=True
        )
        self._listener.start()

    def shutdown(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def debug(self, msg: str, *args: Any) -> None:
        self._root.debug(msg, *args)

    def info(self, msg: str, *args: Any) -> None:
        self._root.info(msg, *args)

    def warning(self, msg: str, *args: Any) -> None:
        self._root.warning(msg, *args)

    def error(self, msg: str, *args: Any) -> None:
        self._root.error(msg, *args)

    def exception(self, msg: str, *args: Any) -> None:
        self._root.exception(msg, *args)

    def lifecycle(self, event: str, **context: object) -> None:
        if context:
            pairs = " ".join(f"{k}={v}" for k, v in context.items())
            self._root.info("%s — %s", event, pairs)
        else:
            self._root.info(event)

    def job_started(self, job_id: str, filename: str) -> None:
        self._root.info("Job %s started — file: %s", job_id, filename)

    def job_chunked(self, job_id: str, chunk_count: int) -> None:
        self._root.debug("Job %s produced %d chunk(s)", job_id, chunk_count)

    def job_complete(self, job_id: str, chunk_count: int, session_id: str) -> None:
        self._root.info(
            "Job %s complete — %d chunks for session %s",
            job_id,
            chunk_count,
            session_id,
        )

    def job_empty(self, job_id: str, filename: str) -> None:
        self._root.warning("Job %s — no text extracted from %s", job_id, filename)

    def job_failed(self, msg: str, job_id: str) -> None:
        self._root.exception(msg, job_id)


app_logger = AppLogger()
"""Singleton pattern app logger."""
