import logging

from src.utils.logging_utils import setup_logging


def test_setup_logging(tmp_path, monkeypatch):
    log_dir = tmp_path
    monkeypatch.setattr("src.utils.logging_utils.LOG_DIR", log_dir)
    monkeypatch.setattr("src.utils.logging_utils.LOG_FORMAT", "%(levelname)s | %(message)s")

    logging.getLogger().handlers.clear()

    setup_logging()

    logging.info("Test Message")

    log_file = log_dir / "log.log"

    content = log_file.read_text()

    assert log_file.exists()
    assert "INFO | Test Message" in content


def test_logging_level_is_info(tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.logging_utils.LOG_DIR", tmp_path)
    monkeypatch.setattr("src.utils.logging_utils.LOG_FORMAT", "%(levelname)s | %(message)s")

    logging.getLogger().handlers.clear()

    setup_logging()

    assert logging.getLogger().level == logging.INFO


def test_logging_appends(tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.logging_utils.LOG_DIR", tmp_path)
    monkeypatch.setattr("src.utils.logging_utils.LOG_FORMAT", "%(levelname)s | %(message)s")

    logging.getLogger().handlers.clear()
    setup_logging()
    logging.info("first")

    logging.getLogger().handlers.clear()
    setup_logging()
    logging.info("second")

    content = (tmp_path / "log.log").read_text()
    assert "first" in content
    assert "second" in content
