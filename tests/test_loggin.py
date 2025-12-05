import logging
import os
import re
import pytest
from utils.logging import get_logger, colorlog

# FIXTURES

@pytest.fixture(scope="session")
def setup_logging_root():
    """Ensure the root logger level is set to DEBUG to allow all messages through."""
    logging.root.setLevel(logging.DEBUG)

@pytest.fixture
def clean_logger(setup_logging_root):
    """
    Fixture to remove all handlers and clear the logger instance 
    to ensure tests run in isolation.
    """
    logger_name = 'test_logger'
    logger = logging.getLogger(logger_name)
    
    # Remove existing handlers if present
    logger.handlers.clear()
    
    # Reset any configuration from previous tests
    logger.propagate = True
    logger.level = logging.NOTSET

    yield logger
    
    # Teardown: Clean up the logger after the test
    logger.handlers.clear()
    logger.level = logging.NOTSET


@pytest.fixture
def tmp_log_file(tmp_path):
    """Creates a temporary log file path within the pytest temporary directory."""
    # Use a subdirectory to test the directory creation logic
    log_dir = tmp_path / "custom_logs"
    log_file = log_dir / "test.log"
    return str(log_file)

# TESTS
def test_get_logger_initial_setup(clean_logger, tmp_log_file):
    """Test the initial setup of the logger."""
    
    logger_name = clean_logger.name
    logger = get_logger(name=logger_name, log_file=tmp_log_file)
    
    # 1. Check logger properties
    assert logger.name == logger_name
    assert logger.level == logging.DEBUG
    
    # 2. Check the number of handlers (Console + File)
    assert len(logger.handlers) == 2
    
    # 3. Check handler types
    console_handler = logger.handlers[0]
    file_handler = logger.handlers[1]
    
    assert isinstance(console_handler, colorlog.StreamHandler)
    assert isinstance(file_handler, logging.FileHandler)
    assert file_handler.baseFilename == tmp_log_file

    # 4. Check that running again returns the same logger without adding new handlers
    second_logger = get_logger(name=logger_name, log_file=tmp_log_file)
    assert second_logger is logger
    assert len(second_logger.handlers) == 2 # Should still be 2

def test_logger_creates_log_directory(clean_logger, tmp_path):
    """Test that the logs directory is created if it doesn't exist."""
    
    # Define a path that doesn't exist yet
    log_dir = tmp_path / "new_logs"
    log_file = log_dir / "test.log"
    assert not os.path.exists(log_dir)

    get_logger(name=clean_logger.name, log_file=str(log_file))
    
    # Check if the directory was created
    assert os.path.isdir(log_dir)

def test_logger_writes_to_file(clean_logger, tmp_log_file):
    """Test that log messages are correctly written to the file."""
    
    logger = get_logger(name=clean_logger.name, log_file=tmp_log_file)
    
    test_message = "File log test message"
    logger.info(test_message)
    
    for handler in logger.handlers:
        handler.close()

    with open(tmp_log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    expected_pattern = re.compile(
        r'^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\s-\s'  # asctime
        r'test_logger\s-\s'                            # name
        r'INFO\s-\s'                                   # levelname
        r'File log test message\s*$'                   # message
    )
    
    assert expected_pattern.search(content)

def test_console_handler_is_colored(clean_logger, tmp_log_file, capsys):
    """Test that the console output is handled by colorlog (implies coloring)."""
    
    logger = get_logger(name=clean_logger.name, log_file=tmp_log_file)
    
    assert isinstance(logger.handlers[0], colorlog.StreamHandler)
    
    test_message = "Console color test"
    logger.warning(test_message)
    
    captured = capsys.readouterr()
    
    # A common ANSI escape sequence for color is '\x1b['
    assert '\x1b[' in captured.err or '\x1b[' in captured.out
    assert test_message in captured.err or test_message in captured.out
