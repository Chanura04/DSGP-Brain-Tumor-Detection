import pytest
from src.utils.decorators import deprecated, get_time, log_action


def test_deprecated_returns_value():
    @deprecated("use new_func instead")
    def old_func(x, y):
        return x + y

    assert old_func(2, 3) == 5


def test_deprecated_emits_warning():
    @deprecated("use new_func instead")
    def old_func():
        return "ok"

    with pytest.warns(DeprecationWarning) as record:
        old_func()

    assert "old_func is deprecated: use new_func instead" in str(record[0].message)


def test_deprecated_logs_warning(caplog):
    @deprecated("use new_func instead")
    def old_func():
        return "ok"

    with caplog.at_level("WARNING"):
        old_func()

    assert "old_func is deprecated: use new_func instead" in caplog.text


def test_deprecated_preserves_metadata():
    @deprecated("use new_func instead")
    def old_func():
        """Some docstring"""
        return 1

    assert old_func.__name__ == "old_func"
    assert old_func.__doc__ == "Some docstring"


def test_get_time_returns_value():
    @get_time
    def func(x, y):
        return x + y

    assert func(2, 3) == 5


def test_get_time_logs_info(caplog):
    @get_time
    def func():
        return "ok"

    with caplog.at_level("INFO"):
        func()

    assert "func took" in caplog.text


def test_get_time_propagates_exception():
    @get_time
    def func():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        func()


def test_get_time_preserves_metadata():
    @get_time
    def func():
        """Some docstring"""
        return 1

    assert func.__name__ == "func"
    assert func.__doc__ == "Some docstring"


def test_log_action_logs_call_and_return(caplog):
    @log_action
    def add(x, y):
        return x + y

    with caplog.at_level("INFO"):
        result = add(2, 3)

    assert result == 5

    # Check call log
    assert any("Calling add" in record.message for record in caplog.records)

    # Check return log
    assert any("Returning add" in record.message for record in caplog.records)


def test_log_action_logs_arguments(caplog):
    @log_action
    def greet(name, age=None):
        return "ok"

    with caplog.at_level("INFO"):
        greet("Alice", age=30)

    assert any(
        "args=('Alice',)" in record.message and "kwargs={'age': 30}" in record.message for record in caplog.records)


def test_log_action_logs_exception_and_reraises(caplog):
    @log_action
    def boom():
        raise ValueError("fail")

    with caplog.at_level("ERROR"):
        with pytest.raises(ValueError):
            boom()

    assert any(record.levelname == "ERROR" and "Stopped boom: failed" in record.message for record in caplog.records)


def test_log_action_preserves_metadata():
    @log_action
    def func():
        """Some docstring"""
        return 1

    assert func.__name__ == "func"
    assert func.__doc__ == "Some docstring"
