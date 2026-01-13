"""
Utility decorators for logging, deprecation warnings, and performance timing.

This module provides:

- `deprecated`: Marks a function as deprecated and logs a warning.
- `get_time`: Measures and logs the execution time of a function.
"""

from functools import wraps
from typing import TypeVar, Callable, Any
import warnings
import logging
import time

logger = logging.getLogger(__name__)
T = TypeVar("T")


def deprecated(reason: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to mark a function as deprecated.

    Emits a `DeprecationWarning` when the function is called and logs
    a warning message using the configured logger.
    :param reason: The reason why the function is deprecated. This message will be included in the warning.
    :return: A decorator that wraps the target function and issues the deprecation warning.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            warnings.warn(
                f"{func.__name__} is deprecated: {reason}",
                DeprecationWarning,
                stacklevel=2,
            )
            logger.warning("%s is deprecated: %s", func.__name__, reason)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_time(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to measure and log the execution time of a function.
    The elapsed time is logged at INFO level using the configured logger.
    :param func: The function whose execution time should be measured.
    :return: A wrapper function that executes the original function and logs its execution time.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start: float = time.perf_counter()
        result: T = func(*args, **kwargs)
        elapsed: float = time.perf_counter() - start
        logger.info("%s took %.3f seconds", func.__name__, elapsed)
        return result

    return wrapper
