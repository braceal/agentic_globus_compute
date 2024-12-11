"""HPC function calling for agentic workflows using Globus Compute."""

from __future__ import annotations

from typing import Any
from typing import Callable

from globus_compute_sdk import Executor


def globus_compute_executor(
    endpoint_id: str,
) -> Callable[..., Callable[..., Any]]:
    """Run a function remotely using Globus Compute.

    Decorator factory to execute functions remotely using Globus Compute.

    Parameters
    ----------
    endpoint_id : str
        The endpoint ID to use for the Globus Compute Executor.

    Returns
    -------
    Callable[..., Callable[..., Any]]
        A decorator that wraps the function to execute it remotely.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with Executor(endpoint_id=endpoint_id) as gce:
                print(
                    f"Submitting function '{func.__name__}' "
                    f'with args={args} kwargs={kwargs}',
                )
                future = gce.submit(func, *args, **kwargs)
                print('Function submitted, waiting for result...')
                result = future.result()
                print(
                    f"Function '{func.__name__}'"
                    f'completed with result: {result}',
                )
                return result

        return wrapper

    return decorator
