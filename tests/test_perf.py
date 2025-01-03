import time

import pytest

pytestmark = [pytest.mark.monitor_skip_test]


def benchmark():
    """Function that needs some serious benchmarking."""
    time.sleep(1)

    # You may return anything you want, like the result of a computation
    return 123


@pytest.mark.monitor_test
def test_performance():
    """Function that is monitored by pytest-monitor."""
    result = benchmark()

    assert result == 123
