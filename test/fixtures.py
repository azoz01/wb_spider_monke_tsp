import pytest
from numpy.random import RandomState


@pytest.fixture(scope="session", autouse=True)
def mock_random(monkeypatch: pytest.MonkeyPatch):
    def stable_random(*args, **kwargs):
        rs = RandomState(12345)
        return rs.random(*args, **kwargs)

    monkeypatch.setattr("numpy.random.random", stable_random)
