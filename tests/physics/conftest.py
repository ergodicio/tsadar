import pytest

from .adapter import TsadarAdapter


@pytest.fixture(scope="session")
def ts_adapter():
    return TsadarAdapter()
