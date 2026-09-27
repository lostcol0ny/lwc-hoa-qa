import pytest
from web_fakes import Harness, build_harness


@pytest.fixture
def harness() -> Harness:
    """A test client over a fresh app with a recording fake asker."""
    return build_harness()
