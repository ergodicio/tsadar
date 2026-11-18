"""tsverify's conformance tests (L2, L3, A10, A12 and the A1 sweep) run against tsadar through ``TsadarAdapter``."""

import pytest
from tsverify.conformance import *  # noqa: F401,F403

pytestmark = pytest.mark.physics
