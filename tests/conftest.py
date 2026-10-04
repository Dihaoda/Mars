from pathlib import Path

import pytest

from mars.config import load_config


@pytest.fixture
def small_config():
    return load_config(Path(__file__).parents[1] / "configs" / "cpu_smoke.yaml")
