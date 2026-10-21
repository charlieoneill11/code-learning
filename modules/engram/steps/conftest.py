from pathlib import Path

import pytest
import torch

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def fixture():
    """fixture("03_hash") -> the dict saved by tools/make_fixtures.py."""
    return lambda name: torch.load(FIXTURES / f"{name}.pt", weights_only=False)
