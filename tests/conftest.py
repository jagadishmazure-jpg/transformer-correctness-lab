import pytest
import torch


@pytest.fixture
def tokens():
    torch.manual_seed(0)
    return torch.randint(0, 64, (2, 10))
