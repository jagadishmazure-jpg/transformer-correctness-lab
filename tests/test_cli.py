import pytest

from tclab.__main__ import main


@pytest.mark.parametrize("mod", ["encoder", "parallel", "positions", "lora", "leak"])
def test_cli_modules_pass(mod):
    assert main([mod]) == 0
