import sys
sys.path.append("..")

import pytest
import torch
import numpy as np
from torch.nn import Module
import utils
from unittest.mock import MagicMock

def test_save_checkpoint(mocker):
    test_model = Module()
    save_mock: MagicMock = mocker.MagicMock()
    with mocker.patch("torch.save", save_mock):
        utils.save_checkpoint(test_model, 123, 456, "pytest_")
    save_mock.assert_called_once()
    args = save_mock.call_args[0]
    assert args[0]["epoch"] == 123
    assert "./saves" in args[1]
    assert "123" in args[1]
    assert "456" in args[1]
