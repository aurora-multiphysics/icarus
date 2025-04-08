import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from icarus import MooseSetup


def test_invalid_file_extension_raises_file_not_found():
    setup = MooseSetup(input_file_path="input.txt", n_tasks=4, n_threads=4)  
    with pytest.raises(FileNotFoundError, match="Unacceptable input file path."):
        setup.setup_moose_runner()


def test_nonexistent_input_file_raises_file_not_found(tmp_path):
    fake_file = tmp_path / "input.i"
    setup = MooseSetup(input_file_path=str(fake_file), n_tasks=4, n_threads=4)

    with pytest.raises(FileNotFoundError, match="Unacceptable input file path."):
        setup.setup_moose_runner()


@patch("icarus.moosesetup.InputModifier")  
def test_no_parameters_found_raises_value_error(mock_input_modifier, tmp_path):
    input_file = tmp_path / "input.i"
    input_file.write_text("Dummy content")

    mock_instance = MagicMock()
    mock_instance.get_vars.return_value = {}
    mock_input_modifier.return_value = mock_instance

    setup = MooseSetup(input_file_path=str(input_file), n_tasks=4, n_threads=4) 

    with pytest.raises(ValueError, match="No parameters found in input file"):
        setup.setup_moose_runner()