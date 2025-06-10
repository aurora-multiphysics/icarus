"""
Test suite for the MooseSetup module of Icarus.

(c) Copyright UKAEA 2024.
"""

from unittest.mock import patch, MagicMock
import pytest
from icarus import MooseSetup


def test_invalid_file_extension_raises_file_not_found():
    """Test to make sure the correct error is raised when input file
    is invalid (doesn't end .i).
    """
    setup = MooseSetup(input_file_path="input.txt", output_file_path="",
                       n_tasks=4, n_threads=4)
    with pytest.raises(FileNotFoundError,
                       match="Unacceptable input file path input.txt. \
                                    Must both exist and end in .i"):
        setup.setup_moose_runner()


def test_nonexistent_input_file_raises_file_not_found(tmp_path):
    """Test to make sure the correct error is raised when input file
    doesn't exist.
    """
    fake_file = tmp_path / "input.i"
    setup = MooseSetup(input_file_path=str(fake_file), output_file_path="",
                       n_tasks=4, n_threads=4)

    with pytest.raises(FileNotFoundError,
                       match=f"Unacceptable input file path {str(fake_file)}. \
                                    Must both exist and end in .i"):
        setup.setup_moose_runner()


@patch("icarus.moosesetup.InputModifier")
def test_no_parameters_found_raises_value_error(mock_input_modifier, tmp_path):
    """Test to make sure the correct error is raised when no parameters
    are found
    """
    input_file = tmp_path / "input.i"
    input_file.write_text("Dummy content")

    mock_instance = MagicMock()
    mock_instance.get_vars.return_value = {}
    mock_input_modifier.return_value = mock_instance

    setup = MooseSetup(input_file_path=str(input_file), output_file_path="",
                       n_tasks=4, n_threads=4)

    with pytest.raises(ValueError, match="No parameters found in input file. \
                             Check input file and try again."):
        setup.setup_moose_runner()
