"""
Test suite for the DatasetGenerator module of Icarus.

(c) Copyright UKAEA 2024.
"""

#import pytest
from unittest.mock import patch
from icarus import DatasetGenerator


values = [10, 20, 30, 40, 50]
with patch.object(DatasetGenerator, '__init__', lambda self: None):
    dataset_generator = DatasetGenerator()

def test_correct_number_of_validation_values_generated():
    """Test to make sure the correct number of validation values are generated
    """
    validation_values = dataset_generator.generate_validation_values(values, 3)
    assert len(validation_values) == 3, f"Incorrect number of validation values generated: was \
        {len(validation_values)}, should've been 3"


def test_validation_values_within_range():
    """Test to make correct validation values are generated
    """
    validation_values = dataset_generator.generate_validation_values(values, 3)
    min_val = min(values)
    max_val = max(values)
    for val in validation_values:
        assert val > min_val, "Validation value generated outside of acceptable range"
        assert val < max_val, "Validation value generated outside of acceptable range"


def test_no_duplicate_validation_values():
    """Test to make sure that no duplicate validation values are generated
    """
    validation_values = dataset_generator.generate_validation_values(values, 3)
    for val in validation_values:
        assert val not in values, "Validation value duplicate of parameter value"

    assert len(validation_values) == len(set(validation_values)), \
        "Same validation value generated multiple times"
