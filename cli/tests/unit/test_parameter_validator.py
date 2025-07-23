import pytest

pytestmark = pytest.mark.asyncio


def test_parameter_validator_import():
    try:
        from integration.parameter_validator import ParameterValidator
    except ImportError:
        pytest.skip("ParameterValidator module not available")


def test_parameter_validator_rules():
    from integration.parameter_validator import ParameterValidator

    validator = ParameterValidator()
    test_cases = [
        ({"reynolds": 1000000}, True),
        ({"reynolds": -1000}, False),
        ({"mach": 0.5}, True),
        ({"mach": 2.0}, False),
    ]
    for params, expected_valid in test_cases:
        result = validator.validate_parameters(params)
        is_valid = result.get("valid", False)
        assert is_valid == expected_valid
