import pytest

pytestmark = pytest.mark.asyncio


def test_result_processor_import():
    try:
        from integration.result_processor import ResultProcessor
    except ImportError:
        pytest.skip("ResultProcessor module not available")


def test_result_processor_basic():
    from integration.result_processor import ResultProcessor

    processor = ResultProcessor()
    raw_results = {
        "data": [1, 2, 3, 4, 5],
        "metadata": {"solver": "xfoil", "type": "polar"},
    }
    processed = processor.process_results(raw_results)
    assert processed is not None
    assert "processed_data" in processed
    formatted = processor.format_results(processed, "table")
    assert formatted is not None
