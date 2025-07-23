import pytest
import asyncio

pytestmark = pytest.mark.asyncio


def test_database_manager_import():
    try:
        from data.database import DatabaseManager
    except ImportError:
        pytest.skip("DatabaseManager module not available")


@pytest.mark.asyncio
async def test_database_manager_crud():
    from data.database import DatabaseManager

    db_manager = DatabaseManager()
    await db_manager.initialize()
    test_data = {"name": "test_analysis", "type": "airfoil"}
    record_id = await db_manager.create_record("analyses", test_data)
    assert record_id is not None
    record = await db_manager.get_record("analyses", record_id)
    assert record is not None
    assert record["name"] == "test_analysis"
    updated_data = {"name": "updated_analysis"}
    success = await db_manager.update_record("analyses", record_id, updated_data)
    assert success
    success = await db_manager.delete_record("analyses", record_id)
    assert success
