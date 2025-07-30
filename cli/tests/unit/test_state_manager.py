import pytest


@pytest.mark.asyncio
async def test_state_manager_session_management():
    try:
        from cli.app.state_manager import StateManager

        state_manager = StateManager()
        session = await state_manager.initialize_session()
        assert session is not None, "Session should be initialized"
        info = state_manager.get_session_info()
        assert "session_id" in info, "Session info should contain session_id"
        await state_manager.update_state("test_key", "test_value")
        state = state_manager.get_current_state()
        assert state.get("test_key") == "test_value", "State should be updated"
    except ImportError:
        pytest.skip("StateManager module not available")
