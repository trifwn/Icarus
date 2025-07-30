import pytest


@pytest.mark.asyncio
async def test_event_system_subscription():
    try:
        from cli.app.event_system import EventSystem

        event_system = EventSystem()
        test_data = {}

        def test_callback(data):
            test_data.update(data)

        event_system.subscribe("test_event", test_callback)
        event_system.emit_sync("test_event", {"message": "test"})
        import asyncio

        await asyncio.sleep(0.1)
        assert "message" in test_data, "Event callback should have been called"
        assert test_data["message"] == "test", "Event data should match"
    except ImportError:
        pytest.skip("EventSystem module not available")


@pytest.mark.asyncio
async def test_event_system_async_events():
    try:
        from cli.app.event_system import EventSystem

        event_system = EventSystem()
        test_data = {"called": False}

        async def async_callback(data):
            test_data["called"] = True
            test_data.update(data)

        event_system.subscribe("async_test", async_callback)
        await event_system.emit("async_test", {"async_message": "test"})
        assert test_data["called"], "Async callback should have been called"
        assert test_data.get("async_message") == "test", "Async event data should match"
    except ImportError:
        pytest.skip("EventSystem module not available")
