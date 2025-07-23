import pytest


@pytest.mark.asyncio
async def test_screen_manager_operations():
    try:

        class MockApp:
            def __init__(self):
                self.screens = {}
                self.event_system = MockEventSystem()
                self.log = MockLogger()

            def install_screen(self, screen, name):
                self.screens[name] = screen

            async def push_screen(self, name):
                pass

        class MockEventSystem:
            async def emit(self, event, data):
                pass

        class MockLogger:
            def error(self, message):
                pass

        from cli.app.screen_manager import ScreenManager

        app = MockApp()
        screen_manager = ScreenManager(app)
        await screen_manager.initialize()
        success = await screen_manager.switch_to("dashboard")
        assert success, "Should successfully switch to dashboard"
        current_screen = screen_manager.get_current_screen()
        assert current_screen is not None, "Should have current screen"
        assert current_screen.screen_name == "dashboard", "Should be dashboard screen"
        await screen_manager.switch_to("analysis")
        history = screen_manager.get_screen_history()
        assert len(history) == 1, "Should have one item in history"
        assert history[0] == "dashboard", "History should contain dashboard"
        success = await screen_manager.go_back()
        assert success, "Should successfully go back"
        assert screen_manager.current_screen == "dashboard", (
            "Should be back to dashboard"
        )
        await screen_manager.refresh_current()
        await screen_manager.cleanup_screen("dashboard")
    except ImportError:
        pytest.skip("ScreenManager module not available")
