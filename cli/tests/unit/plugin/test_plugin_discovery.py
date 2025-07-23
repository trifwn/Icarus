import pytest
from pathlib import Path
import tempfile

def test_discovery_creation():
    from cli.plugins.discovery import PluginDiscovery
    discovery = PluginDiscovery()
    assert discovery is not None
    assert len(discovery.search_paths) >= 0

def test_add_search_path():
    from cli.plugins.discovery import PluginDiscovery
    discovery = PluginDiscovery()
    initial_count = len(discovery.search_paths)
    temp_dir = Path(tempfile.mkdtemp())
    discovery.add_search_path(temp_dir)
    assert len(discovery.search_paths) == initial_count + 1
    temp_dir.rmdir()
