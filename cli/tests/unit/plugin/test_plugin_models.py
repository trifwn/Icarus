import pytest

def test_plugin_version_from_string():
    from cli.plugins.models import PluginVersion
    version = PluginVersion.from_string("1.2.3")
    assert version.major == 1
    assert version.minor == 2
    assert version.patch == 3
    assert version.pre_release is None
    version_pre = PluginVersion.from_string("2.0.0-beta1")
    assert version_pre.major == 2
    assert version_pre.minor == 0
    assert version_pre.patch == 0
    assert version_pre.pre_release == "beta1"

def test_plugin_version_to_string():
    from cli.plugins.models import PluginVersion
    version = PluginVersion(1, 2, 3)
    assert str(version) == "1.2.3"
    version_pre = PluginVersion(2, 0, 0, "beta1")
    assert str(version_pre) == "2.0.0-beta1"

def test_plugin_manifest_serialization():
    from cli.plugins.models import PluginManifest, PluginAuthor, PluginType, SecurityLevel, PluginVersion
    manifest = PluginManifest(
        name="test",
        version=PluginVersion(1, 0, 0),
        description="Test plugin",
        author=PluginAuthor("Test Author"),
        plugin_type=PluginType.UTILITY,
        security_level=SecurityLevel.SAFE,
        main_module="test",
        main_class="TestPlugin",
    )
    data = manifest.to_dict()
    assert data["name"] == "test"
    assert data["version"] == "1.0.0"
    assert data["type"] == "utility"
    manifest2 = PluginManifest.from_dict(data)
    assert manifest2.name == manifest.name
    assert str(manifest2.version) == str(manifest.version)
    assert manifest2.plugin_type == manifest.plugin_type
