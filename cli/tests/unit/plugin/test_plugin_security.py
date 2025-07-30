import pytest

def test_security_validator_creation() -> None:
    from cli.plugins.security import SecurityValidator
    validator = SecurityValidator()
    assert validator is not None
    assert len(validator.DANGEROUS_IMPORTS) > 0
    assert len(validator.DANGEROUS_FUNCTIONS) > 0

def test_manifest_validation() -> None:
    from cli.plugins.security import SecurityValidator
    from cli.plugins.models import PluginManifest, PluginAuthor, PluginType, SecurityLevel, PluginVersion
    validator = SecurityValidator()
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
    issues = validator._validate_manifest(manifest)
    assert len(issues) == 0
    manifest.name = ""
    issues = validator._validate_manifest(manifest)
    assert len(issues) > 0
    assert any("name is required" in issue for issue in issues)
