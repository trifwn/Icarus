import pytest


@pytest.mark.asyncio
async def test_workflow_engine_operations():
    try:
        from cli.core.workflow import WorkflowEngine

        engine = WorkflowEngine()
        workflows = engine.get_workflows()
        assert isinstance(workflows, list) and len(workflows) > 0, (
            "Should have built-in workflows"
        )
        workflow_info = engine.get_available_workflows()
        assert isinstance(workflow_info, list), "Should return list of workflow info"
        templates = engine.get_workflow_templates()
        if templates:
            workflow = engine.create_workflow(templates[0])
            assert workflow is not None, "Should create workflow from template"
    except ImportError:
        pytest.skip("WorkflowEngine module not available")
