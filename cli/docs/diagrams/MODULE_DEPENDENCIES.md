# ICARUS CLI Module Dependencies

This document visualizes the dependencies between the major modules of the ICARUS CLI system.

## Module Dependency Graph

```mermaid
graph TD
    cli_main[cli/__main__.py] --> cli_icarus[cli/icarus.py]
    cli_main --> cli_streamlined[cli/streamlined_main.py]

    cli_icarus --> app_core[cli/app/core.py]
    cli_icarus --> app_config[cli/app/config.py]

    app_core --> tui_app[cli/tui/app.py]
    app_core --> core_session[cli/core/session.py]
    app_core --> core_events[cli/core/events.py]

    tui_app --> tui_screens[cli/tui/screens.py]
    tui_app --> tui_widgets[cli/tui/widgets.py]
    tui_app --> tui_themes[cli/tui/themes.py]

    tui_screens --> analysis_screens[cli/tui/analysis_screens.py]
    tui_screens --> workflow_screens[cli/tui/workflow_screens.py]
    tui_screens --> results_screens[cli/tui/results_screens.py]

    analysis_screens --> analysis_service[cli/analysis/service.py]
    workflow_screens --> workflow_engine[cli/workflows/engine.py]
    results_screens --> visualization_service[cli/visualization/service.py]

    analysis_service --> icarus_integration[cli/integration/icarus.py]
    analysis_service --> data_manager[cli/data/manager.py]

    workflow_engine --> analysis_service
    workflow_engine --> data_manager

    visualization_service --> data_manager

    app_config --> config_manager[cli/config/manager.py]

    core_session --> collaboration_service[cli/collaboration/service.py]
    core_session --> security_service[cli/security/service.py]

    collaboration_service --> api_websocket[cli/api/websocket.py]

    api_rest[cli/api/rest.py] --> analysis_service
    api_rest --> workflow_engine
    api_rest --> data_manager

    plugin_manager[cli/plugins/manager.py] --> app_core
    plugin_manager --> analysis_service
    plugin_manager --> workflow_engine
    plugin_manager --> visualization_service
```

## Layer Dependencies

```mermaid
graph TD
    subgraph "Presentation Layer"
        tui[TUI Components]
        api_endpoints[API Endpoints]
    end

    subgraph "Business Logic Layer"
        app_core[Application Core]
        analysis[Analysis Services]
        workflow[Workflow Engine]
        data[Data Management]
        visualization[Visualization]
    end

    subgraph "Integration Layer"
        icarus[ICARUS Integration]
        external[External Tools]
    end

    subgraph "Infrastructure Layer"
        config[Configuration]
        security[Security]
        collaboration[Collaboration]
        plugins[Plugin System]
    end

    tui --> app_core
    api_endpoints --> app_core

    app_core --> analysis
    app_core --> workflow
    app_core --> data
    app_core --> visualization

    analysis --> icarus
    analysis --> external

    workflow --> analysis
    workflow --> data

    visualization --> data

    app_core --> config
    app_core --> security
    app_core --> collaboration
    app_core --> plugins

    plugins --> analysis
    plugins --> workflow
    plugins --> visualization
```

## Package Dependencies

```mermaid
graph TD
    cli --> app
    cli --> tui
    cli --> api

    app --> core
    app --> config

    tui --> core
    tui --> analysis
    tui --> workflows
    tui --> visualization

    api --> analysis
    api --> workflows
    api --> data

    analysis --> integration
    analysis --> data

    workflows --> analysis
    workflows --> data

    visualization --> data

    core --> collaboration
    core --> security
    core --> plugins

    plugins --> analysis
    plugins --> workflows
    plugins --> visualization

    integration --> ICARUS[ICARUS Library]
```

## External Dependencies

```mermaid
graph TD
    cli[ICARUS CLI] --> textual[Textual]
    cli --> fastapi[FastAPI]
    cli --> pydantic[Pydantic]
    cli --> websockets[Websockets]
    cli --> matplotlib[Matplotlib]
    cli --> numpy[NumPy]
    cli --> icarus[ICARUS Library]

    textual --> rich[Rich]
    fastapi --> starlette[Starlette]

    icarus --> numpy
    icarus --> scipy[SciPy]
    icarus --> matplotlib
```
