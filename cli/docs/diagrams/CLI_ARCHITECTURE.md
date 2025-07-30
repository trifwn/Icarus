# ICARUS CLI Architecture

This document provides a visual representation of the ICARUS CLI architecture.

## High-Level Architecture

```mermaid
graph TB
    subgraph "Presentation Layer"
        TUI[Textual TUI Framework]
        WebUI[Web UI Framework - Future]
        Themes[Theme System]
        Layouts[Layout Manager]
        Widgets[Custom Widgets]
    end

    subgraph "API Layer"
        REST[REST API]
        WebSocket[WebSocket API]
        Events[Event System]
        Auth[Authentication]
    end

    subgraph "Business Logic Layer"
        Core[Core Application]
        Session[Session Manager]
        Workflow[Workflow Engine]
        Plugin[Plugin System]
    end

    subgraph "Service Layer"
        Analysis[Analysis Services]
        Data[Data Management]
        Collab[Collaboration]
        Export[Export/Import]
    end

    subgraph "Integration Layer"
        ICARUS[ICARUS Modules]
        Solvers[External Solvers]
        CAD[CAD Integration]
        Cloud[Cloud Services]
    end

    TUI --> REST
    WebUI -.-> REST
    REST --> Core
    WebSocket --> Events
    Events --> Core

    Core --> Session
    Core --> Workflow
    Core --> Plugin

    Session --> Analysis
    Workflow --> Analysis
    Analysis --> Data
    Analysis --> Export

    Analysis --> ICARUS
    Analysis --> Solvers
    Data --> CAD
    Data --> Cloud
```

## Component Relationships

```mermaid
classDiagram
    class IcarusCLI {
        +screen_manager: ScreenManager
        +session_manager: SessionManager
        +workflow_engine: WorkflowEngine
        +plugin_manager: PluginManager
        +run()
        +register_screen(screen)
        +switch_screen(screen_id)
    }

    class ScreenManager {
        +screens: Dict[str, Screen]
        +active_screen: Screen
        +register(screen)
        +switch_to(screen_id)
        +get_screen(screen_id)
    }

    class SessionManager {
        +user_id: str
        +workspace: str
        +active_analyses: List[str]
        +preferences: UserPreferences
        +save_session()
        +load_session()
        +update_state(key, value)
    }

    class WorkflowEngine {
        +create_workflow(template)
        +execute_workflow(workflow)
        +get_templates()
        +save_workflow(workflow)
    }

    class PluginManager {
        +discover_plugins()
        +load_plugin(plugin_id)
        +get_plugin_api()
        +validate_plugin(plugin)
    }

    class AnalysisService {
        +get_available_modules()
        +run_analysis(config)
        +validate_parameters(params)
        +get_solver_info(solver_name)
    }

    IcarusCLI --> ScreenManager
    IcarusCLI --> SessionManager
    IcarusCLI --> WorkflowEngine
    IcarusCLI --> PluginManager
    WorkflowEngine --> AnalysisService
    SessionManager --> AnalysisService
```

## Data Flow

```mermaid
flowchart TD
    User[User Input] --> TUI[TUI Layer]
    TUI --> EventSystem[Event System]
    EventSystem --> CoreApp[Core Application]
    CoreApp --> BusinessLogic[Business Logic]
    BusinessLogic --> ICARUS[ICARUS Modules]
    ICARUS --> Results[Analysis Results]
    Results --> DataManagement[Data Management]
    DataManagement --> Visualization[Visualization]
    Visualization --> TUI

    CoreApp --> API[API Layer]
    API --> WebClient[Web Client - Future]
```

## Workflow System

```mermaid
stateDiagram-v2
    [*] --> Configure
    Configure --> Validate
    Validate --> Execute
    Execute --> Results
    Results --> Visualization
    Visualization --> Export
    Export --> [*]

    Validate --> Configure : Invalid Parameters
    Execute --> Error : Execution Failure
    Error --> Configure : Retry
    Error --> [*] : Abort
```

## Plugin System

```mermaid
graph LR
    subgraph "Plugin System"
        PluginAPI[Plugin API]
        PluginLoader[Plugin Loader]
        PluginRegistry[Plugin Registry]
        PluginSandbox[Plugin Sandbox]
    end

    subgraph "Core System"
        CoreApp[Core Application]
        EventSystem[Event System]
        ExtensionPoints[Extension Points]
    end

    subgraph "Plugins"
        CustomAnalysis[Custom Analysis]
        CustomVisualization[Custom Visualization]
        ExternalIntegration[External Integration]
    end

    PluginAPI --> PluginLoader
    PluginLoader --> PluginRegistry
    PluginRegistry --> PluginSandbox
    PluginSandbox --> ExtensionPoints
    ExtensionPoints --> CoreApp
    CoreApp --> EventSystem

    CustomAnalysis --> PluginAPI
    CustomVisualization --> PluginAPI
    ExternalIntegration --> PluginAPI
```

## Collaboration System

```mermaid
graph TB
    subgraph "Local System"
        LocalUser[Local User]
        LocalTUI[Local TUI]
        LocalState[Local State]
        SyncManager[Sync Manager]
    end

    subgraph "Server"
        WebSocket[WebSocket Server]
        StateManager[State Manager]
        AuthService[Auth Service]
    end

    subgraph "Remote System"
        RemoteUser[Remote User]
        RemoteTUI[Remote TUI]
        RemoteState[Remote State]
        RemoteSyncManager[Sync Manager]
    end

    LocalUser --> LocalTUI
    LocalTUI --> LocalState
    LocalState --> SyncManager
    SyncManager --> WebSocket

    WebSocket --> StateManager
    StateManager --> AuthService

    WebSocket --> RemoteSyncManager
    RemoteSyncManager --> RemoteState
    RemoteState --> RemoteTUI
    RemoteTUI --> RemoteUser
```
