# ICARUS CLI Project Structure

This document provides an overview of the ICARUS CLI project structure and organization.

## Directory Structure

```
icarus-cli/
├── cli/                    # Main CLI application code
│   ├── api/                # REST API implementation
│   ├── app/                # Core application components
│   ├── collaboration/      # Collaboration features
│   ├── commands/           # CLI command implementations
│   ├── config/             # Configuration management
│   ├── core/               # Core functionality and utilities
│   ├── data/               # Data management and storage
│   ├── documentation/      # Documentation generation
│   ├── examples/           # Example code and templates
│   ├── integration/        # Integration with external systems
│   ├── learning/           # Interactive learning system
│   ├── logs/               # Logging configuration
│   ├── plugins/            # Plugin system implementation
│   ├── security/           # Security and authentication
│   ├── testing/            # Test utilities and helpers
│   ├── tui/                # Terminal User Interface components
│   ├── visualization/      # Data visualization components
│   └── workflows/          # Workflow system implementation
├── docs/                   # Project documentation
│   ├── api/                # API documentation
│   ├── architecture/       # Architecture documentation
│   ├── developer_guides/   # Developer guides
│   ├── diagrams/           # Architecture diagrams
│   └── user_guides/        # User guides and tutorials
├── tests/                  # Test suite
│   ├── cli/                # CLI-specific tests
│   │   ├── analysis/       # Analysis system tests
│   │   ├── collaboration/  # Collaboration system tests
│   │   ├── framework/      # Core framework tests
│   │   ├── performance/    # Performance tests
│   │   ├── security/       # Security tests
│   │   └── workflow/       # Workflow system tests
│   ├── config/             # Configuration tests
│   ├── functional/         # Functional tests
│   ├── TestData/           # Test data files
│   └── unit/               # Unit tests
└── ICARUS/                 # Core ICARUS library
```

## Key Components

### CLI Application

The CLI application is built using the Textual framework and provides a modern terminal user interface for interacting with ICARUS functionality.

### API Layer

The API layer provides a RESTful interface to ICARUS functionality, enabling future web application development.

### Workflow System

The workflow system allows users to create, save, and execute complex analysis workflows.

### Collaboration Features

The collaboration system enables real-time collaboration between multiple users.

### Plugin System

The plugin system allows extending the CLI with custom functionality.

## Documentation Structure

- **API Documentation**: Details on API endpoints and usage
- **Architecture Documentation**: System architecture and design decisions
- **Developer Guides**: Guides for developers contributing to the project
- **User Guides**: Guides for end users of the CLI
- **Diagrams**: Visual representations of system architecture and components

## Testing Structure

- **Unit Tests**: Tests for individual components
- **Functional Tests**: Tests for complete features
- **Integration Tests**: Tests for integration between components
- **Performance Tests**: Tests for system performance
- **Security Tests**: Tests for security features
