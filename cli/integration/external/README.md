# External Tool Integration System

The External Tool Integration System provides comprehensive capabilities for integrating ICARUS CLI with external tools and services. This system enables seamless import/export of CAD files, cloud service integration, format conversion, and API adaptation for external tool updates.

## Features

### 1. CAD File Integration (`cad_integration.py`)

- **Supported Formats**: STEP, IGES, STL, OBJ, PLY
- **Geometry Validation**: Comprehensive validation with detailed error reporting
- **Batch Processing**: Concurrent import of multiple CAD files
- **Format Detection**: Automatic format detection by extension and content analysis

#### Key Capabilities:
- Import CAD files with automatic format detection
- Validate geometry integrity and structure
- Extract geometry information (vertices, faces, edges)
- Identify and report validation issues with suggestions
- Support for both ASCII and binary formats (where applicable)

#### Usage Example:
```python
from cli.integration.external import CADIntegration

cad_integration = CADIntegration()

# Import and validate a CAD file
cad_file = await cad_integration.import_cad_file("model.step")

if cad_file.is_valid:
    print(f"Geometry: {cad_file.validation_result.geometry_info.vertices} vertices")
else:
    for issue in cad_file.validation_result.issues:
        print(f"Issue: {issue.message}")
```

### 2. Cloud Service Integration (`cloud_integration.py`)

- **Supported Services**: AWS S3, Google Cloud Storage, Azure Blob Storage, Dropbox, OneDrive
- **Authentication Methods**: API Key, OAuth2, JWT, Basic Auth, Certificate-based
- **Operations**: Upload, download, list files, delete files
- **Security**: Secure credential management and encrypted communications

#### Key Capabilities:
- Register and manage multiple cloud services
- Secure authentication with multiple methods
- File upload/download with progress tracking
- Directory listing and file management
- Automatic retry and error handling

#### Usage Example:
```python
from cli.integration.external import CloudIntegration, CloudService

async with CloudIntegration() as cloud:
    # Register a service
    service = CloudService(
        name="my-s3",
        type=CloudServiceType.AWS_S3,
        auth_config=auth_config,
        base_url="https://s3.amazonaws.com",
        bucket_name="my-bucket"
    )
    cloud.register_service(service)

    # Upload a file
    success = await cloud.upload_file("my-s3", local_path, "remote/path")
```

### 3. Export Manager (`export_manager.py`)

- **Supported Formats**: JSON, CSV, XML, HDF5, MATLAB, Excel, ParaView, Tecplot
- **Format Conversion**: Convert between different data formats
- **Batch Export**: Export multiple datasets concurrently
- **Customization**: Configurable export options for each format

#### Key Capabilities:
- Export analysis results to multiple formats
- Convert between different file formats
- Batch processing for multiple datasets
- Format-specific optimization and compression
- Metadata preservation during conversion

#### Usage Example:
```python
from cli.integration.external import ExportManager, ExportFormatType

export_manager = ExportManager()

# Export data to JSON
result = await export_manager.export_data(
    data, output_path, ExportFormatType.JSON
)

# Convert format
result = await export_manager.convert_format(
    input_path, output_path, ExportFormatType.CSV
)
```

### 4. API Adapter (`api_adapter.py`)

- **Version Management**: Handle API version changes automatically
- **Request/Response Adaptation**: Adapt data formats between API versions
- **Change Detection**: Monitor API changes and notify handlers
- **Connectivity Testing**: Test API endpoint availability

#### Key Capabilities:
- Register external tool configurations
- Automatic API version detection and adaptation
- Handle breaking changes with migration strategies
- Monitor API updates and changes
- Test connectivity to external services

#### Usage Example:
```python
from cli.integration.external import APIAdapter, ExternalToolConfig

async with APIAdapter() as adapter:
    # Register external tool
    adapter.register_tool(tool_config)

    # Make API call with automatic adaptation
    result = await adapter.make_api_call(
        "tool-name", "endpoint-name", data
    )

    # Check for updates
    changes = await adapter.check_for_updates("tool-name")
```

## Data Models

The system uses comprehensive data models defined in `models.py`:

- **CADFile**: Represents imported CAD files with validation results
- **CloudService**: Configuration for cloud service connections
- **ExportFormat**: Defines supported export formats and options
- **APIEndpoint**: Configuration for external API endpoints
- **ValidationResult**: Detailed validation results with issues and suggestions

## Installation Requirements

```bash
# Core dependencies
pip install aiohttp aiofiles pandas numpy

# Optional dependencies for specific formats
pip install h5py scipy openpyxl  # For HDF5, MATLAB, Excel export
pip install semver  # For API version management
```

## Configuration

### Environment Variables

```bash
# Cloud service credentials
export AWS_ACCESS_KEY_ID="your-aws-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret"
export GOOGLE_CLOUD_CREDENTIALS="path/to/credentials.json"

# API endpoints
export EXTERNAL_TOOL_API_KEY="your-api-key"
export EXTERNAL_TOOL_BASE_URL="https://api.external-tool.com"
```

### Configuration Files

Create configuration files for external tools:

```json
{
  "tools": {
    "cad-tool": {
      "name": "cad-tool",
      "version": "2.1.0",
      "api_endpoints": [
        {
          "name": "version",
          "url": "https://api.cad-tool.com/v2/version",
          "method": "GET"
        }
      ],
      "supported_formats": ["step", "iges", "stl"]
    }
  }
}
```

## Error Handling

The system provides comprehensive error handling:

- **Validation Errors**: Detailed geometry validation with suggestions
- **Network Errors**: Automatic retry with exponential backoff
- **Authentication Errors**: Clear error messages with resolution steps
- **Format Errors**: Graceful handling of unsupported formats

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest cli/integration/external/test_external_integration.py

# Run specific test categories
python -m pytest cli/integration/external/test_external_integration.py::TestCADIntegration
python -m pytest cli/integration/external/test_external_integration.py::TestExportManager
```

## Demo

Run the demo script to see the system in action:

```bash
python cli/integration/external/demo_external_integration.py
```

The demo showcases:
- CAD file import and validation
- Multi-format data export
- Cloud service registration
- API adapter configuration
- Complete integration workflow

## Integration with ICARUS CLI

The external tool integration system is designed to integrate seamlessly with the main ICARUS CLI application:

```python
# In your ICARUS CLI application
from cli.integration.external import (
    CADIntegration, CloudIntegration,
    ExportManager, APIAdapter
)

class IcarusApp:
    def __init__(self):
        self.cad_integration = CADIntegration()
        self.export_manager = ExportManager()
        # ... other components

    async def import_cad_file(self, file_path):
        return await self.cad_integration.import_cad_file(file_path)

    async def export_results(self, data, format_type):
        return await self.export_manager.export_data(data, format_type)
```

## Security Considerations

- **Credential Storage**: Use secure credential storage (keyring, environment variables)
- **Network Security**: All communications use TLS/SSL encryption
- **Input Validation**: Comprehensive validation of all input data
- **Sandboxing**: External tool operations are isolated from core system
- **Audit Logging**: All operations are logged for security monitoring

## Performance Optimization

- **Async Operations**: All I/O operations are asynchronous for better performance
- **Concurrent Processing**: Batch operations use concurrent processing
- **Caching**: Intelligent caching of authentication tokens and API responses
- **Streaming**: Large file operations use streaming to minimize memory usage
- **Compression**: Automatic compression for supported formats

## Extensibility

The system is designed for easy extension:

1. **New CAD Formats**: Add format-specific validators to `CADIntegration`
2. **New Cloud Services**: Implement service-specific methods in `CloudIntegration`
3. **New Export Formats**: Add converters to `ExportManager`
4. **New API Adapters**: Register version-specific adapters in `APIAdapter`

## Troubleshooting

### Common Issues

1. **CAD Import Failures**
   - Check file format support
   - Verify file integrity
   - Review validation error messages

2. **Cloud Authentication Failures**
   - Verify credentials are correct
   - Check network connectivity
   - Review service-specific authentication requirements

3. **Export Failures**
   - Ensure output directory exists
   - Check available disk space
   - Verify format-specific dependencies are installed

4. **API Connectivity Issues**
   - Test network connectivity
   - Verify API endpoint URLs
   - Check authentication credentials

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

To contribute to the external tool integration system:

1. Follow the existing code structure and patterns
2. Add comprehensive tests for new functionality
3. Update documentation for any new features
4. Ensure backward compatibility when possible
5. Follow security best practices for external integrations

## License

This module is part of the ICARUS CLI system and follows the same licensing terms as the main project.
