# ICARUS CLI API Structure

This document describes the API structure for the ICARUS CLI project.

## API Overview

The ICARUS CLI provides a comprehensive API for interacting with ICARUS functionality. The API is designed to be:

- RESTful: Following REST principles for resource-based interactions
- Well-documented: Using OpenAPI specifications
- Versioned: Supporting multiple API versions
- Secure: With proper authentication and authorization

## API Endpoints

### Analysis API

The Analysis API provides endpoints for running analyses and retrieving results.

#### Endpoints

- `GET /api/v1/analysis/modules`: Get available analysis modules
- `GET /api/v1/analysis/solvers`: Get available solvers
- `POST /api/v1/analysis/run`: Run an analysis
- `GET /api/v1/analysis/status/{job_id}`: Get analysis status
- `GET /api/v1/analysis/results/{job_id}`: Get analysis results

#### Example Request

```json
POST /api/v1/analysis/run
{
  "analysis_type": "airfoil",
  "target": "NACA0012",
  "solver": "xfoil",
  "parameters": {
    "alpha": 5.0,
    "reynolds": 1000000,
    "mach": 0.0
  },
  "output_format": "json"
}
```

#### Example Response

```json
{
  "job_id": "12345",
  "status": "submitted",
  "estimated_time": 10
}
```

### Workflow API

The Workflow API provides endpoints for creating, managing, and executing workflows.

#### Endpoints

- `GET /api/v1/workflows`: Get available workflows
- `GET /api/v1/workflows/{workflow_id}`: Get workflow details
- `POST /api/v1/workflows`: Create a new workflow
- `PUT /api/v1/workflows/{workflow_id}`: Update a workflow
- `DELETE /api/v1/workflows/{workflow_id}`: Delete a workflow
- `POST /api/v1/workflows/{workflow_id}/execute`: Execute a workflow
- `GET /api/v1/workflows/{workflow_id}/status`: Get workflow execution status
- `GET /api/v1/workflows/{workflow_id}/results`: Get workflow execution results

#### Example Request

```json
POST /api/v1/workflows
{
  "name": "Airfoil Analysis",
  "description": "Analyze an airfoil at multiple angles of attack",
  "steps": [
    {
      "id": "step1",
      "name": "Analyze at alpha=0",
      "analysis_config": {
        "analysis_type": "airfoil",
        "target": "NACA0012",
        "solver": "xfoil",
        "parameters": {
          "alpha": 0.0,
          "reynolds": 1000000,
          "mach": 0.0
        }
      },
      "dependencies": []
    },
    {
      "id": "step2",
      "name": "Analyze at alpha=5",
      "analysis_config": {
        "analysis_type": "airfoil",
        "target": "NACA0012",
        "solver": "xfoil",
        "parameters": {
          "alpha": 5.0,
          "reynolds": 1000000,
          "mach": 0.0
        }
      },
      "dependencies": []
    }
  ]
}
```

### Data API

The Data API provides endpoints for managing data.

#### Endpoints

- `GET /api/v1/data/airfoils`: Get available airfoils
- `GET /api/v1/data/airfoils/{name}`: Get airfoil data
- `POST /api/v1/data/airfoils`: Upload airfoil data
- `GET /api/v1/data/airplanes`: Get available airplane configurations
- `GET /api/v1/data/airplanes/{name}`: Get airplane configuration
- `POST /api/v1/data/airplanes`: Upload airplane configuration
- `GET /api/v1/data/results`: Get available results
- `GET /api/v1/data/results/{result_id}`: Get result data
- `DELETE /api/v1/data/results/{result_id}`: Delete result data

### Collaboration API

The Collaboration API provides endpoints for collaboration features.

#### Endpoints

- `POST /api/v1/collaboration/sessions`: Create a collaboration session
- `GET /api/v1/collaboration/sessions/{session_id}`: Get session details
- `POST /api/v1/collaboration/sessions/{session_id}/join`: Join a session
- `POST /api/v1/collaboration/sessions/{session_id}/leave`: Leave a session
- `GET /api/v1/collaboration/sessions/{session_id}/users`: Get session users

### WebSocket API

The WebSocket API provides real-time communication for collaboration and progress updates.

#### Endpoints

- `ws://server/api/v1/ws/collaboration/{session_id}`: Collaboration session WebSocket
- `ws://server/api/v1/ws/progress/{job_id}`: Analysis progress WebSocket

#### Message Types

- `state_update`: Update to session state
- `user_joined`: User joined the session
- `user_left`: User left the session
- `progress_update`: Analysis progress update
- `result_available`: Analysis result available

## Authentication

The API uses JWT (JSON Web Tokens) for authentication.

### Authentication Endpoints

- `POST /api/v1/auth/login`: Login and get token
- `POST /api/v1/auth/refresh`: Refresh token
- `POST /api/v1/auth/logout`: Logout and invalidate token

### Example Authentication

```
POST /api/v1/auth/login
{
  "username": "user",
  "password": "password"
}
```

Response:

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

## API Documentation

The API is documented using OpenAPI specifications. The documentation is available at:

- `/api/docs`: Swagger UI for interactive documentation
- `/api/openapi.json`: OpenAPI specification in JSON format

## Error Handling

The API uses standard HTTP status codes and provides detailed error messages.

### Error Response Format

```json
{
  "error": {
    "code": "invalid_parameters",
    "message": "Invalid parameters provided",
    "details": {
      "alpha": "Value must be between -10 and 20"
    }
  }
}
```

### Common Error Codes

- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `409 Conflict`: Resource conflict
- `500 Internal Server Error`: Server error

## Rate Limiting

The API implements rate limiting to prevent abuse.

### Rate Limit Headers

- `X-RateLimit-Limit`: Maximum requests per time window
- `X-RateLimit-Remaining`: Remaining requests in current time window
- `X-RateLimit-Reset`: Time when the rate limit resets

## Versioning

The API is versioned using URL path versioning (e.g., `/api/v1/`).

### Version Support

- `v1`: Current stable version
- `v2`: (Future) Next version with enhanced features

## Client Libraries

Client libraries are available for common programming languages:

- Python: `icarus-client-python`
- JavaScript: `icarus-client-js`
- MATLAB: `icarus-client-matlab`
