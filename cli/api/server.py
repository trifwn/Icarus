"""
ICARUS CLI API Server

This script starts the FastAPI server for the ICARUS CLI API layer.
Use this to test the API endpoints and WebSocket functionality.
"""

import logging

import uvicorn

from .app import create_api_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Start the API server"""
    logger.info("Starting ICARUS CLI API server...")

    # Create the FastAPI app
    app = create_api_app()

    # Run the server
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
        reload=False,  # Set to True for development
    )


if __name__ == "__main__":
    main()
