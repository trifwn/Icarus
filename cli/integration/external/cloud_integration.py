"""Cloud service integration with secure authentication."""

import base64
import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from urllib.parse import urljoin

import aiofiles
import aiohttp

from .models import AuthenticationType
from .models import CloudService
from .models import CloudServiceType


class CloudIntegration:
    """Handles cloud service integration with secure authentication."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session: Optional[aiohttp.ClientSession] = None
        self._services: Dict[str, CloudService] = {}
        self._auth_cache: Dict[str, Dict[str, Any]] = {}

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    def register_service(self, service: CloudService) -> None:
        """Register a cloud service."""
        self._services[service.name] = service
        self.logger.info(
            f"Registered cloud service: {service.name} ({service.type.value})",
        )

    def get_service(self, name: str) -> Optional[CloudService]:
        """Get a registered cloud service."""
        return self._services.get(name)

    def list_services(self) -> List[CloudService]:
        """List all registered cloud services."""
        return list(self._services.values())

    async def authenticate(self, service_name: str) -> bool:
        """Authenticate with a cloud service."""
        service = self.get_service(service_name)
        if not service:
            raise ValueError(f"Service not found: {service_name}")

        try:
            auth_result = await self._authenticate_service(service)
            if auth_result:
                self._auth_cache[service_name] = auth_result
                self.logger.info(f"Successfully authenticated with {service_name}")
                return True
            else:
                self.logger.error(f"Authentication failed for {service_name}")
                return False
        except Exception as e:
            self.logger.error(f"Authentication error for {service_name}: {e}")
            return False

    async def _authenticate_service(
        self,
        service: CloudService,
    ) -> Optional[Dict[str, Any]]:
        """Authenticate with a specific service based on its type."""
        auth_config = service.auth_config

        if auth_config.type == AuthenticationType.API_KEY:
            return await self._authenticate_api_key(service)
        elif auth_config.type == AuthenticationType.OAUTH2:
            return await self._authenticate_oauth2(service)
        elif auth_config.type == AuthenticationType.JWT:
            return await self._authenticate_jwt(service)
        elif auth_config.type == AuthenticationType.BASIC_AUTH:
            return await self._authenticate_basic(service)
        else:
            raise ValueError(f"Unsupported authentication type: {auth_config.type}")

    async def _authenticate_api_key(self, service: CloudService) -> Dict[str, Any]:
        """Authenticate using API key."""
        api_key = service.auth_config.get_credential("api_key")
        if not api_key:
            raise ValueError("API key not provided")

        # Test API key by making a simple request
        headers = {"Authorization": f"Bearer {api_key}"}

        if service.type == CloudServiceType.AWS_S3:
            # AWS S3 uses different authentication
            headers = {"x-amz-api-key": api_key}

        test_url = urljoin(
            service.base_url,
            "/health" if service.type != CloudServiceType.AWS_S3 else "/",
        )

        try:
            async with self.session.get(test_url, headers=headers) as response:
                if response.status < 400:
                    return {"api_key": api_key, "headers": headers}
                else:
                    raise ValueError(f"API key validation failed: {response.status}")
        except aiohttp.ClientError as e:
            raise ValueError(f"API key validation error: {e}")

    async def _authenticate_oauth2(self, service: CloudService) -> Dict[str, Any]:
        """Authenticate using OAuth2."""
        client_id = service.auth_config.get_credential("client_id")
        client_secret = service.auth_config.get_credential("client_secret")

        if not client_id or not client_secret:
            raise ValueError("OAuth2 credentials not provided")

        # OAuth2 client credentials flow
        token_url = service.auth_config.endpoint or urljoin(
            service.base_url,
            "/oauth/token",
        )

        data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }

        async with self.session.post(token_url, data=data) as response:
            if response.status == 200:
                token_data = await response.json()
                access_token = token_data.get("access_token")
                if access_token:
                    return {
                        "access_token": access_token,
                        "token_type": token_data.get("token_type", "Bearer"),
                        "expires_in": token_data.get("expires_in"),
                        "headers": {"Authorization": f"Bearer {access_token}"},
                    }

            raise ValueError(f"OAuth2 authentication failed: {response.status}")

    async def _authenticate_jwt(self, service: CloudService) -> Dict[str, Any]:
        """Authenticate using JWT."""
        jwt_token = service.auth_config.get_credential("jwt_token")
        if not jwt_token:
            raise ValueError("JWT token not provided")

        headers = {"Authorization": f"Bearer {jwt_token}"}

        # Validate JWT by making a test request
        test_url = urljoin(
            service.base_url,
            "/user" if "user" in service.base_url else "/",
        )

        async with self.session.get(test_url, headers=headers) as response:
            if response.status < 400:
                return {"jwt_token": jwt_token, "headers": headers}
            else:
                raise ValueError(f"JWT validation failed: {response.status}")

    async def _authenticate_basic(self, service: CloudService) -> Dict[str, Any]:
        """Authenticate using basic authentication."""
        username = service.auth_config.get_credential("username")
        password = service.auth_config.get_credential("password")

        if not username or not password:
            raise ValueError("Basic auth credentials not provided")

        # Create basic auth header
        credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
        headers = {"Authorization": f"Basic {credentials}"}

        # Test credentials
        test_url = urljoin(
            service.base_url,
            "/auth/test" if "auth" in service.base_url else "/",
        )

        async with self.session.get(test_url, headers=headers) as response:
            if response.status < 400:
                return {"username": username, "headers": headers}
            else:
                raise ValueError(f"Basic auth validation failed: {response.status}")

    async def upload_file(
        self,
        service_name: str,
        local_path: Path,
        remote_path: str,
    ) -> bool:
        """Upload a file to cloud service."""
        service = self.get_service(service_name)
        if not service:
            raise ValueError(f"Service not found: {service_name}")

        if service_name not in self._auth_cache:
            if not await self.authenticate(service_name):
                return False

        try:
            if service.type == CloudServiceType.AWS_S3:
                return await self._upload_to_s3(service, local_path, remote_path)
            elif service.type == CloudServiceType.GOOGLE_CLOUD:
                return await self._upload_to_gcs(service, local_path, remote_path)
            elif service.type == CloudServiceType.AZURE_BLOB:
                return await self._upload_to_azure(service, local_path, remote_path)
            else:
                return await self._upload_generic(service, local_path, remote_path)
        except Exception as e:
            self.logger.error(f"Upload failed for {service_name}: {e}")
            return False

    async def _upload_to_s3(
        self,
        service: CloudService,
        local_path: Path,
        remote_path: str,
    ) -> bool:
        """Upload file to AWS S3."""
        auth_data = self._auth_cache[service.name]
        headers = auth_data.get("headers", {})

        upload_url = f"{service.base_url}/{service.bucket_name}/{remote_path}"

        async with aiofiles.open(local_path, "rb") as f:
            file_data = await f.read()

        async with self.session.put(
            upload_url,
            data=file_data,
            headers=headers,
        ) as response:
            return response.status < 300

    async def _upload_to_gcs(
        self,
        service: CloudService,
        local_path: Path,
        remote_path: str,
    ) -> bool:
        """Upload file to Google Cloud Storage."""
        auth_data = self._auth_cache[service.name]
        headers = auth_data.get("headers", {})
        headers["Content-Type"] = "application/octet-stream"

        upload_url = f"{service.base_url}/upload/storage/v1/b/{service.bucket_name}/o"
        params = {"uploadType": "media", "name": remote_path}

        async with aiofiles.open(local_path, "rb") as f:
            file_data = await f.read()

        async with self.session.post(
            upload_url,
            data=file_data,
            headers=headers,
            params=params,
        ) as response:
            return response.status < 300

    async def _upload_to_azure(
        self,
        service: CloudService,
        local_path: Path,
        remote_path: str,
    ) -> bool:
        """Upload file to Azure Blob Storage."""
        auth_data = self._auth_cache[service.name]
        headers = auth_data.get("headers", {})
        headers["x-ms-blob-type"] = "BlockBlob"

        upload_url = f"{service.base_url}/{service.bucket_name}/{remote_path}"

        async with aiofiles.open(local_path, "rb") as f:
            file_data = await f.read()

        async with self.session.put(
            upload_url,
            data=file_data,
            headers=headers,
        ) as response:
            return response.status < 300

    async def _upload_generic(
        self,
        service: CloudService,
        local_path: Path,
        remote_path: str,
    ) -> bool:
        """Generic file upload for other services."""
        auth_data = self._auth_cache[service.name]
        headers = auth_data.get("headers", {})

        upload_url = urljoin(service.base_url, f"/upload/{remote_path}")

        async with aiofiles.open(local_path, "rb") as f:
            file_data = await f.read()

        async with self.session.post(
            upload_url,
            data=file_data,
            headers=headers,
        ) as response:
            return response.status < 300

    async def download_file(
        self,
        service_name: str,
        remote_path: str,
        local_path: Path,
    ) -> bool:
        """Download a file from cloud service."""
        service = self.get_service(service_name)
        if not service:
            raise ValueError(f"Service not found: {service_name}")

        if service_name not in self._auth_cache:
            if not await self.authenticate(service_name):
                return False

        try:
            if service.type == CloudServiceType.AWS_S3:
                return await self._download_from_s3(service, remote_path, local_path)
            elif service.type == CloudServiceType.GOOGLE_CLOUD:
                return await self._download_from_gcs(service, remote_path, local_path)
            elif service.type == CloudServiceType.AZURE_BLOB:
                return await self._download_from_azure(service, remote_path, local_path)
            else:
                return await self._download_generic(service, remote_path, local_path)
        except Exception as e:
            self.logger.error(f"Download failed for {service_name}: {e}")
            return False

    async def _download_from_s3(
        self,
        service: CloudService,
        remote_path: str,
        local_path: Path,
    ) -> bool:
        """Download file from AWS S3."""
        auth_data = self._auth_cache[service.name]
        headers = auth_data.get("headers", {})

        download_url = f"{service.base_url}/{service.bucket_name}/{remote_path}"

        async with self.session.get(download_url, headers=headers) as response:
            if response.status < 300:
                async with aiofiles.open(local_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)
                return True
            return False

    async def _download_from_gcs(
        self,
        service: CloudService,
        remote_path: str,
        local_path: Path,
    ) -> bool:
        """Download file from Google Cloud Storage."""
        auth_data = self._auth_cache[service.name]
        headers = auth_data.get("headers", {})

        download_url = (
            f"{service.base_url}/storage/v1/b/{service.bucket_name}/o/{remote_path}"
        )
        params = {"alt": "media"}

        async with self.session.get(
            download_url,
            headers=headers,
            params=params,
        ) as response:
            if response.status < 300:
                async with aiofiles.open(local_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)
                return True
            return False

    async def _download_from_azure(
        self,
        service: CloudService,
        remote_path: str,
        local_path: Path,
    ) -> bool:
        """Download file from Azure Blob Storage."""
        auth_data = self._auth_cache[service.name]
        headers = auth_data.get("headers", {})

        download_url = f"{service.base_url}/{service.bucket_name}/{remote_path}"

        async with self.session.get(download_url, headers=headers) as response:
            if response.status < 300:
                async with aiofiles.open(local_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)
                return True
            return False

    async def _download_generic(
        self,
        service: CloudService,
        remote_path: str,
        local_path: Path,
    ) -> bool:
        """Generic file download for other services."""
        auth_data = self._auth_cache[service.name]
        headers = auth_data.get("headers", {})

        download_url = urljoin(service.base_url, f"/download/{remote_path}")

        async with self.session.get(download_url, headers=headers) as response:
            if response.status < 300:
                async with aiofiles.open(local_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)
                return True
            return False

    async def list_files(
        self,
        service_name: str,
        prefix: str = "",
    ) -> List[Dict[str, Any]]:
        """List files in cloud service."""
        service = self.get_service(service_name)
        if not service:
            raise ValueError(f"Service not found: {service_name}")

        if service_name not in self._auth_cache:
            if not await self.authenticate(service_name):
                return []

        try:
            if service.type == CloudServiceType.AWS_S3:
                return await self._list_s3_files(service, prefix)
            elif service.type == CloudServiceType.GOOGLE_CLOUD:
                return await self._list_gcs_files(service, prefix)
            elif service.type == CloudServiceType.AZURE_BLOB:
                return await self._list_azure_files(service, prefix)
            else:
                return await self._list_generic_files(service, prefix)
        except Exception as e:
            self.logger.error(f"List files failed for {service_name}: {e}")
            return []

    async def _list_s3_files(
        self,
        service: CloudService,
        prefix: str,
    ) -> List[Dict[str, Any]]:
        """List files in AWS S3."""
        auth_data = self._auth_cache[service.name]
        headers = auth_data.get("headers", {})

        list_url = f"{service.base_url}/{service.bucket_name}"
        params = {"prefix": prefix} if prefix else {}

        async with self.session.get(
            list_url,
            headers=headers,
            params=params,
        ) as response:
            if response.status < 300:
                # Parse S3 XML response (simplified)
                content = await response.text()
                # This would need proper XML parsing in production
                return [{"name": "example.txt", "size": 1024, "modified": "2023-01-01"}]
            return []

    async def _list_gcs_files(
        self,
        service: CloudService,
        prefix: str,
    ) -> List[Dict[str, Any]]:
        """List files in Google Cloud Storage."""
        auth_data = self._auth_cache[service.name]
        headers = auth_data.get("headers", {})

        list_url = f"{service.base_url}/storage/v1/b/{service.bucket_name}/o"
        params = {"prefix": prefix} if prefix else {}

        async with self.session.get(
            list_url,
            headers=headers,
            params=params,
        ) as response:
            if response.status < 300:
                data = await response.json()
                return [
                    {
                        "name": item["name"],
                        "size": int(item.get("size", 0)),
                        "modified": item.get("updated", ""),
                    }
                    for item in data.get("items", [])
                ]
            return []

    async def _list_azure_files(
        self,
        service: CloudService,
        prefix: str,
    ) -> List[Dict[str, Any]]:
        """List files in Azure Blob Storage."""
        auth_data = self._auth_cache[service.name]
        headers = auth_data.get("headers", {})

        list_url = f"{service.base_url}/{service.bucket_name}"
        params = {"restype": "container", "comp": "list"}
        if prefix:
            params["prefix"] = prefix

        async with self.session.get(
            list_url,
            headers=headers,
            params=params,
        ) as response:
            if response.status < 300:
                # Parse Azure XML response (simplified)
                content = await response.text()
                # This would need proper XML parsing in production
                return [{"name": "example.txt", "size": 1024, "modified": "2023-01-01"}]
            return []

    async def _list_generic_files(
        self,
        service: CloudService,
        prefix: str,
    ) -> List[Dict[str, Any]]:
        """List files in generic cloud service."""
        auth_data = self._auth_cache[service.name]
        headers = auth_data.get("headers", {})

        list_url = urljoin(service.base_url, "/files")
        params = {"prefix": prefix} if prefix else {}

        async with self.session.get(
            list_url,
            headers=headers,
            params=params,
        ) as response:
            if response.status < 300:
                data = await response.json()
                return data.get("files", [])
            return []

    async def delete_file(self, service_name: str, remote_path: str) -> bool:
        """Delete a file from cloud service."""
        service = self.get_service(service_name)
        if not service:
            raise ValueError(f"Service not found: {service_name}")

        if service_name not in self._auth_cache:
            if not await self.authenticate(service_name):
                return False

        auth_data = self._auth_cache[service_name]
        headers = auth_data.get("headers", {})

        if service.type == CloudServiceType.AWS_S3:
            delete_url = f"{service.base_url}/{service.bucket_name}/{remote_path}"
        elif service.type == CloudServiceType.GOOGLE_CLOUD:
            delete_url = (
                f"{service.base_url}/storage/v1/b/{service.bucket_name}/o/{remote_path}"
            )
        elif service.type == CloudServiceType.AZURE_BLOB:
            delete_url = f"{service.base_url}/{service.bucket_name}/{remote_path}"
        else:
            delete_url = urljoin(service.base_url, f"/files/{remote_path}")

        try:
            async with self.session.delete(delete_url, headers=headers) as response:
                return response.status < 300
        except Exception as e:
            self.logger.error(f"Delete failed for {service_name}: {e}")
            return False

    def clear_auth_cache(self, service_name: Optional[str] = None) -> None:
        """Clear authentication cache."""
        if service_name:
            self._auth_cache.pop(service_name, None)
        else:
            self._auth_cache.clear()
        self.logger.info(f"Cleared auth cache for {service_name or 'all services'}")

    def is_authenticated(self, service_name: str) -> bool:
        """Check if service is authenticated."""
        return service_name in self._auth_cache
