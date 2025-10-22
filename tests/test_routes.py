"""
Unit tests for general API routes (vector status, CSV download, store helper)
"""

import pytest
import json
import tempfile
import os
import uuid
from unittest.mock import Mock, patch, AsyncMock
from fastapi.responses import JSONResponse, FileResponse
from app.api.aoss_routes import check_vectordb_status
import app.api.domain_schema_routes as domains_module
from app.api.domain_schema_routes import download_csv_file
from app.aoss.column_store import get_store

class TestVectorStatusRoute:
    @pytest.mark.asyncio
    @patch('app.api.aoss_routes.get_store')
    async def test_vector_status_success(self, mock_get_store):
        mock_client = Mock()
        mock_client.indices.exists.return_value = True
        mock_client.indices.stats.return_value = {
            "indices": {
                "test_index": {
                    "total": {
                        "docs": {"count": 100},
                        "store": {"size_in_bytes": 1024}
                    }
                }
            }
        }
        mock_store = Mock()
        mock_store.client = mock_client
        mock_store.index_name = "test_index"
        mock_get_store.return_value = mock_store
        result = await check_vectordb_status()
        assert isinstance(result, JSONResponse)
        assert result.status_code == 200
        content = json.loads(result.body.decode())
        assert content["status"] == "connected"
        assert content["index_exists"] is True
        assert content["document_count"] == 100

    @pytest.mark.asyncio
    @patch('app.api.aoss_routes.get_store')
    async def test_vector_status_no_index(self, mock_get_store):
        mock_client = Mock()
        mock_client.indices.exists.return_value = False
        mock_store = Mock()
        mock_store.client = mock_client
        mock_store.index_name = "test_index"
        mock_get_store.return_value = mock_store
        result = await check_vectordb_status()
        assert isinstance(result, JSONResponse)
        assert result.status_code == 200
        content = json.loads(result.body.decode())
        assert content["status"] == "connected"
        assert content["index_exists"] is False

    @pytest.mark.asyncio
    @patch('app.api.aoss_routes.get_store')
    async def test_vector_status_store_unavailable(self, mock_get_store):
        mock_get_store.return_value = None
        result = await check_vectordb_status()
        assert isinstance(result, JSONResponse)
        assert result.status_code == 503
        content = json.loads(result.body.decode())
        assert content["status"] == "error"
        assert "OpenSearch store not available" in content["message"]

class TestDownloadCSVRoute:
    @pytest.mark.asyncio
    async def test_download_csv_success(self):
        # Clear any existing mappings
        domains_module.clear_file_mappings()

        temp_dir = tempfile.gettempdir()
        filename = "test_customer_schema.csv"
        csv_path = os.path.join(temp_dir, filename)
        
        with open(csv_path, 'w') as f:
            f.write("email,name,age\n")
        
        try:
            # Register the file and get a secure file ID
            file_id = domains_module.register_csv_file(filename, csv_path)
            
            result = await download_csv_file(file_id)
            assert isinstance(result, FileResponse)
            assert result.filename == filename
            assert result.media_type == "text/csv"
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)
            domains_module.clear_file_mappings()

    @pytest.mark.asyncio
    async def test_download_csv_invalid_filename(self):
        result = await download_csv_file("not-a-uuid")
        assert isinstance(result, JSONResponse)
        assert result.status_code == 400
        content = json.loads(result.body.decode())
        assert "error" in content
        assert "Invalid file ID format" in content["error"]

    @pytest.mark.asyncio
    async def test_download_csv_file_not_found(self):
        # Use a valid UUID that doesn't exist in mappings
        file_id = str(uuid.uuid4())
        result = await download_csv_file(file_id)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 404
        content = json.loads(result.body.decode())
        assert "error" in content
        assert "File not found" in content["error"]

@pytest.fixture(autouse=True)
def reset_store():
    """Reset the global _store before and after each test in this module"""
    import app.api.aoss_routes
    app.api.aoss_routes._store = None
    yield
    app.api.aoss_routes._store = None
