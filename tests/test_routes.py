"""
Unit tests for general API routes (vector status, CSV download, store helper)
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from fastapi.responses import JSONResponse, FileResponse
from app.api.routes import (
    check_vectordb_status, download_csv_file, get_store
)

class TestVectorStatusRoute:
    @pytest.mark.asyncio
    @patch('app.api.routes.get_store')
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
    @patch('app.api.routes.get_store')
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
    @patch('app.api.routes.get_store')
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
        temp_dir = tempfile.gettempdir()
        filename = "test_customer_schema.csv"
        csv_path = os.path.join(temp_dir, filename)
        with open(csv_path, 'w') as f:
            f.write("email,name,age\n")
        try:
            result = await download_csv_file(filename)
            assert isinstance(result, FileResponse)
            assert result.filename == filename
            assert result.media_type == "text/csv"
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

    @pytest.mark.asyncio
    async def test_download_csv_invalid_filename(self):
        result = await download_csv_file("../malicious.csv")
        assert isinstance(result, JSONResponse)
        assert result.status_code == 400
        content = json.loads(result.body.decode())
        assert "error" in content
        assert "Invalid filename" in content["error"]

    @pytest.mark.asyncio
    async def test_download_csv_file_not_found(self):
        result = await download_csv_file("nonexistent.csv")
        assert isinstance(result, JSONResponse)
        assert result.status_code == 404
        content = json.loads(result.body.decode())
        assert "error" in content
        assert "CSV file not found" in content["error"]

@pytest.fixture(autouse=True)
def reset_store():
    """Reset the global _store before and after each test in this module"""
    import app.api.routes
    app.api.routes._store = None
    yield
    app.api.routes._store = None
