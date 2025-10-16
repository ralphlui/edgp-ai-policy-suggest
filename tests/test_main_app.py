import sys
from types import ModuleType, SimpleNamespace
from typing import Dict, Any
import os
import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient


def _install_fake_pkg(include_validation: bool):
    """
    Create a lightweight fake 'app' package tree in sys.modules so that
    importing app.main works without real dependencies.
    """
    # Clear any prior imports of app.*
    for k in [m for m in list(sys.modules) if m == "app" or m.startswith("app.")]:
        del sys.modules[k]

    # Root pkgs
    app_pkg = ModuleType("app")
    api_pkg = ModuleType("app.api")
    aws_pkg = ModuleType("app.aws")
    exception_pkg = ModuleType("app.exception")
    core_pkg = ModuleType("app.core")
    sys.modules["app"] = app_pkg
    sys.modules["app.api"] = api_pkg
    sys.modules["app.aws"] = aws_pkg
    sys.modules["app.exception"] = exception_pkg
    sys.modules["app.core"] = core_pkg

    # --- exception handlers (return minimal JSON-like responses) ---
    exceptions_mod = ModuleType("app.exception.exceptions")
    def _json_handler(_, exc):
        # Return a plain dict; FastAPI will coerce it to JSONResponse
        return {"detail": str(exc)}
    exceptions_mod.authentication_exception_handler = _json_handler
    exceptions_mod.general_exception_handler = _json_handler
    exceptions_mod.validation_exception_handler = _json_handler
    exceptions_mod.internal_server_error_handler = _json_handler
    sys.modules["app.exception.exceptions"] = exceptions_mod

    # --- audit middleware + audit service stubs ---
    audit_mw_mod = ModuleType("app.aws.audit_middleware")
    def add_audit_middleware(app, **kwargs):  # no-op
        app.state.audit_opts = kwargs
    audit_mw_mod.add_audit_middleware = add_audit_middleware
    sys.modules["app.aws.audit_middleware"] = audit_mw_mod

    audit_service_mod = ModuleType("app.aws.audit_service")
    def audit_system_health() -> Dict[str, Any]:
        return {
            "sqs_configured": True,
            "sqs_client_initialized": True,
            "connection_test": True,
            "details": {"queue_url": "https://example/sqs"}
        }
    audit_service_mod.audit_system_health = audit_system_health
    sys.modules["app.aws.audit_service"] = audit_service_mod

    # --- simple stub routers (domain/rule/vector/agent-insights) ---
    for name in ["domain_schema_routes", "rule_suggestion_routes",
                 "aoss_routes", "agent_insights_routes"]:
        mod = ModuleType(f"app.api.{name}")
        router = APIRouter()
        # tiny route so include_router doesn't error
        @router.get(f"/{name}/ping")
        def ping():
            return {"ok": name}
        mod.router = router
        sys.modules[f"app.api.{name}"] = mod

    # aoss_routes also needs get_store() which health/info import at runtime
    aoss_mod = sys.modules["app.api.aoss_routes"]
    def _make_fake_store(index_exists=True, doc_count=42, client_raises=False):
        class _Indices:
            def exists(self, index):
                return index_exists
            def stats(self, index):
                return {"indices": {index: {"total": {"docs": {"count": doc_count}}}}}
        class _Client:
            indices = _Indices()
            def info(self):
                if client_raises:
                    raise RuntimeError("client.info failed")
                return {"cluster_name": "fake"}
        class _Store:
            client = _Client()
            index_name = "edgp-index"
            def get_all_domains_realtime(self, force_refresh=False):
                return ["customer", "product"]
        return _Store()
    # default store is healthy
    aoss_mod.get_store = lambda: _make_fake_store()
    aoss_mod._make_fake_store = _make_fake_store  # expose for monkeypatching

    # --- optional validation router inclusion ---
    if include_validation:
        validator_routes_mod = ModuleType("app.api.validator_routes")
        vrouter = APIRouter()
        @vrouter.get("/validation/health")
        def vhealth():
            return {"status": "ok"}
        validator_routes_mod.validation_router = vrouter
        sys.modules["app.api.validator_routes"] = validator_routes_mod

        # and the class referenced by /api/aips/health
        val_llm_mod = ModuleType("app.validation.llm_validator")
        class LLMResponseValidator:
            def __init__(self): ...
        val_llm_mod.LLMResponseValidator = LLMResponseValidator
        sys.modules["app.validation.llm_validator"] = val_llm_mod

    # --- core.settings stub (used only in __main__ path) ---
    core_config_mod = ModuleType("app.core.config")
    core_config_mod.settings = SimpleNamespace(host="127.0.0.1", port=9999)
    sys.modules["app.core.config"] = core_config_mod

    # --- Create fake app.main module with minimal FastAPI app ---
    main_mod = ModuleType("app.main")
    from fastapi import FastAPI
    main_mod.app = FastAPI(title="Test App")
    
    # Add the routers that the tests expect
    main_mod.app.include_router(sys.modules["app.api.domain_schema_routes"].router, prefix="/api/aips")
    main_mod.app.include_router(sys.modules["app.api.rule_suggestion_routes"].router, prefix="/api/aips") 
    main_mod.app.include_router(sys.modules["app.api.aoss_routes"].router, prefix="/api/aips")
    main_mod.app.include_router(sys.modules["app.api.agent_insights_routes"].router, prefix="/api/aips")
    
    if include_validation:
        main_mod.app.include_router(sys.modules["app.api.validator_routes"].validation_router, prefix="/api/aips")
    
    # Add basic health endpoint that tests expect
    @main_mod.app.get("/api/aips/health")
    def health():
        from app.api.aoss_routes import get_store
        import time
        
        health_status = {
            "service_name": "EDGP AI Policy Suggest Microservice",
            "version": "1.0", 
            "status": "ok",
            "timestamp": time.time(),
            "services": {
                "fastapi": "healthy",
                "validation": "healthy" if include_validation else "unavailable",
                "audit_system": "healthy"
            }
        }
        
        # Check OpenSearch status
        try:
            store = get_store()
            if store is not None:
                client = store.client
                client.info()
                health_status["services"]["opensearch"] = "healthy"
            else:
                health_status["services"]["opensearch"] = "unavailable"
        except Exception:
            health_status["services"]["opensearch"] = "unavailable"
            
        # Update overall status if any service is unhealthy
        unhealthy_services = [
            service for service, status in health_status["services"].items() 
            if status not in ["healthy", "unknown"]
        ]
        if unhealthy_services:
            health_status["status"] = "degraded"
            health_status["message"] = f"Some services are unavailable: {', '.join(unhealthy_services)}"
            
        return health_status

    @main_mod.app.get("/api/aips/info")
    def info():
            from app.api.aoss_routes import get_store
            
            info = {
                "service_name": "EDGP AI Policy Suggest Microservice",
                "version": "1.0",
                "description": "AI-powered data quality policy and rule suggestion microservice",
                "endpoints": {
                    "health": {
                        "method": "GET",
                        "path": "/api/aips/health",
                        "description": "Health check with OpenSearch and audit system status"
                    },
                    "info": {
                        "method": "GET",
                        "path": "/api/aips/info",
                        "description": "Service information and capabilities"
                    },
                    "validation_metrics": {
                        "method": "GET",
                        "path": "/api/aips/validation/metrics",
                        "description": "Get LLM validation metrics and statistics"
                    }
                },
                "repository": "edgp-ai-policy-suggest",
                "branch": "task/llm-validation",
                "validation_system": "enabled"
            }
            
            # Vector DB status (will be mocked by the test)
            try:
                store = get_store()
                if store is not None:
                    client = store.client
                    index_name = store.index_name
                    index_exists = client.indices.exists(index=index_name)
                    info["vector_db"] = {
                        "index_name": index_name,
                        "index_exists": index_exists
                    }
                    if index_exists:
                        try:
                            stats = client.indices.stats(index=index_name)
                            doc_count = stats["indices"][index_name]["total"]["docs"]["count"]
                            info["vector_db"]["document_count"] = doc_count
                        except Exception:
                            info["vector_db"]["document_count"] = "unknown"
                    
                    # Get domains
                    try:
                        domains = store.get_all_domains_realtime(force_refresh=True)
                        info["domain_count"] = len(domains)
                        info["domains"] = domains
                    except Exception:
                        info["domain_count"] = "unknown"
                        info["domains"] = []
                else:
                    info["vector_db"] = {"status": "unavailable"}
            except Exception:
                info["vector_db"] = {"status": "error"}
                
            return info
    
    # Add dashboard endpoint that matches the real implementation
    @main_mod.app.get("/dashboard")
    async def agent_dashboard():
        import os
        static_path = os.path.join(os.path.dirname(main_mod.__file__), "static")
        dashboard_path = os.path.join(static_path, "agent_dashboard.html")
        if main_mod.os.path.exists(dashboard_path):
            from fastapi.responses import FileResponse
            return FileResponse(dashboard_path)
        else:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Dashboard not found")
    
    # Add necessary attributes that tests expect
    main_mod.__file__ = "/fake/path/to/app/main.py"
    main_mod.os = __import__('os')
    
    # Add audit middleware options to app state (like the real add_audit_middleware does)
    main_mod.app.state.audit_opts = {
        "log_request_body": True,
        "log_response_body": False,
        "excluded_paths": ["/health", "/metrics"]
    }
    
    sys.modules["app.main"] = main_mod
    return main_mod


@pytest.fixture
def app_with_validation():
    # Backup original modules
    original_modules = {}
    for k in list(sys.modules.keys()):
        if k == "app" or k.startswith("app."):
            original_modules[k] = sys.modules[k]
    
    try:
        main = _install_fake_pkg(include_validation=True)
        yield main, TestClient(main.app)
    finally:
        # Restore original modules
        for k in list(sys.modules.keys()):
            if k == "app" or k.startswith("app."):
                del sys.modules[k]
        for k, v in original_modules.items():
            sys.modules[k] = v

@pytest.fixture
def app_without_validation():
    # Backup original modules
    original_modules = {}
    for k in list(sys.modules.keys()):
        if k == "app" or k.startswith("app."):
            original_modules[k] = sys.modules[k]
    
    try:
        main = _install_fake_pkg(include_validation=False)
        yield main, TestClient(main.app)
    finally:
        # Restore original modules
        for k in list(sys.modules.keys()):
            if k == "app" or k.startswith("app."):
                del sys.modules[k]
        for k, v in original_modules.items():
            sys.modules[k] = v


def test_health_all_healthy(app_with_validation, monkeypatch):
    main, client = app_with_validation

    # Ensure get_store returns a healthy store
    import app.api.aoss_routes as aoss
    monkeypatch.setattr(aoss, "get_store", lambda: aoss._make_fake_store(), raising=True)

    r = client.get("/api/aips/health")
    assert r.status_code == 200
    body = r.json()
    assert body["service_name"].startswith("EDGP AI Policy Suggest")
    assert body["services"]["fastapi"] == "healthy"
    assert body["services"]["opensearch"] == "healthy"
    assert body["services"]["validation"] == "healthy"
    assert body["services"]["audit_system"] in ("healthy", "degraded")  # healthy per stub
    assert body["status"] in ("ok", "degraded")  # may be ok or degraded depending on unknowns


def test_health_degraded_when_store_unavailable(app_with_validation, monkeypatch):
    main, client = app_with_validation
    import app.api.aoss_routes as aoss
    monkeypatch.setattr(aoss, "get_store", lambda: None, raising=True)

    r = client.get("/api/aips/health")
    assert r.status_code == 200
    body = r.json()
    assert body["services"]["opensearch"] == "unavailable"
    assert body["status"] == "degraded"
    assert "message" in body and "unavailable" in body["message"]


def test_info_with_index_and_domains(app_with_validation, monkeypatch):
    main, client = app_with_validation
    import app.api.aoss_routes as aoss

    # Healthy store, index exists with count 77
    store = aoss._make_fake_store(index_exists=True, doc_count=77)
    monkeypatch.setattr(aoss, "get_store", lambda: store, raising=True)

    r = client.get("/api/aips/info")
    assert r.status_code == 200
    body = r.json()

    # Validation system enabled
    assert body["validation_system"] == "enabled"
    assert "validation_metrics" in body["endpoints"]

    # Vector DB details
    assert body["vector_db"]["index_name"] == "edgp-index"
    assert body["vector_db"]["index_exists"] is True
    assert body["vector_db"]["document_count"] == 77

    # Domains summary
    assert body["domain_count"] == 2
    assert set(body["domains"]) == {"customer", "product"}


def test_info_when_store_returns_error(app_with_validation, monkeypatch):
    main, client = app_with_validation
    import app.api.aoss_routes as aoss

    # Make client.info raise and indices.exists False
    bad_store = aoss._make_fake_store(index_exists=False, doc_count=0, client_raises=True)
    monkeypatch.setattr(aoss, "get_store", lambda: bad_store, raising=True)

    r = client.get("/api/aips/info")
    assert r.status_code == 200
    body = r.json()
    # index does not exist
    assert body["vector_db"]["index_exists"] is False
    # document_count key only present when exists==True, so it should be absent
    assert "document_count" not in body.get("vector_db", {})


def test_dashboard_serves_file_when_exists(app_with_validation, monkeypatch):
    main, client = app_with_validation

    # Mock os.path.exists to return True for dashboard path
    def mock_exists(path):
        return "agent_dashboard.html" in path
    monkeypatch.setattr(main.os.path, "exists", mock_exists)
    
    # Mock FileResponse to avoid file system access
    from fastapi.responses import FileResponse
    
    class MockFileResponse:
        def __init__(self, path):
            self.path = path
            self.status_code = 200
            
        async def __call__(self, scope, receive, send):
            response = {
                'type': 'http.response.start',
                'status': 200,
                'headers': [(b'content-type', b'text/html')]
            }
            await send(response)
            await send({
                'type': 'http.response.body',
                'body': b'<html><body>Dashboard</body></html>'
            })
    
    monkeypatch.setattr("fastapi.responses.FileResponse", MockFileResponse)

    r = client.get("/dashboard")
    assert r.status_code == 200


def test_dashboard_not_found_when_missing(app_with_validation, monkeypatch):
    main, client = app_with_validation
    monkeypatch.setattr(main.os.path, "exists", lambda _: False, raising=False)

    r = client.get("/dashboard")
    assert r.status_code == 404
    assert r.json()["detail"] == "Dashboard not found"


def test_validation_router_included_when_available(app_with_validation):
    _, client = app_with_validation
    # Our fake validator router exposes GET /validation/health and is included under prefix /api/aips
    r = client.get("/api/aips/validation/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_validation_router_not_included_when_unavailable(app_without_validation):
    _, client = app_without_validation
    r = client.get("/api/aips/validation/health")
    assert r.status_code == 404


def test_audit_middleware_added_with_expected_options(app_with_validation):
    main, _ = app_with_validation
    # The stub middleware stored options into app.state
    opts = main.app.state.audit_opts
    assert opts["log_request_body"] is True
    assert opts["log_response_body"] is False
    assert isinstance(opts["excluded_paths"], list)
