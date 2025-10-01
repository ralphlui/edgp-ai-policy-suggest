from fastapi import APIRouter, HTTPException, Body, Depends
from fastapi.responses import JSONResponse
from app.agents.agent_runner import run_agent
from app.agents.schema_suggester import bootstrap_schema_for_domain
from app.vector_db.schema_loader import get_schema_by_domain
from app.aoss.column_store import OpenSearchColumnStore
from app.core.config import settings
from app.auth.bearer import verify_any_scope_token, UserInfo
import traceback
import logging
import time
from app.aoss.column_store import ColumnDoc
from app.embedding.embedder import embed_column_names_batched_async
from fastapi.responses import StreamingResponse
from fastapi import Body
from fastapi.responses import JSONResponse, StreamingResponse
import pandas as pd, io

logger = logging.getLogger(__name__)

# Lazy initialization of OpenSearch store to prevent startup failures
_store = None

def get_store() -> OpenSearchColumnStore:
    """Get OpenSearch store with lazy initialization and error handling"""
    global _store
    if _store is None:
        try:
            _store = OpenSearchColumnStore(index_name=settings.opensearch_index)
            logger.info(" OpenSearch store initialized successfully")
        except Exception as e:
            logger.error(f" Failed to initialize OpenSearch store: {e}")
            logger.error("This is likely due to AWS permission issues. See AWS_ADMIN_SETUP_GUIDE.md")
          
            return None
    return _store

router = APIRouter()



@router.post("/api/aips/suggest-rules")
async def suggest_rules(
    domain: str = Body(..., embed=True),
    user: UserInfo = Depends(verify_any_scope_token)
):
    try:
        # Log authenticated user information
        logger.info(f"🔐 Suggest rules request from user: {user.email} with scopes: {user.scopes}")
        
        # Try to get schema from vector database
        logger.info(f"Attempting to retrieve schema for domain: {domain}")
        
        # First, test vector DB connectivity
        vector_db_status = "unknown"
        connection_error = None
        
        try:
            schema = get_schema_by_domain(domain)
            vector_db_status = "connected"
            logger.info(f"Vector DB connection successful. Schema retrieval result for domain {domain}: {schema}")
        except Exception as db_error:
            connection_error = str(db_error)
            schema = None
            # Check if it's a connection/auth issue vs schema not found
            if "AuthorizationException" in connection_error or "RetryError" in connection_error or "ConnectionError" in connection_error:
                vector_db_status = "connection_failed"
                logger.error(f"Vector DB connection failed for domain {domain}: {connection_error}")
            else:
                vector_db_status = "accessible_but_error"
                logger.warning(f"Vector DB accessible but error occurred for domain {domain}: {connection_error}")

        if schema:
            logger.info(f"Successfully retrieved schema for domain {domain}, generating rules")
            rule_suggestions = run_agent(schema)
            return { "rule_suggestions": rule_suggestions }

        # Handle different scenarios when no schema is found
        if vector_db_status == "connection_failed":
            # Case 1: Cannot connect to vector DB at all - Don't generate LLM schema
            return JSONResponse({
                "error": "Vector database connection failed",
                "alert": {
                    "type": "error",
            
                    "message": "Cannot connect to the vector database. Please check your connection and try again.",
                    "details": "Vector database is currently unavailable. Unable to retrieve or create schemas."
                },
                "domain": domain,
                "error_type": "connection_failed"
            }, status_code=503)
        
        else:
            # Case 2: Can connect to vector DB but schema doesn't exist for this domain
            # Only now generate LLM-suggested column names (no data types)
            logger.info(f"Vector DB accessible but no schema found. Generating column name suggestions for domain: {domain}")
            suggested_schema = bootstrap_schema_for_domain(domain)
            
            # Create column names only (no data types) when domain not found in vector DB
            suggested_column_names = list(suggested_schema.keys())
            
            return JSONResponse({
                "error": "Domain not found",
                "alert": {
                    "type": "confirmation",
                    "message": f"No schema exists for domain '{domain}'. Would you like to create one using our AI suggestions?",
                    "details": f"We've generated {len(suggested_column_names)} suggested column names for this domain."
                },
                "domain": domain,
                "suggested_columns": suggested_column_names,
                "actions": {
                    "note": "Column names only suggested. Data types will be inferred from actual CSV data.",
                    "create_schema_with_csv": {
                        "description": "Use suggested column names to create schema from CSV",
                        "endpoint": "/api/aips/create/domain",
                        "method": "POST",
                        "payload": {
                            "domain": domain,
                            "columns": suggested_column_names,
                            "return_csv": True
                        }
                    },
                    "create_schema_only": {
                        "description": "Use suggested column names to create schema",
                        "endpoint": "/api/aips/create/domain",
                        "method": "POST",
                        "payload": {
                            "domain": domain,
                            "columns": suggested_column_names,
                            "return_csv": False
                        }
                    }
                },
                "next_steps": {
                    "after_creation": [
                        "Schema will be saved to vector database",
                        "Optional: Download sample CSV data",
                        "Call /api/aips/suggest-rules again to get validation rules"
                    ]
                }
            }, status_code=404)

    except Exception as e:
        # This catches any unexpected errors not handled above
        error_message = str(e)
        logger.error(f"Unexpected error in suggest_rules for domain {domain}: {error_message}")
        traceback.print_exc()
        
        return JSONResponse({
            "error": "Internal server error",
            "alert": {
             
                "message": f"An unexpected error occurred while processing domain '{domain}'",
                "details": "Please try again or contact support if the issue persists."
            },
            "domain": domain
        }, status_code=500)


@router.post("/api/aips/create/domain")
async def create_domain(
    payload: dict = Body(...),
    user: UserInfo = Depends(verify_any_scope_token)
):
    try:
        # Log authenticated user information
        logger.info(f" Create domain request from user: {user.email} with scopes: {user.scopes}")
        
        # Validate required fields
        if "domain" not in payload:
            return JSONResponse({
                "error": "Missing required field: 'domain'",
                "message": "The 'domain' field is required in the request payload."
            }, status_code=400)
        
        domain = payload["domain"].strip()
        
        # Normalize domain to lowercase for consistency
        normalized_domain = domain.lower()
        
        # Check if domain already exists (case-insensitive)
        store = get_store()
        if store is not None:
            try:
                domain_check = store.check_domain_exists_case_insensitive(domain)
                if domain_check["exists"]:
                    existing_domain = domain_check["existing_domain"]
                    existing_columns = store.get_columns_by_domain(existing_domain)
                    
                    # Determine if it's exact match or case difference
                    if existing_domain == normalized_domain:
                        message = f"Domain '{domain}' already exists"
                        note = "Domain was not created because it already exists in the database"
                    else:
                        message = f"Domain '{domain}' conflicts with existing domain '{existing_domain}' (case-insensitive)"
                        note = f"Domain creation blocked due to case-insensitive conflict. Existing domain: '{existing_domain}'. All domains are stored in lowercase."
                    
                    return JSONResponse({
                        "message": message,
                        "status": "exists", 
                        "existing_domain": existing_domain,
                        "requested_domain": domain,
                        "normalized_domain": normalized_domain,
                        "existing_columns": [col.get("column_name") for col in existing_columns],
                        "column_count": len(existing_columns),
                        "note": note,
                        "case_conflict": existing_domain != normalized_domain
                    }, status_code=409)  # 409 Conflict status for duplicate/conflicting resource
            except Exception as e:
                logger.warning(f"Failed to check if domain exists: {e}")
                # Continue with creation if we can't check (don't fail on this)
        
        # Support both formats: columns array or schema object
        if "columns" in payload:
            # New format: just column names
            columns = payload["columns"]
            if not isinstance(columns, list):
                return JSONResponse({
                    "error": "Invalid format: 'columns' must be an array",
                    "message": "The 'columns' field should be an array of column names.",
                    "expected_format": {
                        "domain": "string",
                        "columns": ["column1", "column2", "column3"],
                        "return_csv": "boolean (optional)"
                    }
                }, status_code=400)
            column_names = columns
            schema = {col: {"type": "string"} for col in columns}  # Default all to string
        elif "schema" in payload:
            # Legacy format: schema object (types are optional)
            schema = payload["schema"]
            column_names = list(schema.keys())
        else:
            return JSONResponse({
                "error": "Missing required field: 'columns' or 'schema'",
                "message": "Either 'columns' (array of names) or 'schema' (object) is required.",
                "expected_formats": {
                    "option1": {
                        "domain": "string",
                        "columns": ["column1", "column2", "column3"],
                        "return_csv": "boolean (optional)"
                    },
                    "option2": {
                        "domain": "string", 
                        "schema": {"column_name": {"type": "optional"}},
                        "return_csv": "boolean (optional)"
                    }
                }
            }, status_code=400)
        
        return_csv = payload.get("return_csv", False)  # toggle from FE
        try:
            embeddings = await embed_column_names_batched_async(column_names)
            logger.info(f"Successfully generated embeddings for {len(column_names)} columns")
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return JSONResponse({
                "error": "Embedding generation failed",
                "message": "Unable to generate embeddings for column names",
                "details": str(e),
                "domain": domain
            }, status_code=503)

        # Step 2: Upsert to vector DB (with error handling)
        storage_success = False
        storage_error = None
        
        # Get store with error handling
        store = get_store()
        if store is None:
            storage_error = "OpenSearch store not available due to configuration or permission issues"
            logger.warning(storage_error)
        else:
            try:
                docs = []
                for i, col_name in enumerate(column_names):
                    col_info = schema.get(col_name, {})
                    # Default type to 'string' if not specified
                    col_type = col_info.get("type", col_info.get("dtype", "string"))
                    
                    docs.append(ColumnDoc(
                        column_id=f"{normalized_domain}.{col_name}",
                        column_name=col_name,
                        embedding=embeddings[i],
                        sample_values=[],  # Don't store sample values in vector DB
                        metadata={
                            "domain": normalized_domain,
                            "type": col_type,
                            "pii": False,
                            "table": normalized_domain,
                            "source": "synthetic",
                            "original_domain_input": domain  # Keep track of original input for reference
                        }
                    ))
                store.upsert_columns(docs)
                logger.info(f"Successfully stored {len(docs)} columns for domain {normalized_domain} (original: {domain}) - column names only, no sample values")
                
                # Force refresh index to make new domain immediately visible
                refresh_success = store.force_refresh_index()
                if refresh_success:
                    logger.info(f"Index refreshed successfully - domain {normalized_domain} should be immediately visible")
                else:
                    logger.warning(f"Index refresh failed - domain {normalized_domain} may take a few seconds to appear in lists")
                
                storage_success = True
            except Exception as e:
                logger.warning(f"Failed to store columns in vector DB: {e}")
                
                # Enhanced error handling to unwrap nested exceptions
                root_exception = e
                exception_chain = []
                
                # Handle RetryError specifically  
                if 'RetryError' in str(e):
                    logger.error("This is a RetryError - attempting to extract the underlying exception")
                    try:
                        # Try to extract the original exception from the RetryError
                        if hasattr(e, 'last_attempt') and hasattr(e.last_attempt, 'exception'):
                            original_exception = e.last_attempt.exception()  # Call the method to get the exception
                            logger.error(f"Extracted exception from RetryError: {original_exception}")
                            logger.error(f"Exception type: {type(original_exception)}")
                            
                            # If it's a BulkIndexError, try to extract detailed error info
                            if 'BulkIndexError' in str(original_exception):
                                logger.error(f"BulkIndexError details: {original_exception}")
                                
                                # Try to access error details if available
                                if hasattr(original_exception, 'errors'):
                                    logger.error(f"Bulk errors: {original_exception.errors}")
                                    for i, error in enumerate(original_exception.errors[:3]):  # Show first 3 errors
                                        logger.error(f"Bulk error {i+1}: {error}")
                                
                                # Also try to access the error args
                                if hasattr(original_exception, 'args') and original_exception.args:
                                    logger.error(f"BulkIndexError args: {original_exception.args}")
                        
                        # Get full traceback
                        tb_str = traceback.format_exception(type(e), e, e.__traceback__)
                        logger.error(f"Full traceback: {''.join(tb_str)}")
                    except Exception as tb_error:
                        logger.error(f"Could not extract traceback or exception details: {tb_error}")
                
                # Unwrap nested exceptions to get to the root cause
                while hasattr(root_exception, '__cause__') and root_exception.__cause__:
                    exception_chain.append(f"{type(root_exception).__name__}: {str(root_exception)}")
                    root_exception = root_exception.__cause__
                
                if root_exception != e and exception_chain:
                    exception_chain.append(f"{type(root_exception).__name__}: {str(root_exception)}")
                    logger.error(f"Exception chain: {' -> '.join(exception_chain)}")
                
                # If it's a BulkIndexError, try to extract detailed error info
                if 'BulkIndexError' in str(root_exception):
                    logger.error(f"BulkIndexError details: {root_exception}")
                    
                    # Try to access error details if available
                    if hasattr(root_exception, 'errors'):
                        logger.error(f"Bulk errors: {root_exception.errors}")
                        for i, error in enumerate(root_exception.errors[:3]):  # Show first 3 errors
                            logger.error(f"Bulk error {i+1}: {error}")
                
                storage_error = str(e)
                # Continue with response generation even if storage fails

        # Step 3: Return based on toggle
        if return_csv:
            # Generate sample data if not provided in schema
            data = {}
            for col, info in schema.items():
                if "sample_values" in info and info["sample_values"]:
                    # Use provided sample values
                    data[col] = info["sample_values"][:5]
                else:
                    # Generate basic sample values based on dtype
                    dtype = info.get("dtype", "string")
                    if dtype == "integer":
                        data[col] = [1, 2, 3, 4, 5]
                    elif dtype == "float":
                        data[col] = [1.0, 2.5, 3.0, 4.5, 5.0]
                    elif dtype == "date":
                        data[col] = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
                    else:  # string or other
                        data[col] = [f"Sample_{col}_1", f"Sample_{col}_2", f"Sample_{col}_3", f"Sample_{col}_4", f"Sample_{col}_5"]
            
            df = pd.DataFrame(data)
            buffer = io.StringIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)

            # Add warning header if storage failed
            headers = {"Content-Disposition": f"attachment; filename={normalized_domain}_schema.csv"}
            if not storage_success:
                headers["X-Storage-Warning"] = "Vector database storage failed - schema not saved"

            return StreamingResponse(
                buffer,
                media_type="text/csv",
                headers=headers
            )

        # JSON response with storage status
        response_data = {"message": f"Schema for domain '{domain}' processed successfully and stored as '{normalized_domain}'."}
        
        if storage_success:
            response_data["storage_status"] = "saved"
            response_data["stored_domain"] = normalized_domain
            response_data["original_domain"] = domain
            
            # Verify domain is immediately visible (for debugging)
            try:
                immediate_domains = store.get_all_domains_realtime(force_refresh=True)
                response_data["domain_immediately_visible"] = normalized_domain in immediate_domains
                response_data["available_domains"] = immediate_domains
                if normalized_domain in immediate_domains:
                    response_data["note"] = "Domain created and immediately visible in domain list"
                else:
                    response_data["note"] = "Domain created but may take a few seconds to appear in domain list due to indexing delays"
            except Exception as e:
                logger.warning(f"Could not verify immediate domain visibility: {e}")
                response_data["note"] = "Domain created successfully"
        else:
            response_data["storage_status"] = "failed"
            response_data["storage_error"] = storage_error
            response_data["note"] = "Schema was processed but not saved to vector database"
        
        return JSONResponse(response_data)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)



@router.get("/api/aips/domains")
async def get_domains(
    user: UserInfo = Depends(verify_any_scope_token)
):
    """Get all available domains in the specified format."""
    try:
        # Log authenticated user information
        logger.info(f" Get domains request from user: {user.email} with scopes: {user.scopes}")
        
        store = get_store()
        if store is None:
            return JSONResponse({
                "success": False,
                "message": "OpenSearch store not available",
                "totalRecord": 0,
                "data": []
            }, status_code=503)
        
        # Get all unique domains from the vector database (with real-time refresh)
        domains = store.get_all_domains_realtime(force_refresh=True)
        
        return JSONResponse({
            "success": True,
            "message": f"Successfully retrieved all domain{'s' if len(domains) != 1 else ''}.",
            "totalRecord": len(domains),
            "data": domains,
            "note": "Real-time domain list with forced index refresh"
        })
        
    except Exception as e:
        logger.error(f"Error retrieving domains: {e}")
        return JSONResponse({
            "success": False,
            "message": f"Failed to retrieve domains: {str(e)}",
            "totalRecord": 0,
            "data": []
        }, status_code=500)



@router.get("/api/aips/vector/status")
async def check_vectordb_status():
    """Check vector database connection and index status."""
    try:
        store = get_store()
        if store is None:
            return JSONResponse({
                "status": "error",
                "message": "OpenSearch store not available",
                "connection": "failed"
            }, status_code=503)
        
        # Test connection and get index info
        client = store.client
        index_name = store.index_name
        
        # Check if index exists
        index_exists = client.indices.exists(index=index_name)
        
        result = {
            "status": "connected",
            "index_name": index_name,
            "index_exists": index_exists,
            "connection": "success"
        }
        
        if index_exists:
            # Get index stats
            try:
                stats = client.indices.stats(index=index_name)
                result["document_count"] = stats["indices"][index_name]["total"]["docs"]["count"]
                result["index_size"] = stats["indices"][index_name]["total"]["store"]["size_in_bytes"]
            except Exception as stats_error:
                logger.warning(f"Could not get index stats: {stats_error}")
                result["document_count"] = "unknown"
                result["index_size"] = "unknown"
                result["stats_error"] = str(stats_error)
        else:
            result["document_count"] = 0
            result["index_size"] = 0
            result["note"] = "Index does not exist yet. It will be created when first data is added."
        
        return JSONResponse(result)
        
    except Exception as e:
        return JSONResponse({
            "status": "error", 
            "message": str(e),
            "connection": "failed"
        }, status_code=500)



@router.get("/api/aips/domains-schema")
async def list_domains_in_vectordb():
    """List all domains stored in the vector database."""
    try:
        store = get_store()
        if store is None:
            return JSONResponse({
                "error": "OpenSearch store not available"
            }, status_code=503)
        
        # Check if index exists first
        if not store.client.indices.exists(index=store.index_name):
            return JSONResponse({
                "total_domains": 0,
                "total_columns": 0,
                "domains": {},
                "message": f"Index {store.index_name} does not exist yet"
            })
        
        # Query all documents and group by domain
        query = {
            "query": {"match_all": {}},
            "size": 1000,
            "_source": ["metadata.domain", "column_name", "metadata.type"]
        }
        
        response = store.client.search(index=store.index_name, body=query)
        
        domains = {}
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            domain = source.get("metadata", {}).get("domain", "unknown")
            column_name = source.get("column_name", "unknown")
            col_type = source.get("metadata", {}).get("type", "string")
            
            if domain not in domains:
                domains[domain] = []
            
            domains[domain].append({
                "column": column_name,
                "type": col_type
            })
        
        return JSONResponse({
            "total_domains": len(domains),
            "total_columns": response["hits"]["total"]["value"],
            "domains": domains
        })
        
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "message": "Failed to retrieve domains from vector database"
        }, status_code=500)


@router.get("/api/aips/domain/{domain_name}")
async def get_domain_from_vectordb(domain_name: str):
    """Get specific domain details from vector database."""
    try:
        store = get_store()
        if store is None:
            return JSONResponse({
                "error": "OpenSearch store not available"
            }, status_code=503)
        
        # Check if index exists first
        if not store.client.indices.exists(index=store.index_name):
            return JSONResponse({
                "domain": domain_name,
                "found": False,
                "message": f"Index {store.index_name} does not exist yet"
            }, status_code=404)
        
        # Query for specific domain
        query = {
            "query": {
                "term": {"metadata.domain": domain_name}
            },
            "size": 100,
            "_source": ["column_name", "metadata", "sample_values"]
        }
        
        response = store.client.search(index=store.index_name, body=query)
        
        if response["hits"]["total"]["value"] == 0:
            return JSONResponse({
                "domain": domain_name,
                "found": False,
                "message": "Domain not found in vector database"
            }, status_code=404)
        
        columns = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            columns.append({
                "column_name": source.get("column_name", "unknown"),
                "type": source.get("metadata", {}).get("type", "string"),
                "sample_values": source.get("sample_values", []),
                "metadata": source.get("metadata", {})
            })
        
        return JSONResponse({
            "domain": domain_name,
            "found": True,
            "column_count": len(columns),
            "columns": columns
        })
        
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "message": f"Failed to retrieve domain '{domain_name}' from vector database"
        }, status_code=500)

