from fastapi import APIRouter, Body, Depends, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from app.auth.authentication import verify_any_scope_token, UserInfo
from app.aoss.column_store import OpenSearchColumnStore, ColumnDoc
from app.core.config import settings
from app.embedding.embedder import embed_column_names_batched_async
import traceback
import logging, time, io, csv

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/aips/domains", tags=["domain-schema"])

# --- Domain & Schema Endpoints ---

# Helper functions
_store = None
def get_store() -> OpenSearchColumnStore:
    global _store
    if _store is None:
        try:
            _store = OpenSearchColumnStore(index_name=settings.opensearch_index)
            logger.info(" OpenSearch store initialized successfully")
        except Exception as e:
            logger.error(f" Failed to initialize OpenSearch store: {e}")
            return None
    return _store

def _is_similar_column_name(name1: str, name2: str) -> bool:
    """Check if two column names are similar to prevent near-duplicates."""
    name1 = name1.lower()
    name2 = name2.lower()
    
    # Direct substring check
    if name1 in name2 or name2 in name1:
        return True
    
    # Remove common prefixes/suffixes and compare
    common_prefixes = ['is_', 'has_', 'was_', 'user_', 'customer_', 'account_']
    common_suffixes = ['_id', '_name', '_date', '_time', '_timestamp', '_count', '_total']
    
    for prefix in common_prefixes:
        name1 = name1[len(prefix):] if name1.startswith(prefix) else name1
        name2 = name2[len(prefix):] if name2.startswith(prefix) else name2
    
    for suffix in common_suffixes:
        name1 = name1[:-len(suffix)] if name1.endswith(suffix) else name1
        name2 = name2[:-len(suffix)] if name2.endswith(suffix) else name2
    
    # If the core parts are very similar
    return name1 == name2 or \
           (len(name1) > 3 and len(name2) > 3 and \
            (name1 in name2 or name2 in name1))

@router.post("/create")
async def create_domain(
    request: Request,
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
                        "case_conflict": existing_domain != normalized_domain,
                        "actions": {
                            "extend-schema": {
                                "description": "Add new columns to existing domain",
                                "endpoint": "/api/aips/domains/extend-schema",
                                "method": "POST",
                                "payload": {
                                    "domain": existing_domain,
                                    "new_columns": ["new_column1", "new_column2"],
                                    "return_csv": True
                                }
                            },
                            "suggest_extensions": {
                                "description": "Get AI suggestions for additional columns",
                                "endpoint": "/api/aips/domains/suggest-extend-schema",
                                "method": "POST",
                                "payload": {
                                    "domain": existing_domain,
                                    "style": "standard",
                                    "exclude_existing": True
                                }
                            },
                            "view_domain": {
                                "description": "View complete domain details",
                                "endpoint": f"/api/aips/domains/{existing_domain}",
                                "method": "GET"
                            }
                        }
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

        # Step 3: Enhanced return logic with agentic rule suggestions
        if return_csv:
            # AGENTIC ENHANCEMENT: Generate rule suggestions for CSV response
            rule_suggestions = []
            
            if storage_success:
                logger.info(f" Generating rule suggestions for CSV download of domain: {normalized_domain}")
                try:
                    # Convert our schema format to the format expected by run_agent
                    agent_schema = {}
                    for col_name in column_names:
                        col_type = schema.get(col_name, {}).get("type", "string")
                        agent_schema[col_name] = {"type": col_type}
                    
                    # Run the agent directly with the schema we just created
                    from app.agents.agent_runner import run_agent
                    rule_suggestions = run_agent(agent_schema)
                    
                    logger.info(f" Generated {len(rule_suggestions)} rule suggestions for CSV download of domain: {normalized_domain}")
                    
                except Exception as agentic_error:
                    logger.error(f" Rule generation failed for CSV download: {agentic_error}")
                    rule_suggestions = []
            
            # Store CSV temporarily and return JSON with download info + rule suggestions
            import tempfile
            import os
            
            # Create a temporary file for CSV
            temp_dir = tempfile.gettempdir()
            csv_filename = f"{normalized_domain}_schema_{int(__import__('time').time())}.csv"
            csv_path = os.path.join(temp_dir, csv_filename)
            
            # Create CSV with just column headers (no sample data) using built-in csv module
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(column_names)  # Write header row
            
            # Return JSON response with rule suggestions AND CSV download info
            # Build response with appropriate rule messaging
            if rule_suggestions:
                message = f"Schema for domain '{normalized_domain}' created successfully with CSV template and {len(rule_suggestions)} validation rules."
                rules_available = True
            else:
                message = f"Schema for domain '{normalized_domain}' created successfully with CSV template. No validation rules available for this schema."
                rules_available = False
            
            response_data = {
                "status": "success",
                "domain": normalized_domain,
                "message": message,
                "columns_created": len(column_names),
                "csv_download": {
                    "available": True,
                    "filename": csv_filename,
                    "download_url": f"{request.base_url}api/aips/download-csv/{csv_filename}",
                    "type": "template",
                    "description": "CSV file with column headers.",
                    "columns": column_names
                },
                "rules_available": rules_available,
                "rule_suggestions": rule_suggestions,
                "total_rules": len(rule_suggestions)
            }
            
            if not storage_success:
                response_data["storage_error"] = storage_error
                response_data["note"] = "Schema processed and CSV generated, but not saved to vector database"
            # CSV generation completed
            
            return JSONResponse(response_data)

        # JSON response with user-friendly structure
        response_data = {
            "status": "success",
            "domain": normalized_domain,
            "message": f"Schema for domain '{normalized_domain}' created successfully.",
            "columns_created": len(column_names)
        }
        
        if storage_success:
            
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
            
            #  AGENTIC : Automatically generate rule suggestions after schema creation
            logger.info(f" Agentic workflow: Automatically generating rule suggestions for newly created domain: {normalized_domain}")
            try:
                # Wait a brief moment for indexing to complete
                import asyncio
                await asyncio.sleep(0.5)
                
                # PERFORMANCE OPTIMIZATION: Use the schema we just created directly
                # instead of waiting for OpenSearch indexing
                logger.info(f" Agentic workflow: Generating rules immediately using created schema for domain: {normalized_domain}")
                
                # Convert our schema format to the format expected by run_agent
                agent_schema = {}
                for col_name in column_names:
                    col_type = schema.get(col_name, {}).get("type", "string")
                    agent_schema[col_name] = {"type": col_type}
                
                # Run the agent directly with the schema we just created
                from app.agents.agent_runner import run_agent
                rule_suggestions = run_agent(agent_schema)
                
                if rule_suggestions:
                    response_data["rules_available"] = True
                    response_data["rule_suggestions"] = rule_suggestions
                    response_data["total_rules"] = len(rule_suggestions)
                    response_data["message"] += f" {len(rule_suggestions)} validation rules automatically generated."
                else:
                    response_data["rules_available"] = False
                    response_data["rule_suggestions"] = []
                    response_data["total_rules"] = 0
                    response_data["message"] += " No validation rules available for this schema."
                
                logger.info(f" Agentic workflow: Successfully generated {len(rule_suggestions)} rule suggestions immediately for domain: {normalized_domain}")
                
            except Exception as agentic_error:
                logger.error(f" Rule generation failed for domain {normalized_domain}: {agentic_error}")
                response_data["rules_available"] = False
                response_data["rule_suggestions"] = []
                response_data["total_rules"] = 0
                response_data["rule_generation_error"] = str(agentic_error)
                response_data["message"] += " Automatic rule generation failed. Call /api/aips/rules/suggest to generate validation rules manually."
                
        else:
            response_data["status"] = "partial_success"
            response_data["storage_error"] = storage_error
            response_data["message"] += " However, schema could not be saved to database."
            response_data["rules_available"] = False
            response_data["rule_suggestions"] = []
            response_data["total_rules"] = 0
        
        return JSONResponse(response_data)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)



@router.get("")
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



@router.get("/verify/{domain_name}")
async def verify_domain_exists(
    domain_name: str,
    user: UserInfo = Depends(verify_any_scope_token)
):
    """Verify if a specific domain exists with real-time refresh - useful right after domain creation."""
    try:
        # Log authenticated user information
        logger.info(f" Verify domain request from user: {user.email} for domain: {domain_name}")
        
        store = get_store()
        if store is None:
            return JSONResponse({
                "success": False,
                "domain": domain_name,
                "exists": False,
                "message": "OpenSearch store not available"
            }, status_code=503)
        
        # Normalize domain name for consistent checking
        normalized_domain = domain_name.lower().strip()
        
        # Force refresh and get all domains
        all_domains = store.get_all_domains_realtime(force_refresh=True)
        domain_exists = normalized_domain in all_domains
        
        return JSONResponse({
            "success": True,
            "domain": domain_name,
            "normalized_domain": normalized_domain,
            "exists": domain_exists,
            "message": f"Domain '{domain_name}' {'exists' if domain_exists else 'does not exist'}",
            "all_domains": all_domains,
            "total_domains": len(all_domains),
            "note": "Real-time check with forced index refresh"
        })
        
    except Exception as e:
        logger.error(f"Error verifying domain {domain_name}: {e}")
        return JSONResponse({
            "success": False,
            "domain": domain_name,
            "exists": False,
            "message": f"Failed to verify domain: {str(e)}"
        }, status_code=500)
    

@router.get("/schema")
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


@router.get("/{domain_name}")
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


@router.get("/download-csv/{filename}")
async def download_csv_file(filename: str):
    """Download CSV file generated during domain creation."""
    try:
        import tempfile
        import os
        from fastapi.responses import FileResponse
        
        # Security check - only allow specific filename pattern
        if not filename.endswith('.csv') or '..' in filename or '/' in filename:
            return JSONResponse({
                "error": "Invalid filename"
            }, status_code=400)
        
        temp_dir = tempfile.gettempdir()
        csv_path = os.path.join(temp_dir, filename)
        
        if not os.path.exists(csv_path):
            return JSONResponse({
                "error": "CSV file not found or expired",
                "message": "The CSV file may have been cleaned up. Please regenerate the domain."
            }, status_code=404)
        
        # Return the file for download
        return FileResponse(
            csv_path,
            media_type="text/csv",
            filename=filename
        )
        
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "message": "Failed to download CSV file"
        }, status_code=500)
    

@router.post("/suggest-schema")
async def regenerate_suggestions(request: Request):
    """
    AI-powered domain schema suggestions. User only needs to provide 'domain'.
    LLM will generate business description and preferences automatically.
    Will avoid suggesting columns that already exist in the domain.
    """
    try:
        body = await request.json()
        domain = body.get("domain")
        if not domain:
            return JSONResponse({
                "error": "Missing required field: 'domain'"
            }, status_code=400)

        # Get existing columns if domain exists
        store = get_store()
        existing_columns = []
        if store:
            try:
                domain_check = store.check_domain_exists_case_insensitive(domain)
                if domain_check["exists"]:
                    existing_domain = domain_check["existing_domain"]
                    existing_columns_data = store.get_columns_by_domain(existing_domain)
                    existing_columns = [col.get("column_name") for col in existing_columns_data]
                    logger.info(f"Found existing domain '{existing_domain}' with {len(existing_columns)} columns")
            except Exception as e:
                logger.warning(f"Failed to check existing columns: {e}")

        # Enhanced business description with context about existing columns
        if existing_columns:
            business_description = f"""Generate additional schema columns for the business domain '{domain}'.
Existing columns: {', '.join(existing_columns)}
Suggest NEW columns that complement the existing ones while avoiding duplication.
Focus on identifying missing business dimensions and valuable extensions to the current schema."""
        else:
            business_description = f"Generate a schema for the business domain '{domain}'. Suggest columns relevant to this domain."

        # Include existing columns in preferences to avoid duplicates
        user_preferences = {
            "style": "standard",
            "column_count": 8,
            "iteration": 1,
            "exclude_columns": existing_columns  # This ensures we don't suggest existing columns
        }

        from app.agents.schema_suggester import SchemaSuggesterEnhanced
        suggester = SchemaSuggesterEnhanced()
        enhanced_schema = await suggester.bootstrap_schema_with_preferences(
            business_description=business_description,
            user_preferences=user_preferences
        )

        # Extract column names from enhanced_schema
        suggested_columns = [col.get("column_name") for col in enhanced_schema.get("columns", [])]

        # Determine if domain exists and prepare appropriate actions
        domain_exists = False
        existing_domain_name = domain
        if store:
            try:
                domain_check = store.check_domain_exists_case_insensitive(domain)
                if domain_check["exists"]:
                    domain_exists = True
                    existing_domain_name = domain_check["existing_domain"]
            except Exception as e:
                logger.warning(f"Failed to check domain existence: {e}")

        if domain_exists:
            # Domain exists - suggest extend-schema actions
            return JSONResponse({
                "domain": existing_domain_name,
                "suggested_columns": suggested_columns,
                "domain_status": "exists",
                "actions": {
                    "note": "Additional columns suggested for existing domain. These will extend the current schema.",
                    "extend_schema_with_csv": {
                        "description": "Extend existing schema with suggested columns and generate CSV",
                        "endpoint": "/api/aips/domains/extend-schema",
                        "method": "PUT",
                        "payload": {
                            "domain": existing_domain_name,
                            "columns": suggested_columns,
                            "extension_preferences": {
                                "column_count": len(suggested_columns),
                                "style": "standard",
                                "focus_area": "business enhancement"
                            },
                            "return_csv": True
                        }
                    },
                    "extend_schema_only": {
                        "description": "Extend existing schema with suggested columns",
                        "endpoint": "/api/aips/domains/extend-schema",
                        "method": "PUT",
                        "payload": {
                            "domain": existing_domain_name,
                            "columns": suggested_columns,
                            "extension_preferences": {
                                "column_count": len(suggested_columns),
                                "style": "standard",
                                "focus_area": "business enhancement"
                            },
                            "return_csv": False
                        }
                    }
                },
                "next_steps": {
                    "after_confirmation": [
                        "New columns will be added to existing schema",
                        "Optional: Download updated CSV template",
                        "Call /api/aips/domains/extend-schema to extend the domain"
                    ]
                }
            })
        else:
            # New domain - suggest create actions
            return JSONResponse({
                "domain": domain,
                "suggested_columns": suggested_columns,
                "domain_status": "new",
                "actions": {
                    "note": "Column names only suggested. Data types will be inferred from actual CSV data.",
                    "create_schema_with_csv": {
                        "description": "Use suggested column names to create schema from CSV",
                        "endpoint": "/api/aips/domains/create",
                        "method": "POST",
                        "payload": {
                            "domain": domain,
                            "columns": suggested_columns,
                            "return_csv": True
                        }
                    },
                    "create_schema_only": {
                        "description": "Use suggested column names to create schema",
                        "endpoint": "/api/aips/domains/create",
                        "method": "POST",
                        "payload": {
                            "domain": domain,
                            "columns": suggested_columns,
                            "return_csv": False
                        }
                    }
                },
                "next_steps": {
                    "after_confirmation": [
                        "Schema will be saved to vector database",
                        "Optional: Download sample CSV data",
                        "Call /api/aips/domains/create to create new domain"
                    ]
                }
            })
    except Exception as e:
        logger.error(f"Error in regenerate suggestions: {str(e)}")
        return JSONResponse({
            "error": str(e),
            "message": "Failed to regenerate schema suggestions"
        }, status_code=500)
    

@router.put("/extend-schema")
async def extend_domain(request: Request):
    """
    Extend an existing domain with new columns using AI suggestions.
    
    Analyzes the existing domain schema and suggests complementary columns
    that don't conflict with existing ones. Supports user preferences for
    the type and style of new columns to add.
    """
    try:
        body = await request.json()
        
        # Validate required fields
        domain_name = body.get("domain")
        if not domain_name:
            return JSONResponse({
                "error": "domain is required"
            }, status_code=400)
        
        # Get existing domain schema first
        store = get_store()
        if store is None:
            return JSONResponse({
                "error": "OpenSearch store not available"
            }, status_code=503)
        
        # Query for existing domain
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
                "error": f"Domain '{domain_name}' not found",
                "message": "Cannot extend a domain that doesn't exist"
            }, status_code=404)
        
        existing_columns = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            existing_columns.append({
                "column_name": source.get("column_name", "unknown"),
                "type": source.get("metadata", {}).get("type", "string"),
                "sample_values": source.get("sample_values", []),
                "metadata": source.get("metadata", {})
            })
        
        existing_domain = {
            "found": True,
            "columns": existing_columns
        }
        
        if not existing_domain.get("found"):
            return JSONResponse({
                "error": f"Domain '{domain_name}' not found",
                "message": "Cannot extend a domain that doesn't exist"
            }, status_code=404)
        
        existing_columns = existing_domain.get("columns", [])
        existing_column_names = [col.get("column_name", "") for col in existing_columns]
        
        # Get user-provided columns and extension preferences
        user_provided_columns = body.get("columns", [])
        extension_preferences = body.get("extension_preferences", {})
        suggested_columns_count = extension_preferences.get("column_count", 3)
        focus_area = extension_preferences.get("focus_area", "")
        style = extension_preferences.get("style", "standard")
        suggest_additional = body.get("suggest_additional", False)
        
        new_columns = []
        user_columns_added = []
        ai_suggested_columns = []
        duplicates_skipped = []
        
        # Enhanced duplicate checking - case insensitive and with detailed reporting
        duplicate_details = []
        
        # Process user-provided columns first
        if user_provided_columns:
            existing_column_map = {col.lower(): col for col in existing_column_names}
            
            for column_name in user_provided_columns:
                column_lower = column_name.lower()
                if column_lower in existing_column_map:
                    # Store detailed duplicate information
                    duplicate_details.append({
                        "requested_column": column_name,
                        "existing_column": existing_column_map[column_lower],
                        "reason": "Column already exists in the domain",
                        "suggestion": "Use a different column name or skip if the same field"
                    })
                    duplicates_skipped.append(column_name)
                    continue
                
                # Check for similar names to prevent near-duplicates
                similar_columns = [
                    existing for existing in existing_column_names 
                    if _is_similar_column_name(column_name, existing)
                ]
                if similar_columns:
                    # Store warning about similar column names
                    duplicate_details.append({
                        "requested_column": column_name,
                        "similar_existing_columns": similar_columns,
                        "reason": "Similar column names found",
                        "suggestion": "Verify if these columns serve different purposes"
                    })
                
                # Create basic column structure for user-provided columns
                user_column = {
                    "column_name": column_name,
                    "type": "string",  # Default type, could be enhanced later
                    "description": f"User-defined column: {column_name}",
                    "sample_values": [],
                    "source": "user_provided"
                }
                new_columns.append(user_column)
                user_columns_added.append(column_name)
            
            # If all columns were duplicates, return detailed error
            if len(duplicates_skipped) == len(user_provided_columns):
                return JSONResponse({
                    "error": "All provided columns already exist",
                    "status": "error",
                    "message": "Cannot extend domain - all requested columns are duplicates",
                    "domain": domain_name,
                    "duplicate_details": duplicate_details,
                    "existing_columns": existing_column_names,
                    "suggestions": [
                        "Use different column names",
                        "Check existing columns first",
                        "Use /api/aips/domains/suggest-extend-schema for suggestions"
                    ]
                }, status_code=400)
        
        # Generate AI suggestions if requested or if no user columns provided
        if suggest_additional or (not user_provided_columns and not extension_preferences.get("no_ai_suggestions", False)):
            from app.agents.schema_suggester import SchemaSuggesterEnhanced
            suggester = SchemaSuggesterEnhanced()
            
            # Update exclude list to include user-provided columns
            all_existing_columns = existing_column_names + user_columns_added
            
            # Create enhanced extension prompt with business intelligence focus
            extension_description = f"""
            You are a Data Integration Specialist with expertise in schema evolution and domain expansion.
            
            **CURRENT DOMAIN:** {domain_name}
            **EXISTING SCHEMA COLUMNS:** {', '.join(existing_column_names)}
            **USER-PROVIDED COLUMNS:** {', '.join(user_columns_added) if user_columns_added else 'None'}
            **FOCUS AREA:** {focus_area if focus_area else 'General business enhancement'}
            
            **EXTENSION ANALYSIS:**
            
            **Gap Analysis:** Identify missing critical business dimensions, regulatory compliance gaps, operational metrics, and integration opportunities.
            
            **Pattern Consistency:** Maintain naming conventions and data type patterns from existing schema while ensuring new columns complement existing ones.
            
            **Business Value Assessment:** Consider adding columns for enhanced analytics, improved insights, regulatory compliance, operational efficiency, and downstream system integration.
            
            **EXTENSION CATEGORIES:**
            - Temporal Enhancements: Additional date/time tracking fields
            - Behavioral Data: User interaction patterns, preferences, activity metrics  
            - Metadata Fields: Data lineage, quality scores, confidence levels
            - Relationship Fields: Foreign keys to other business domains
            - Computed Fields: Derived metrics and KPIs
            
            Generate {suggested_columns_count} additional columns that meaningfully extend the existing schema without duplication.
            Focus on business-critical fields that would provide operational value and analytical insights.
            """
            
            # Generate extension suggestions
            user_preferences = {
                "exclude_columns": all_existing_columns,  # Avoid duplicates
                "style": style,
                "column_count": suggested_columns_count,
                "include_keywords": extension_preferences.get("include_keywords", []),
                "iteration": extension_preferences.get("iteration", 1)
            }
            
            extension_schema = await suggester.bootstrap_schema_with_preferences(
                business_description=extension_description,
                user_preferences=user_preferences
            )
            
            ai_columns = extension_schema.get("columns", [])
            for col in ai_columns:
                col["source"] = "ai_suggested"
                ai_suggested_columns.append(col["column_name"])
            
            new_columns.extend(ai_columns)
        
        # Add the new columns to the existing domain
        combined_schema = existing_columns + new_columns

        # Store the extended domain
        domain_data = {
            "domain": domain_name,
            "columns": combined_schema,
            "extended": True,
            "extension_info": {
                "original_column_count": len(existing_columns),
                "new_column_count": len(new_columns),
                "total_column_count": len(combined_schema),
                "extension_focus": focus_area,
                "extension_style": style
            }
        }

        # Generate embeddings for all new columns
        from app.embedding.embedder import embed_column_names_batched_async
        new_column_names = [col["column_name"] for col in new_columns]
        try:
            new_embeddings = await embed_column_names_batched_async(new_column_names)
        except Exception as e:
            logger.error(f"Failed to generate embeddings for new columns: {e}")
            new_embeddings = [[] for _ in new_column_names]  # fallback to empty embeddings

        # Create extended domain in vector store
        store = get_store()
        # Batch upsert all new columns at once
        column_docs = []
        for i, column in enumerate(new_columns):
            column_doc = ColumnDoc(
                column_id=f"{domain_name}.{column['column_name']}",
                column_name=column["column_name"],
                embedding=new_embeddings[i],
                sample_values=column.get("sample_values", []),
                metadata={
                    "domain": domain_name,
                    "type": column.get("type", "string"),
                    "description": column.get("description", ""),
                    "is_extension": True,
                    "extension_focus": focus_area
                }
            )
            column_docs.append(column_doc)
        store.upsert_columns(column_docs)
        
        return JSONResponse({
            "status": "success",
            "message": f"Domain '{domain_name}' extended successfully",
            "domain": domain_name,
            "extension_summary": {
                "original_columns": len(existing_columns),
                "user_columns_added": len(user_columns_added),
                "ai_suggested_columns": len(ai_suggested_columns),
                "total_new_columns": len(new_columns),
                "total_columns": len(combined_schema),
                "duplicates_skipped": len(duplicates_skipped),
                "focus_area": focus_area,
                "style": style
            },
            "new_columns": new_columns,
            "user_provided_columns": user_columns_added,
            "ai_suggested_columns": ai_suggested_columns,
            "duplicates_skipped": duplicates_skipped,
            "complete_schema": combined_schema,
            "actions": {
                "suggest_more": f"/api/aips/domains/suggest-extend-schema/{domain_name}",
                "view_domain": f"/api/aips/domains/{domain_name}",
                "regenerate_extensions": "/api/aips/domains/extend-schema"
            }
        })
        
    except Exception as e:
        logger.error(f"Error extending domain: {str(e)}")
        return JSONResponse({
            "error": str(e),
            "message": f"Failed to extend domain '{domain_name}'"
        }, status_code=500)
    

@router.post("/suggest-extend-schema/{domain_name}")
async def suggest_extensions(domain_name: str, request: Request):
    """
    Suggest additional columns for an existing domain without modifying it.
    
    Provides AI-powered suggestions for columns that could enhance the domain
    based on the existing schema and user-specified preferences or focus areas.
    """
    try:
        try:
            body = await request.json()
        except Exception:
            body = {}  # Default to empty dict if body is empty or invalid
        
        # Get existing domain schema
        store = get_store()
        if store is None:
            return JSONResponse({
                "error": "OpenSearch store not available"
            }, status_code=503)
        
        # Query for existing domain
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
                "error": f"Domain '{domain_name}' not found",
                "message": "Cannot suggest extensions for a domain that doesn't exist"
            }, status_code=404)
        
        existing_columns = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            existing_columns.append({
                "column_name": source.get("column_name", "unknown"),
                "type": source.get("metadata", {}).get("type", "string"),
                "sample_values": source.get("sample_values", []),
                "metadata": source.get("metadata", {})
            })
        
        existing_domain = {
            "found": True,
            "columns": existing_columns
        }
        
        if not existing_domain.get("found"):
            return JSONResponse({
                "error": f"Domain '{domain_name}' not found",
                "message": "Cannot suggest extensions for a domain that doesn't exist"
            }, status_code=404)
        
        existing_columns = existing_domain.get("columns", [])
        existing_column_names = [col.get("column_name", "") for col in existing_columns]
        
        # Get suggestion preferences
        suggestion_preferences = body.get("suggestion_preferences", {})
        suggested_count = suggestion_preferences.get("column_count", 5)
        focus_areas = suggestion_preferences.get("focus_areas", ["analytics", "compliance", "operations"])
        style = suggestion_preferences.get("style", "comprehensive")
        
        # Use enhanced schema suggester
        from app.agents.schema_suggester import SchemaSuggesterEnhanced
        suggester = SchemaSuggesterEnhanced()
        
        suggestions_by_focus = {}
        
        # Generate suggestions for each focus area
        for focus_area in focus_areas:
            extension_description = f"""
            Suggest columns to extend the domain '{domain_name}' with focus on {focus_area}.
            
            Existing columns: {', '.join(existing_column_names)}
            
            Generate {suggested_count} columns that would enhance the domain for {focus_area} purposes.
            Avoid duplicating existing columns.
            """
            
            user_preferences = {
                "exclude_columns": existing_column_names,
                "style": style,
                "column_count": suggested_count,
                "include_keywords": [focus_area],
                "iteration": 1
            }
            
            focus_suggestions = await suggester.bootstrap_schema_with_preferences(
                business_description=extension_description,
                user_preferences=user_preferences
            )
            
            suggestions_by_focus[focus_area] = {
                "columns": focus_suggestions.get("columns", []),
                "total_columns": len(focus_suggestions.get("columns", [])),
                "description": f"Columns focused on {focus_area} enhancement"
            }
        
        return JSONResponse({
            "status": "success",
            "message": f"Extension suggestions generated for domain '{domain_name}'",
            "domain_name": domain_name,
            "existing_column_count": len(existing_columns),
            "suggestion_summary": {
                "total_focus_areas": len(focus_areas),
                "columns_per_focus": suggested_count,
                "style": style
            },
            "suggestions": suggestions_by_focus,
            "existing_columns": [col.get("column_name") for col in existing_columns],
            "actions": {
                "extend_domain": "/api/aips/domains/extend-schema",
                "view_domain": f"/api/aips/domains/{domain_name}",
                "resuggest_domain_schema": "/api/aips/domains/suggest-schema"
            }
        })
        
    except Exception as e:
        logger.error(f"Error suggesting extensions for domain: {str(e)}")
        return JSONResponse({
            "error": str(e),
            "message": f"Failed to suggest extensions for domain '{domain_name}'"
        }, status_code=500)
