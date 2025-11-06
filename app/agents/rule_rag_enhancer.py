"""
Enhanced rule suggestion using RAG
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
import logging
from app.aoss.policy_history_store import PolicyHistoryStore, PolicyHistoryDoc
from app.embedding.embedder import embed_column_names_batched_async
from app.core.config import settings

logger = logging.getLogger(__name__)

class RuleRAGEnhancer:
    def __init__(self):
        self.policy_store = PolicyHistoryStore()

    async def enhance_prompt_with_history(
        self,
        schema: Dict[str, Any],
        domain: str
    ) -> str:
        """Enhance the prompt with relevant historical context"""
        try:
            logger.info(f"Starting RAG enhancement for domain: {domain}")
            logger.info(f"Schema contains {len(schema)} columns")

            # Generate embedding for the current schema
            logger.info("Converting schema to string format for embedding")
            schema_str = self._schema_to_string(schema)
            logger.debug(f"Schema string generated: {len(schema_str)} characters")
            
            logger.info("Generating embeddings for schema")
            schema_embedding = await embed_column_names_batched_async([schema_str])
            logger.info(f"Embedding generated successfully: {len(schema_embedding[0])} dimensions")
            
            # Retrieve similar policies
            logger.info(f"Retrieving similar policies for domain {domain} with min success rate 0.8")
            similar_policies = await self.policy_store.retrieve_similar_policies(
                query_embedding=schema_embedding[0],
                domain=domain,
                min_success_rate=0.8,
                top_k=3
            )
            logger.info(f"Found {len(similar_policies)} similar policies")

            # Format historical context
            logger.info("Formatting historical context from similar policies")
            historical_context = self._format_historical_context(similar_policies)
            logger.debug(f"Historical context generated: {len(historical_context)} characters")

            # Build enhanced prompt
            logger.info("Building enhanced prompt with historical context")
            prompt = f"""You are an expert data governance specialist. Generate Great Expectations validation rules for the given schema.

**CURRENT SCHEMA:**
{schema_str}

**HISTORICAL SUCCESS PATTERNS:**
{historical_context}

**INSTRUCTIONS:**
Based on the historical patterns above that have proven successful, suggest appropriate validation rules for the current schema.

**RULES TO FOLLOW:**
1. Follow patterns that have shown high success rates (>80%)
2. Adapt rules to the specific data types and constraints of the current schema
3. Maintain consistency with domain-specific validation requirements
4. Consider both data quality and business logic validations
5. Use the exact Great Expectations rule format

**OUTPUT FORMAT:**
Return ONLY a valid JSON array with this structure:
[{{"column":"column_name","expectations":[{{"expectation_type":"rule_type","kwargs":{{}},"meta":{{"reasoning":"why this rule"}}}}]}}]

**DOMAIN CONTEXT:** {domain}

Generate rules now:"""
            return prompt

        except Exception as e:
            logger.error(f"Failed to enhance prompt with history: {e}", exc_info=True)
            logger.warning("Falling back to basic prompt without historical context")
            # Return a basic prompt as fallback
            return f"Given the schema: {schema_str}\nSuggest appropriate validation rules."

    async def store_successful_policy(
        self,
        domain: str,
        schema: Dict[str, Any],
        rules: List[Dict[str, Any]],
        performance_metrics: Dict[str, Any]
    ):
        """Store successful policy for future reference"""
        try:
            logger.info(f"Starting to store successful policy for domain: {domain}")
            logger.info(f"Policy contains {len(rules)} rules with performance metrics: {performance_metrics}")

            # Generate embedding for the schema
            logger.info("Converting schema to string format for embedding")
            schema_str = self._schema_to_string(schema)
            logger.debug(f"Schema string generated: {len(schema_str)} characters")

            logger.info("Generating embeddings for schema storage")
            schema_embedding = await embed_column_names_batched_async([schema_str])
            logger.info(f"Embedding generated successfully: {len(schema_embedding[0])} dimensions")

            # Create policy document
            policy_id = str(uuid.uuid4())
            logger.info(f"Creating policy document with ID: {policy_id}")
            policy = PolicyHistoryDoc(
                policy_id=policy_id,
                domain=domain,
                schema=schema,
                rules=rules,
                performance_metrics=performance_metrics,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                embedding=schema_embedding[0],
                metadata={
                    "status": "active",
                    "version": "1.0"
                }
            )

            # Store the policy
            logger.info(f"Storing policy {policy_id} in policy history store")
            await self.policy_store.store_policy(policy)
            logger.info(f"Successfully stored policy {policy_id} for domain {domain}")
            logger.debug(f"Policy metrics: success_rate={performance_metrics.get('success_rate', 0):.2%}")

        except Exception as e:
            logger.error(f"Failed to store successful policy: {e}", exc_info=True)
            logger.error(f"Domain: {domain}, Rule count: {len(rules)}, Schema size: {len(schema)}")
            raise

    def _schema_to_string(self, schema: Dict[str, Any]) -> str:
        """Convert schema to string format for embedding"""
        schema_parts = []
        for col_name, col_info in schema.items():
            dtype = col_info.get("dtype", "unknown")
            sample_values = col_info.get("sample_values", [])
            schema_parts.append(f"Column: {col_name}, Type: {dtype}, Examples: {', '.join(map(str, sample_values[:3]))}")
        return "\n".join(schema_parts)

    def _format_historical_context(self, similar_policies: List[Dict[str, Any]]) -> str:
        """Format historical policies into readable context"""
        if not similar_policies:
            return "No similar historical policies found. Use your expertise to suggest appropriate rules."

        context_parts = []
        for idx, policy in enumerate(similar_policies, 1):
            success_rate = policy.get("performance_metrics", {}).get("success_rate", 0)
            rules = policy.get("rules", [])
            
            context_parts.append(f"\n--- Historical Policy {idx} (Success Rate: {success_rate:.1%}) ---")
            
            if rules:
                context_parts.append("Successful Rules Used:")
                for rule in rules:
                    rule_name = rule.get('rule_name', rule.get('type', 'Unknown'))
                    column_name = rule.get('column_name', rule.get('column', 'N/A'))
                    value = rule.get('value', rule.get('parameters', {}))
                    
                    context_parts.append(f"• {rule_name} on column '{column_name}'")
                    if value:
                        context_parts.append(f"  Parameters: {value}")
                        
                    description = rule.get('description', rule.get('meta', {}).get('reasoning', ''))
                    if description:
                        context_parts.append(f"  Purpose: {description}")
            else:
                context_parts.append("No specific rule details available")

        context_parts.append("\n--- End Historical Context ---\n")
        return "\n".join(context_parts)