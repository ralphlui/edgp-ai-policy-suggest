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
            # Generate embedding for the current schema
            schema_str = self._schema_to_string(schema)
            schema_embedding = await embed_column_names_batched_async([schema_str])
            
            # Retrieve similar policies
            similar_policies = await self.policy_store.retrieve_similar_policies(
                query_embedding=schema_embedding[0],
                domain=domain,
                min_success_rate=0.8,
                top_k=3
            )

            # Format historical context
            historical_context = self._format_historical_context(similar_policies)

            # Build enhanced prompt
            prompt = f"""
Given the current schema:
{schema_str}

Here are some successful validation rules that worked well for similar schemas:
{historical_context}

Based on these successful patterns and the current schema, suggest appropriate validation rules.
Focus on rules that have proven effective in similar contexts while adapting them to the specific
requirements of the current schema.

Ensure the suggested rules:
1. Follow patterns that have shown high success rates
2. Are adapted to the specific data types and constraints of the current schema
3. Maintain consistency with domain-specific validation requirements
4. Consider both data quality and business logic validations
"""
            return prompt

        except Exception as e:
            logger.error(f"Failed to enhance prompt with history: {e}")
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
            # Generate embedding for the schema
            schema_str = self._schema_to_string(schema)
            schema_embedding = await embed_column_names_batched_async([schema_str])

            # Create policy document
            policy = PolicyHistoryDoc(
                policy_id=str(uuid.uuid4()),
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
            await self.policy_store.store_policy(policy)
            logger.info(f"Stored successful policy for domain {domain}")

        except Exception as e:
            logger.error(f"Failed to store successful policy: {e}")
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
            return "No similar historical policies found."

        context_parts = []
        for idx, policy in enumerate(similar_policies, 1):
            success_rate = policy.get("performance_metrics", {}).get("success_rate", 0)
            rules = policy.get("rules", [])
            
            context_parts.append(f"\nHistorical Policy {idx} (Success Rate: {success_rate:.2%}):")
            for rule in rules:
                context_parts.append(f"- Rule Type: {rule.get('type')}")
                context_parts.append(f"  Parameters: {rule.get('parameters')}")
                if rule.get('description'):
                    context_parts.append(f"  Description: {rule.get('description')}")

        return "\n".join(context_parts)