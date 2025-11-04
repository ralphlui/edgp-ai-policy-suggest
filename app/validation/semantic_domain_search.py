"""
Semantic Domain Search Module
Provides intelligent domain matching for natural language queries using LLM.
"""

import logging
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from difflib import SequenceMatcher
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.aws.aws_secrets_service import require_openai_api_key
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class DomainMatch:
    """Represents a domain match result"""
    domain: str
    score: float
    match_type: str
    explanation: str


class SemanticDomainSearch:
    """
    LLM-powered semantic domain search that supports:
    1. Natural language understanding
    2. Intent recognition
    3. Context-aware matching
    4. Semantic similarity analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize LLM with same configuration as rule tools
        self._llm = None
        
        # Fallback algorithmic matching for backup
        self.domain_keywords = {
            'customer': ['customer', 'client', 'user', 'account', 'profile', 'member', 'person', 'individual'],
            'product': ['product', 'item', 'goods', 'merchandise', 'catalog', 'inventory', 'stock'],
            'order': ['order', 'purchase', 'transaction', 'sale', 'payment', 'invoice', 'receipt'],
            'employee': ['employee', 'staff', 'worker', 'personnel', 'team', 'hr', 'human'],
            'finance': ['finance', 'financial', 'money', 'payment', 'billing', 'accounting', 'revenue'],
            'inventory': ['inventory', 'stock', 'warehouse', 'storage', 'supply', 'asset'],
            'analytics': ['analytics', 'metrics', 'stats', 'report', 'dashboard', 'insight', 'analysis'],
            'marketing': ['marketing', 'campaign', 'promotion', 'advertisement', 'lead', 'prospect'],
            'support': ['support', 'help', 'ticket', 'issue', 'service', 'assistance', 'contact'],
            'shipping': ['shipping', 'delivery', 'logistics', 'transport', 'fulfillment', 'dispatch'],
            'location': ['location', 'address', 'geography', 'region', 'place', 'site', 'venue', 'coordinates']
        }
    
    @property
    def llm(self):
        """Lazy-load LLM to avoid initialization issues"""
        if self._llm is None:
            try:
                openai_key = require_openai_api_key()
                self._llm = ChatOpenAI(
                    model=settings.schema_llm_model,
                    openai_api_key=openai_key,
                    temperature=0.1,  # Low temperature for consistent results
                    timeout=30  # 30 second timeout
                )
                self.logger.info(" [LLM] Initialized ChatOpenAI for semantic domain search")
            except Exception as e:
                self.logger.error(f" [LLM] Failed to initialize: {e}")
                self._llm = None
        return self._llm
    
    def analyze_query_with_llm(self, query: str, available_domains: List[str]) -> Dict:
        """
        Use LLM to analyze the query and find semantic matches
        
        Args:
            query: Natural language query
            available_domains: List of available domains
            
        Returns:
            Dict with LLM analysis results
        """
        if not self.llm:
            self.logger.warning(" [LLM] LLM not available, falling back to algorithmic matching")
            return {"success": False, "error": "LLM not available"}
        
        try:
            system_prompt = """You are an expert at understanding domain names and matching user queries to appropriate data domains.

Your task is to analyze a user's natural language query and find the most relevant domain(s) from the available list.

Consider:
1. Semantic meaning and intent
2. Synonyms and related concepts
3. Business context and typical data organization
4. Partial matches and fuzzy understanding

Return your analysis as a JSON object with:
{
    "matches": [
        {
            "domain": "exact_domain_name",
            "confidence": 0.95,
            "reasoning": "Why this domain matches the query"
        }
    ],
    "query_intent": "Brief description of what the user is looking for",
    "keywords_extracted": ["key", "terms", "from", "query"]
}

Confidence scale:
- 0.9-1.0: Very high confidence (exact semantic match)
- 0.7-0.89: High confidence (strong semantic similarity)
- 0.5-0.69: Medium confidence (related concepts)
- 0.3-0.49: Low confidence (weak connection)
- 0.0-0.29: Very low confidence (minimal relevance)

Only include matches with confidence >= 0.3. Return empty matches array if no relevant domains found."""

            user_prompt = f"""User Query: "{query}"

Available Domains:
{json.dumps(available_domains, indent=2)}

Analyze the query and find matching domains. Focus on semantic meaning, not just literal string matching."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            self.logger.info(f" [LLM] Analyzing query: '{query}' against {len(available_domains)} domains")
            
            response = self.llm.invoke(messages)
            response_text = response.content.strip()
            
            # Parse JSON response
            try:
                llm_result = json.loads(response_text)
                self.logger.info(f" [LLM] Analysis completed - found {len(llm_result.get('matches', []))} matches")
                
                # Log top matches for debugging
                for match in llm_result.get('matches', [])[:3]:
                    self.logger.info(f"    {match.get('domain')} (confidence: {match.get('confidence', 0):.2f})")
                
                return {
                    "success": True,
                    "result": llm_result,
                    "method": "llm"
                }
                
            except json.JSONDecodeError as e:
                self.logger.error(f" [LLM] Failed to parse JSON response: {e}")
                self.logger.debug(f"Raw response: {response_text}")
                return {"success": False, "error": "Invalid JSON response from LLM"}
                
        except Exception as e:
            self.logger.error(f" [LLM] Error during analysis: {e}")
            return {"success": False, "error": str(e)}
    
    def search_domains_with_llm(self, query: str, available_domains: List[str], max_results: int = 5) -> List[DomainMatch]:
        """
        Perform LLM-powered semantic domain search with algorithmic fallback
        
        Args:
            query: Natural language query
            available_domains: List of available domain names
            max_results: Maximum number of results to return
            
        Returns:
            List of DomainMatch objects sorted by relevance score
        """
        self.logger.info(f" [LLM] Starting semantic search for: '{query}'")
        
        if not query or not available_domains:
            return []
        
        # Try LLM analysis first
        llm_analysis = self.analyze_query_with_llm(query, available_domains)
        
        if llm_analysis.get("success"):
            # Convert LLM results to DomainMatch objects
            matches = []
            llm_result = llm_analysis["result"]
            
            for match_data in llm_result.get("matches", []):
                domain = match_data.get("domain")
                confidence = match_data.get("confidence", 0)
                reasoning = match_data.get("reasoning", "LLM semantic match")
                
                # Validate domain exists in available list
                if domain in available_domains:
                    matches.append(DomainMatch(
                        domain=domain,
                        score=confidence,
                        match_type="llm_semantic",
                        explanation=f"LLM Analysis: {reasoning}"
                    ))
                else:
                    self.logger.warning(f" [LLM] Suggested domain '{domain}' not in available list")
            
            # Sort by confidence and limit results
            matches.sort(key=lambda x: x.score, reverse=True)
            final_matches = matches[:max_results]
            
            self.logger.info(f" [LLM] Found {len(final_matches)} semantic matches")
            return final_matches
        
        else:
            # Fallback to algorithmic matching
            self.logger.warning(" [LLM] LLM analysis failed, using algorithmic fallback")
            return self.search_domains_algorithmic(query, available_domains, max_results)
    
    def search_domains_algorithmic(self, query: str, available_domains: List[str], max_results: int = 5) -> List[DomainMatch]:
        """
        Algorithmic fallback for when LLM is not available
        """
        self.logger.info(f" [ALGO] Performing algorithmic search for: '{query}'")
        
        all_matches = []
        
        # Extract keywords and find keyword-based matches
        keywords = self.extract_keywords(query)
        if keywords:
            keyword_matches = self.find_keyword_matches(keywords, available_domains)
            all_matches.extend(keyword_matches)
        
        # Find fuzzy matches
        fuzzy_matches = self.find_fuzzy_matches(query, available_domains)
        all_matches.extend(fuzzy_matches)
        
        # Find partial matches
        partial_matches = self.find_partial_matches(query, available_domains)
        all_matches.extend(partial_matches)
        
        # Deduplicate and merge scores
        domain_scores = {}
        for match in all_matches:
            if match.domain in domain_scores:
                existing = domain_scores[match.domain]
                combined_score = max(existing.score, match.score)
                domain_scores[match.domain] = DomainMatch(
                    domain=match.domain,
                    score=combined_score,
                    match_type=f"{existing.match_type}+{match.match_type}",
                    explanation=f"{existing.explanation}; {match.explanation}"
                )
            else:
                domain_scores[match.domain] = match
        
        # Sort and return top results
        final_matches = sorted(domain_scores.values(), key=lambda x: x.score, reverse=True)[:max_results]
        
        self.logger.info(f" [ALGO] Found {len(final_matches)} algorithmic matches")
        return final_matches
    
    def search_domains(self, query: str, available_domains: List[str], max_results: int = 5) -> List[DomainMatch]:
        """
        Main search method - tries LLM first, falls back to algorithmic
        """
        return self.search_domains_with_llm(query, available_domains, max_results)
    
    # Algorithmic helper methods (used as fallback)
    def extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from natural language query"""
        query_lower = query.lower().strip()
        words = re.findall(r'\b\w+\b', query_lower)
        stop_words = {
            'can', 'i', 'have', 'get', 'for', 'the', 'a', 'an', 'is', 'are', 'was', 'were',
            'rule', 'rules', 'suggest', 'suggestion', 'domain', 'available', 'please',
            'want', 'need', 'looking', 'search', 'find', 'show', 'me', 'any', 'some'
        }
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords
    
    def find_keyword_matches(self, keywords: List[str], available_domains: List[str]) -> List[DomainMatch]:
        """Find domains that match extracted keywords (algorithmic fallback)"""
        matches = []
        
        for domain in available_domains:
            domain_lower = domain.lower()
            domain_score = 0.0
            matched_keywords = []
            
            # Direct keyword matching in domain name
            for keyword in keywords:
                if keyword in domain_lower:
                    domain_score += 0.8
                    matched_keywords.append(keyword)
            
            # Semantic keyword matching using our mappings
            for category, synonyms in self.domain_keywords.items():
                if category in domain_lower or any(syn in domain_lower for syn in synonyms):
                    for keyword in keywords:
                        if keyword in synonyms:
                            domain_score += 0.6
                            matched_keywords.append(f"{keyword}→{category}")
            
            if domain_score > 0:
                normalized_score = min(1.0, domain_score / len(keywords) if keywords else 0)
                matches.append(DomainMatch(
                    domain=domain,
                    score=normalized_score,
                    match_type="keyword",
                    explanation=f"Matched keywords: {', '.join(matched_keywords)}"
                ))
        
        return matches
    
    def find_fuzzy_matches(self, query: str, available_domains: List[str], threshold: float = 0.4) -> List[DomainMatch]:
        """Find domains using fuzzy string matching (algorithmic fallback)"""
        matches = []
        query_clean = re.sub(r'[^a-zA-Z0-9_]', '', query.lower())
        
        for domain in available_domains:
            domain_clean = re.sub(r'[^a-zA-Z0-9_]', '', domain.lower())
            similarity = SequenceMatcher(None, query_clean, domain_clean).ratio()
            
            if similarity >= threshold:
                matches.append(DomainMatch(
                    domain=domain,
                    score=similarity,
                    match_type="fuzzy",
                    explanation=f"String similarity: {similarity:.2f}"
                ))
        
        return matches
    
    def find_partial_matches(self, query: str, available_domains: List[str]) -> List[DomainMatch]:
        """Find domains using partial substring matching (algorithmic fallback)"""
        matches = []
        query_parts = re.findall(r'\w+', query.lower())
        
        for domain in available_domains:
            domain_lower = domain.lower()
            match_count = 0
            matched_parts = []
            
            for part in query_parts:
                if len(part) > 2:
                    if part in domain_lower:
                        match_count += 1
                        matched_parts.append(part)
                    elif any(part in domain_part for domain_part in domain_lower.split('_')):
                        match_count += 0.5
                        matched_parts.append(f"{part}*")
            
            if match_count > 0:
                score = min(1.0, match_count / len(query_parts))
                matches.append(DomainMatch(
                    domain=domain,
                    score=score,
                    match_type="partial",
                    explanation=f"Partial matches: {', '.join(matched_parts)}"
                ))
        
        return matches
    
    def search_domains(self, query: str, available_domains: List[str], max_results: int = 5) -> List[DomainMatch]:
        """
        NOTE: The LLM-first `search_domains_with_llm` method is the intended public
        entry point and will be used. The algorithmic-only implementation that used
        to be here has been removed so the LLM path is honored and algorithmic
        methods remain available as fallback via `search_domains_algorithmic`.

        This method delegates to the LLM-first implementation.
        """
        return self.search_domains_with_llm(query, available_domains, max_results)
    
    def get_search_suggestions(self, query: str, available_domains: List[str]) -> Dict:
        """
        Get search suggestions and explanations for a query using LLM analysis
        
        Returns a structured response with suggestions and explanations
        """
        matches = self.search_domains(query, available_domains)
        
        if not matches:
            return {
                "found_domains": False,
                "message": "No matching domains found",
                "method": "llm_with_algorithmic_fallback",
                "suggestions": {
                    "check_spelling": "Verify domain names are spelled correctly",
                    "try_keywords": "Try using keywords like 'customer', 'product', 'order', 'location'",
                    "view_all": "Use /domains endpoint to see all available domains"
                },
                "available_keywords": list(self.domain_keywords.keys())
            }
        
        # Categorize matches by score (using user-friendly confidence levels)
        # Lowered high confidence threshold to 0.8 for better user experience
        high_confidence = [m for m in matches if m.score >= 0.8]  # 80%+ is high confidence
        medium_confidence = [m for m in matches if 0.6 <= m.score < 0.8]  # 60-79% is medium  
        low_confidence = [m for m in matches if m.score < 0.6]  # <60% is low
        
        response = {
            "found_domains": True,
            "total_matches": len(matches),
            "method": "llm_semantic_analysis" if matches and matches[0].match_type.startswith("llm") else "algorithmic_fallback",
            "query_analysis": {
                "extracted_keywords": self.extract_keywords(query),
                "query_intent": "domain_search"
            }
        }
        
        if high_confidence:
            response["high_confidence_matches"] = [
                {
                    "domain": m.domain,
                    "confidence": f"{m.score:.1%}",
                    "reason": m.explanation,
                    "method": m.match_type
                } for m in high_confidence
            ]
            response["recommended_action"] = f"Try using domain: '{high_confidence[0].domain}'"
        
        if medium_confidence:
            response["possible_matches"] = [
                {
                    "domain": m.domain,
                    "confidence": f"{m.score:.1%}",
                    "reason": m.explanation,
                    "method": m.match_type
                } for m in medium_confidence
            ]
        
        if low_confidence and not high_confidence and not medium_confidence:
            response["weak_matches"] = [
                {
                    "domain": m.domain,
                    "confidence": f"{m.score:.1%}",
                    "reason": m.explanation,
                    "method": m.match_type
                } for m in low_confidence[:3]  # Show only top 3 weak matches
            ]
        
        return response


def enhance_domain_validation(query: str, available_domains: List[str]) -> Tuple[bool, Optional[str], Dict]:
    """
    Enhanced domain validation with LLM-powered semantic search
    
    Args:
        query: User input (could be exact domain or natural language)
        available_domains: List of available domains
        
    Returns:
        Tuple of (is_exact_match, suggested_domain, semantic_results)
    """
    logger.info(f" [DOMAIN] Enhancing domain validation with LLM for: '{query}'")
    
    # First check for exact match
    query_normalized = query.lower().strip()
    exact_matches = [d for d in available_domains if d.lower() == query_normalized]
    
    if exact_matches:
        logger.info(f" [DOMAIN] Exact match found: '{exact_matches[0]}'")
        return True, exact_matches[0], {"match_type": "exact"}
    
    # No exact match, try LLM-powered semantic search
    semantic_search = SemanticDomainSearch()
    semantic_results = semantic_search.get_search_suggestions(query, available_domains)
    
    # Check if we have a high-confidence suggestion (using LLM confidence levels)
    suggested_domain = None
    numeric_confidence = 0.0
    
    # First check high confidence matches (80%+)
    if semantic_results.get("found_domains") and "high_confidence_matches" in semantic_results:
        best_match = semantic_results["high_confidence_matches"][0]
        suggested_domain = best_match["domain"]
        
        # Extract numeric confidence for decision making
        confidence_str = best_match.get("confidence", "0%")
        try:
            numeric_confidence = float(confidence_str.rstrip('%')) / 100.0  # Convert to 0-1 scale
        except (ValueError, AttributeError):
            numeric_confidence = 0.8  # Default if parsing fails
    
    # If no high confidence match, check medium confidence matches (60-79%) for good enough matches
    elif semantic_results.get("found_domains") and "possible_matches" in semantic_results:
        best_match = semantic_results["possible_matches"][0]
        suggested_domain = best_match["domain"]
        
        # Extract numeric confidence for decision making
        confidence_str = best_match.get("confidence", "0%")
        try:
            numeric_confidence = float(confidence_str.rstrip('%')) / 100.0  # Convert to 0-1 scale
        except (ValueError, AttributeError):
            numeric_confidence = 0.7  # Default if parsing fails
        
        # Only accept medium confidence if it's actually quite good (75%+)
        if numeric_confidence < 0.75:
            suggested_domain = None
            numeric_confidence = 0.0
    
    if suggested_domain and numeric_confidence >= 0.75:
        
        logger.info(f" [DOMAIN] LLM suggestion accepted: '{suggested_domain}' (confidence: {numeric_confidence:.1%})")
        
        # Add numeric confidence to results for decision making
        semantic_results["numeric_confidence"] = numeric_confidence
        semantic_results["confidence_threshold_met"] = numeric_confidence >= 0.75  # 75% threshold for auto-acceptance
        semantic_results["llm_powered"] = True
        
        return False, suggested_domain, semantic_results
    
    # No good matches found
    logger.info(f" [DOMAIN] No good matches found for: '{query}'")
    semantic_results["llm_powered"] = semantic_results.get("method", "").startswith("llm")
    return False, None, semantic_results