"""
Input Guardrails for Rule Suggestion API
Pre-validates user input before LLM processing to prevent unsafe or irrelevant requests.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class GuardrailViolationType(Enum):
    """Types of guardrail violations"""
    UNSAFE_CONTENT = "unsafe_content"
    INVALID_DOMAIN = "invalid_domain"
    SPAM_PATTERN = "spam_pattern"
    INJECTION_ATTEMPT = "injection_attempt"
    IRRELEVANT_REQUEST = "irrelevant_request"
    PROFANITY = "profanity"


@dataclass
class GuardrailViolation:
    """Represents a guardrail violation"""
    violation_type: GuardrailViolationType
    message: str
    suggested_action: str
    confidence: float
    detected_pattern: Optional[str] = None


class InputGuardrails:
    """
    Pre-LLM input validation to block unsafe or irrelevant requests
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Enhanced unsafe content patterns with more comprehensive coverage
        self.unsafe_patterns = {
            "violence_weapons": r'(?i)\b(?:kill|murder|assassin|shoot|gun|weapon|bomb|explos|terrorist|violence|harm|hurt|attack|knife|blade|poison|toxin)\b',
            "explosives_weapons": r'(?i)\b(?:bomb|explos|dynamite|tnt|c4|grenade|missile|rocket|ammunition|gunpowder|nitrate|fertilizer.*bomb|pipe.*bomb|pressure.*cooker)\b',
            "illegal_activities": r'(?i)\b(?:steal|theft|rob|burglar|hack|crack|break.*in|fraud|scam|counterfeit|forge|launder|smuggl|traffick|illegal|contraband)\b',
            "financial_crimes": r'(?i)\b(?:credit.*card.*theft|steal.*credit|fraud.*card|skimm|phish|identity.*theft|money.*launder|bitcoin.*hack|crypto.*steal)\b',
            "drugs_substances": r'(?i)\b(?:cocaine|heroin|methamphetamine|meth|fentanyl|lsd|ecstasy|mdma|marijuana.*grow|drug.*deal|drug.*manufactur|chemical.*weapon)\b',
            "hate_speech": r'(?i)\b(?:hate|racist|sexist|discriminat|bigot|nazi|supremacist|genocide|ethnic.*cleans|racial.*purity|white.*power|black.*face)\b',
            "self_harm": r'(?i)\b(?:suicide|self.*harm|cut.*myself|kill.*myself|end.*life|overdose|jump.*off|hang.*myself)\b',
            "cybercrime": r'(?i)\b(?:ddos|botnet|malware|ransomware|keylogger|trojan|virus.*create|hack.*into|sql.*injection|xss.*attack|social.*engineer)\b',
            "child_exploitation": r'(?i)\b(?:child.*porn|underage|minor.*sexual|pedophile|grooming|exploitation.*child)\b',
            "terrorism": r'(?i)\b(?:terrorist|isis|al.*qaeda|jihad|radicalize|extremist|terror.*attack|mass.*shooting|vehicle.*attack)\b'
        }
        
        # Enhanced injection attempt patterns
        self.injection_patterns = {
            "sql_injection": r'(?i)\b(?:select|insert|update|delete|drop|union|exec|execute)\s*(?:from|where|into)\b',
            "script_injection": r'(?i)<script[^>]*>|javascript:|data:text/html|onload=|onerror=',
            "command_injection": r'(?i)\b(?:system|exec|eval|shell_exec|passthru|`.*`|\$\(.*\))\s*\(',
            "prompt_injection": r'(?i)(?:ignore|forget|disregard|override|bypass).*(?:previous|above|instruction|prompt|rule|safety|filter)',
            "role_hijack": r'(?i)(?:you are|act as|pretend to be|roleplay|assume the role|now you|from now)',
            "system_override": r'(?i)(?:admin|root|sudo|administrator|superuser|system.*access|privilege.*escalat)',
            "encoding_bypass": r'(?i)(?:base64|hex|url.*encod|ascii|unicode|utf-8).*(?:decode|convert)',
        }
        
        # Spam/irrelevant patterns
        self.spam_patterns = {
            "repeated_chars": r'(.)\1{10,}',  # Same character repeated 10+ times
            "excessive_caps": r'[A-Z]{20,}',   # 20+ consecutive uppercase letters
            "url_spam": r'https?://[^\s]+',    # URLs (might be spam)
            "email_spam": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone_spam": r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'
        }
        
        # Domain-related keywords that should exist in legitimate requests
        self.domain_related_keywords = {
            'schema', 'table', 'column', 'data', 'database', 'field', 'record', 'row', 
            'validation', 'rule', 'policy', 'quality', 'check', 'constraint', 'format',
            'type', 'structure', 'metadata', 'attribute', 'property', 'dimension',
            'measure', 'entity', 'relationship', 'key', 'index', 'foreign', 'primary'
        }
        
    def validate_domain_input(self, domain_input: str, available_domains: List[str]) -> Optional[GuardrailViolation]:
        """
        Validate domain input with semantic search capabilities
        
        Args:
            domain_input: User's domain input (exact domain or natural language)
            available_domains: List of available domain names
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not domain_input or not available_domains:
            self.logger.warning("⚠️ [DOMAIN] Empty domain input or no available domains")
            return GuardrailViolation(
                violation_type=GuardrailViolationType.INVALID_DOMAIN,
                message="Domain name is required",
                suggested_action="Please provide a valid domain name",
                confidence=1.0
            )
        
        # Import here to avoid circular imports
        from app.validation.semantic_domain_search import enhance_domain_validation
        
        self.logger.debug(f" [DOMAIN] Validating domain: '{domain_input}' against {len(available_domains)} available domains")
        
        # Use enhanced domain validation with semantic search
        is_exact, suggested_domain, semantic_results = enhance_domain_validation(domain_input, available_domains)
        
        if is_exact:
            self.logger.info(f" [DOMAIN] Exact match found: '{suggested_domain}'")
            return None  # Valid - no violation
        
        if suggested_domain:
            # High-confidence semantic match found - this is valid
            confidence_info = semantic_results.get("high_confidence_matches", [{}])[0]
            confidence = confidence_info.get("confidence", "high")
            reason = confidence_info.get("reason", "semantic match")
            
            self.logger.info(f" [DOMAIN] Semantic match found: '{suggested_domain}' ({confidence}) - {reason}")
            return None  # Valid - no violation
        
        # No good matches found - provide helpful feedback
        if semantic_results.get("found_domains"):
            # We have some matches but not high confidence
            response_parts = [f"No exact match found for '{domain_input}'."]
            
            if "possible_matches" in semantic_results:
                possible = semantic_results["possible_matches"][:3]
                suggestions = [f"{m['domain']} ({m['confidence']})" for m in possible]
                response_parts.append(f"Possible matches: {', '.join(suggestions)}")
            
            if "weak_matches" in semantic_results:
                weak = semantic_results["weak_matches"][:2]
                suggestions = [m['domain'] for m in weak]
                response_parts.append(f"Similar domains: {', '.join(suggestions)}")
            
            # Add helpful suggestions
            if "suggestions" in semantic_results:
                suggestions = semantic_results["suggestions"]
                if "try_keywords" in suggestions:
                    response_parts.append("Try using keywords like 'customer', 'product', 'order'")
            
            message = " ".join(response_parts)
            self.logger.warning(f" [DOMAIN] Partial matches found for: '{domain_input}'")
            return GuardrailViolation(
                violation_type=GuardrailViolationType.INVALID_DOMAIN,
                message=message,
                suggested_action="Try one of the suggested similar domains or use the /domains endpoint to see all available domains",
                confidence=0.7,
                detected_pattern="partial_matches_available"
            )
        
        else:
            # No matches at all
            self.logger.warning(f" [DOMAIN] No matches found for domain: '{domain_input}'")
            
            # Provide helpful suggestions
            suggestions = []
            if len(available_domains) <= 10:
                suggestions = available_domains[:5]
            else:
                # Show a few example domains
                import random
                suggestions = random.sample(available_domains, min(5, len(available_domains)))
            
            suggestion_text = f"Available domains include: {', '.join(suggestions)}"
            if len(available_domains) > 5:
                suggestion_text += f" (and {len(available_domains) - 5} more)"
            
            return GuardrailViolation(
                violation_type=GuardrailViolationType.INVALID_DOMAIN,
                message=f"Domain '{domain_input}' not found. {suggestion_text}",
                suggested_action="Use natural language like 'customer domain' or call /domains endpoint to see all available domains",
                confidence=1.0
            )
    
    def detect_harmful_intent(self, content: str) -> List[GuardrailViolation]:
        """
        Advanced harmful intent detection using pattern analysis and context understanding
        
        Args:
            content: The input content to analyze
            
        Returns:
            List of GuardrailViolation objects for harmful content
        """
        violations = []
        content_lower = content.lower().strip()
        
        # Specific harmful instruction patterns
        harmful_instruction_patterns = [
            # Bomb/explosive making (enhanced patterns)
            (r'(?i)\b(?:how.*to.*(?:make|build|create|construct).*(?:bomb|explos|dynamite|pipe.*bomb|pressure.*cooker.*bomb|boom))\b', 
             "Instructions for creating explosive devices", "explosive_making"),
            
            # More bomb-related patterns
            (r'(?i)\b(?:create.*boom|make.*boom|build.*boom|boom.*making)\b',
             "Instructions for creating explosive devices", "explosive_making"),
             
            # Weapon creation/acquisition (enhanced)
            (r'(?i)\b(?:how.*to.*(?:make|build|get|buy|acquire|create).*(?:weapon|gun|knife|poison|toxin|firearm))\b',
             "Instructions for weapon creation or acquisition", "weapon_acquisition"),
             
            # Weapon making instructions
            (r'(?i)\b(?:instructions.*for.*making.*weapons|weapon.*manufacturing|making.*weapons)\b',
             "Instructions for weapon creation", "weapon_acquisition"),
            
            # Theft/financial crimes (enhanced)
            (r'(?i)\b(?:how.*to.*(?:steal|rob|hack|fraud|scam|take|get).*(?:credit.*card|money|cash|bank|wallet|purse|identity))\b',
             "Instructions for theft or financial crimes", "financial_theft"),
            
            # Breaking and entering (enhanced)
            (r'(?i)\b(?:how.*to.*(?:break.*in|pick.*lock|bypass.*security|disable.*alarm|enter.*house|break.*into))\b',
             "Instructions for breaking and entering", "breaking_entering"),
             
            # Break into patterns
            (r'(?i)\b(?:break.*into.*house|breaking.*into|enter.*illegally)\b',
             "Instructions for breaking and entering", "breaking_entering"),
            
            # Hacking/cybercrime
            (r'(?i)\b(?:how.*to.*(?:hack|crack|breach|infiltrate|ddos|attack).*(?:computer|system|network|website|server|account))\b',
             "Instructions for cybercrime activities", "cybercrime"),
            
            # Drug manufacturing
            (r'(?i)\b(?:how.*to.*(?:make|cook|manufacture|synthesize|grow).*(?:meth|cocaine|heroin|lsd|ecstasy|drugs))\b',
             "Instructions for drug manufacturing", "drug_manufacturing"),
            
            # Identity theft
            (r'(?i)\b(?:how.*to.*(?:steal|fake|forge|copy).*(?:identity|passport|license|id|documents|social.*security))\b',
             "Instructions for identity theft or document forgery", "identity_theft"),
            
            # Harmful pranks/attacks
            (r'(?i)\b(?:how.*to.*(?:poison|hurt|harm|injure|disable|incapacitate).*(?:someone|person|people))\b',
             "Instructions for harming others", "harm_others"),
             
            # Explosives making (broader patterns)
            (r'(?i)\b(?:how.*to.*make.*explosives|explosive.*making|create.*explosives)\b',
             "Instructions for creating explosive devices", "explosive_making"),
        ]
        
        for pattern, description, violation_category in harmful_instruction_patterns:
            if re.search(pattern, content):
                self.logger.error(f"🚨 [SECURITY] Harmful intent detected: {violation_category}")
                violations.append(GuardrailViolation(
                    violation_type=GuardrailViolationType.UNSAFE_CONTENT,
                    message=f"Request contains instructions for illegal/harmful activities: {description}",
                    suggested_action="This service is for data schema validation only. Please provide a legitimate domain name for database rule generation.",
                    confidence=0.95,
                    detected_pattern=violation_category
                ))
        
        # Check for suspicious question patterns that might be trying to bypass filters
        bypass_patterns = [
            (r'(?i)\b(?:hypothetically|theoretically|for.*research|academic.*purpose|just.*curious|asking.*friend)\b.*(?:illegal|harmful|dangerous)',
             "Potential attempt to bypass safety filters", "bypass_attempt"),
            
            (r'(?i)\b(?:not.*that.*i.*would|just.*wondering|purely.*educational|for.*story|creative.*writing)\b.*(?:steal|bomb|hack|kill)',
             "Suspicious disclaimer paired with harmful content", "suspicious_disclaimer"),
        ]
        
        for pattern, description, violation_category in bypass_patterns:
            if re.search(pattern, content):
                self.logger.warning(f"⚠️ [SECURITY] Bypass attempt detected: {violation_category}")
                violations.append(GuardrailViolation(
                    violation_type=GuardrailViolationType.UNSAFE_CONTENT,
                    message=f"Content appears to be attempting to bypass safety measures: {description}",
                    suggested_action="Please provide a straightforward database domain name for rule generation.",
                    confidence=0.85,
                    detected_pattern=violation_category
                ))
        
        return violations

    def validate_content_safety(self, content: str) -> List[GuardrailViolation]:
        """
        Check for unsafe content patterns with enhanced harmful intent detection
        
        Args:
            content: The input content to validate
            
        Returns:
            List of GuardrailViolation objects
        """
        violations = []
        
        self.logger.info(f" [GUARDRAIL] Checking content safety for {len(content)} chars")
        
        # 1. Advanced harmful intent detection (highest priority)
        harmful_intent_violations = self.detect_harmful_intent(content)
        violations.extend(harmful_intent_violations)
        
        # If we found harmful intent, that's the most serious violation - log and continue with other checks
        if harmful_intent_violations:
            self.logger.error(f" [SECURITY] CRITICAL: Harmful intent detected in user input!")
            for violation in harmful_intent_violations:
                self.logger.error(f"    {violation.detected_pattern}: {violation.message}")
        
        # 2. Check general unsafe content patterns
        for pattern_name, pattern in self.unsafe_patterns.items():
            matches = re.findall(pattern, content)
            if matches:
                self.logger.warning(f" [GUARDRAIL] Unsafe content detected: {pattern_name}")
                violations.append(GuardrailViolation(
                    violation_type=GuardrailViolationType.UNSAFE_CONTENT,
                    message=f"Content contains inappropriate {pattern_name.replace('_', ' ')}",
                    suggested_action="Please rephrase your request using professional, appropriate language",
                    confidence=0.9,
                    detected_pattern=pattern_name
                ))
        
        # 3. Check injection attempts
        for pattern_name, pattern in self.injection_patterns.items():
            matches = re.findall(pattern, content)
            if matches:
                self.logger.warning(f"❌ [GUARDRAIL] Injection attempt detected: {pattern_name}")
                violations.append(GuardrailViolation(
                    violation_type=GuardrailViolationType.INJECTION_ATTEMPT,
                    message=f"Content appears to contain {pattern_name.replace('_', ' ')} attempt",
                    suggested_action="Please provide a straightforward domain name for schema analysis",
                    confidence=0.95,
                    detected_pattern=pattern_name
                ))
        
        # 4. Check spam patterns
        for pattern_name, pattern in self.spam_patterns.items():
            matches = re.findall(pattern, content)
            if matches:
                self.logger.warning(f"❌ [GUARDRAIL] Spam pattern detected: {pattern_name}")
                violations.append(GuardrailViolation(
                    violation_type=GuardrailViolationType.SPAM_PATTERN,
                    message=f"Content contains suspicious {pattern_name.replace('_', ' ')} pattern",
                    suggested_action="Please provide a clean, simple domain name",
                    confidence=0.8,
                    detected_pattern=pattern_name
                ))
        
        if violations:
            self.logger.warning(f"❌ [GUARDRAIL] Found {len(violations)} content safety violations")
            # Log summary of violation types for monitoring
            violation_types = [v.detected_pattern for v in violations if v.detected_pattern]
            self.logger.warning(f"   Violation types: {', '.join(set(violation_types))}")
        else:
            self.logger.info(f"✅ [GUARDRAIL] Content passed all safety checks")
        
        return violations
    
    def validate_request_relevance(self, content: str, available_domains: List[str] = None) -> Optional[GuardrailViolation]:
        """
        Check if the request is relevant to data schema and policy domain
        
        Args:
            content: The input content to validate
            available_domains: List of available domains for context-aware validation
            
        Returns:
            GuardrailViolation if content seems irrelevant, None if relevant
        """
        self.logger.info(f" [GUARDRAIL] Checking request relevance")
        
        content_lower = content.lower().strip()
        
        # First, check if this looks like a domain name request
        # If we have available domains, check if the content could be a domain-related query
        if available_domains:
            # Import here to avoid circular imports
            from app.validation.semantic_domain_search import enhance_domain_validation
            
            # Check if this could be a valid domain or semantic match
            is_exact, suggested_domain, semantic_results = enhance_domain_validation(content, available_domains)
            
            if is_exact or suggested_domain or semantic_results.get("found_domains"):
                self.logger.info(f" [GUARDRAIL] Content appears to be domain-related: found semantic matches")
                return None  # This is clearly domain-related
        
        # Check for domain-related keywords (expanded to include common domain terms)
        expanded_domain_keywords = self.domain_related_keywords.union({
            'customer', 'product', 'order', 'user', 'account', 'profile', 'employee', 
            'staff', 'inventory', 'sales', 'finance', 'financial', 'marketing', 
            'support', 'service', 'transaction', 'payment', 'shipping', 'delivery'
        })
        
        found_keywords = [kw for kw in expanded_domain_keywords if kw in content_lower]
        
        # If content is very short BUT contains domain-related terms, it's likely valid
        if len(content.strip()) < 10:
            if found_keywords:
                self.logger.info(f" [GUARDRAIL] Short content but contains domain keywords: {found_keywords}")
                return None
            elif available_domains and any(domain_word in content_lower for domain_word in ['domain', 'schema', 'rules', 'suggest', 'table']):
                self.logger.info(f" [GUARDRAIL] Short content but contains domain-request terms")
                return None
        
        # More lenient check for medium-length content
        if len(content.strip()) < 30 and not found_keywords:
            # Check if it contains obvious non-domain content
            off_topic_indicators = ['weather', 'sports', 'cooking', 'movie', 'music', 'game', 'joke', 'story', 'hello', 'hi', 'how are you']
            if any(indicator in content_lower for indicator in off_topic_indicators):
                self.logger.warning(f" [GUARDRAIL] Request appears irrelevant: contains off-topic content")
                return GuardrailViolation(
                    violation_type=GuardrailViolationType.IRRELEVANT_REQUEST,
                    message="Request appears to be unrelated to data schema or policy validation",
                    suggested_action="Please provide a domain name for which you need data quality rules. This service generates validation rules for database schemas.",
                    confidence=0.8
                )
        
        # Check for completely off-topic content (only for longer content)
        if len(content.strip()) > 20:
            off_topic_patterns = [
                r'(?i)\b(?:weather|sports|cooking|recipe|movie|music|game|joke|story)\b',
                r'(?i)\b(?:hello|hi|how are you|what.*up|good morning|good evening)\b',
                r'(?i)\b(?:love|relationship|dating|marriage|family|friend)\b'
            ]
            
            for pattern in off_topic_patterns:
                if re.search(pattern, content):
                    self.logger.warning(f" [GUARDRAIL] Off-topic content detected in longer text")
                    return GuardrailViolation(
                        violation_type=GuardrailViolationType.IRRELEVANT_REQUEST,
                        message="Request appears to be unrelated to data schema validation",
                        suggested_action="This service is designed for generating data quality rules. Please provide a domain name from your database schema.",
                        confidence=0.8
                    )
        
        if found_keywords:
            self.logger.info(f" [GUARDRAIL] Request appears relevant: found {len(found_keywords)} domain keywords")
        else:
            self.logger.info(f" [GUARDRAIL] Request relevance unclear but not clearly irrelevant")
        
        return None
    
    def comprehensive_validate(self, domain: str, available_domains: List[str]) -> Tuple[bool, List[GuardrailViolation]]:
        """
        Run comprehensive validation on user input
        
        Args:
            domain: The domain input from user
            available_domains: List of available domains
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        self.logger.info(f" [GUARDRAIL] Starting comprehensive validation for domain: '{domain}'")
        
        violations = []
        
        # 1. Domain validation
        domain_violation = self.validate_domain_input(domain, available_domains)
        if domain_violation:
            violations.append(domain_violation)
        
        # 2. Content safety validation
        safety_violations = self.validate_content_safety(domain)
        violations.extend(safety_violations)
        
        # 3. Relevance validation (pass available_domains for context-aware validation)
        relevance_violation = self.validate_request_relevance(domain, available_domains)
        if relevance_violation:
            violations.append(relevance_violation)
        
        is_valid = len(violations) == 0
        
        if is_valid:
            self.logger.info(f" [GUARDRAIL] Comprehensive validation PASSED for domain: '{domain}'")
        else:
            self.logger.warning(f" [GUARDRAIL] Comprehensive validation FAILED: {len(violations)} violations")
            for i, violation in enumerate(violations, 1):
                self.logger.warning(f"   {i}. {violation.violation_type.value}: {violation.message}")
        
        return is_valid, violations


def create_guardrail_response(violations: List[GuardrailViolation]) -> Dict:
    """
    Create a structured error response for guardrail violations
    
    Args:
        violations: List of guardrail violations
        
    Returns:
        Structured error response dictionary
    """
    if not violations:
        return {"success": True}
    
    # Group violations by type
    grouped_violations = {}
    for violation in violations:
        vtype = violation.violation_type.value
        if vtype not in grouped_violations:
            grouped_violations[vtype] = []
        grouped_violations[vtype].append({
            "message": violation.message,
            "suggested_action": violation.suggested_action,
            "confidence": violation.confidence,
            "detected_pattern": violation.detected_pattern
        })
    
    return {
        "success": False,
        "error": "Input validation failed",
        "error_type": "guardrail_violation",
        "violations": grouped_violations,
        "total_violations": len(violations),
        "suggested_action": "Please review the violations and modify your request accordingly"
    }