#!/usr/bin/env python3
"""
Enhanced Prompt Configuration and Management
Centralized configuration for all AI prompts with quality controls
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class PromptComplexity(Enum):
    """Prompt complexity levels based on use case"""
    BASIC = "basic"
    STANDARD = "standard"
    ADVANCED = "advanced"
    EXPERT = "expert"


class IndustryDomain(Enum):
    """Industry-specific prompt variations"""
    GENERAL = "general"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    RETAIL = "retail"
    MANUFACTURING = "manufacturing"
    TECHNOLOGY = "technology"


@dataclass
class PromptConfig:
    """Configuration for prompt customization"""
    complexity: PromptComplexity = PromptComplexity.STANDARD
    industry: IndustryDomain = IndustryDomain.GENERAL
    include_reasoning: bool = True
    include_compliance: bool = True
    include_performance_guidance: bool = True
    max_output_tokens: int = 4000
    temperature: float = 0.1  # Low temperature for consistent, focused outputs


class EnhancedPromptManager:
    """Centralized management of enhanced prompts"""

    def __init__(self, config: PromptConfig = None):
        self.config = config or PromptConfig()
        self._industry_patterns = self._load_industry_patterns()
        self._compliance_requirements = self._load_compliance_requirements()

    def get_rule_generation_prompt(self, domain: str, data_schema: dict, gx_rules: list) -> str:
        """Get enhanced rule generation prompt with context awareness"""

        base_expertise = self._get_base_expertise_context()
        industry_context = self._get_industry_context(domain)
        compliance_context = self._get_compliance_context(domain)

        prompt = f"""{base_expertise}

**CONTEXT:**
Domain: {domain}
Industry Focus: {industry_context}
Schema Analysis: {len(data_schema)} columns detected
Available Rules: {len(gx_rules)} Great Expectations patterns

**SCHEMA DETAILS:**
{self._format_schema_for_analysis(data_schema)}

**AVAILABLE VALIDATION PATTERNS:**
{self._format_rules_for_analysis(gx_rules)}

**REASONING FRAMEWORK:**
{self._get_reasoning_framework()}

**VALIDATION STRATEGY:**
{self._get_validation_strategy(domain)}

{compliance_context}

**OUTPUT REQUIREMENTS:**
Return ONLY a valid JSON array with the following structure:
[
  {{
    "column": "column_name",
    "expectations": [
      {{
        "expectation_type": "expectation_name",
        "kwargs": {{"parameter": "value"}},
        "meta": {{
          "reasoning": "Business justification for this rule",
          "pii_classification": "none|personal|sensitive|restricted",
          "performance_impact": "low|medium|high",
          "compliance_requirement": "optional compliance standard"
        }}
      }}
    ]
  }}
]

**QUALITY REQUIREMENTS:**
- Each rule must have clear business justification
- PII fields must have appropriate protection patterns
- Performance impact must be considered for large datasets
- Output must be valid JSON without markdown formatting"""

        return prompt

    def get_schema_design_prompt(self, domain: str, config_params: dict) -> str:
        """Get enhanced schema design prompt with business intelligence"""

        base_expertise = self._get_schema_design_expertise()
        domain_patterns = self._get_domain_specific_patterns(domain)
        quality_standards = self._get_quality_standards()

        prompt = f"""{base_expertise}

**DOMAIN FOCUS:** {domain}
**INDUSTRY PATTERNS:** {domain_patterns}

**DESIGN REQUIREMENTS:**
- Generate {config_params.get('min_columns', 5)}-{config_params.get('max_columns', 15)} columns
- Use data types: {', '.join(config_params.get('supported_types', ['string', 'integer', 'float', 'date', 'boolean']))}
- Provide {config_params.get('min_samples', 3)} realistic sample values per column

**BUSINESS ARCHITECTURE FRAMEWORK:**
{self._get_business_architecture_framework()}

**COLUMN DESIGN PATTERNS:**
{self._get_column_design_patterns(domain)}

**QUALITY STANDARDS:**
{quality_standards}

**OUTPUT FORMAT:**
{config_params.get('format_instructions', 'Valid JSON schema format required')}"""

        return prompt

    def get_column_suggestion_prompt(self, domain: str) -> str:
        """Get enhanced column name suggestion prompt"""

        expertise = self._get_column_suggestion_expertise()
        domain_analysis = self._get_domain_analysis_framework(domain)

        prompt = f"""{expertise}

**DOMAIN:** {domain}

**DOMAIN ANALYSIS FRAMEWORK:**
{domain_analysis}

**BUSINESS INTELLIGENCE APPROACH:**
{self._get_business_intelligence_approach()}

**NAMING STANDARDS:**
{self._get_naming_standards()}

**OUTPUT REQUIREMENTS:**
Return ONLY a JSON array of 5-11 essential column names.
Example: ["primary_id", "created_date", "status", "category", "value_metric"]"""

        return prompt

    def _get_base_expertise_context(self) -> str:
        """Get base expertise context for rule generation"""
        return """You are an expert data governance and quality assurance specialist with 15+ years of experience in enterprise data validation. Your expertise includes:

- Great Expectations framework and advanced validation patterns
- Data quality best practices across Fortune 500 companies
- PII/sensitive data identification and GDPR/HIPAA compliance
- Regulatory requirements (SOX, PCI-DSS, CCPA)
- Performance-optimized validation strategies for big data
- Industry-specific data governance frameworks"""

    def _get_industry_context(self, domain: str) -> str:
        """Get industry-specific context"""
        industry_map = {
            "customer": "Customer Data Management & CRM",
            "financial": "Financial Services & Banking",
            "healthcare": "Healthcare & Life Sciences",
            "product": "Product Management & E-commerce",
            "employee": "Human Resources & Workforce Management",
        }
        return industry_map.get(domain.lower(), "General Business Domain")

    def _get_compliance_context(self, domain: str) -> str:
        """Get compliance requirements based on domain"""
        if not self.config.include_compliance:
            return ""

        compliance_map = {
            "customer": "GDPR Article 6 & 7 compliance for personal data, right to be forgotten",
            "financial": "SOX compliance for financial records, PCI-DSS for payment data",
            "healthcare": "HIPAA compliance for PHI, FDA 21 CFR Part 11 for clinical data",
            "employee": "Employment law compliance, equal opportunity data protection",
        }

        requirement = compliance_map.get(domain.lower(), "General data protection requirements")

        return f"""
**COMPLIANCE REQUIREMENTS:**
{requirement}
- Data minimization principles
- Audit trail requirements
- Data retention policies"""

    def _get_reasoning_framework(self) -> str:
        """Get structured reasoning framework"""
        if not self.config.include_reasoning:
            return "Apply standard validation patterns."

        return """For each column, systematically analyze:
1. **Data Type Validation:** Ensure type consistency and format compliance
2. **Business Logic:** Apply domain-specific business rules and constraints  \
3. **Data Quality:** Identify potential issues (nulls, duplicates, outliers)
4. **Security Classification:** Assess PII risk and protection requirements
5. **Performance Impact:** Consider validation cost for large-scale processing
6. **Regulatory Compliance:** Apply industry-specific compliance patterns"""

    def _get_validation_strategy(self, domain: str) -> str:
        """Get domain-specific validation strategy"""
        base_strategy = """
- **Primary Identifiers:** NOT NULL + UNIQUE constraints
- **Email Fields:** RFC 5322 format validation + NOT NULL
- **Phone Numbers:** International format support + optional validation
- **Dates:** ISO 8601 format + reasonable range constraints
- **Numeric Fields:** Type validation + business range constraints
- **Text Fields:** Length constraints + character set validation
- **PII Fields:** Enhanced protection + audit logging"""

        domain_specific = {
            "financial": """
- **Account Numbers:** Luhn algorithm validation where applicable
- **Currency Amounts:** Precision constraints + non-negative validation
- **Transaction Dates:** Business day validation + temporal consistency
- **Regulatory IDs:** Format validation per jurisdiction""",
            "healthcare": """
- **Patient IDs:** Format validation + check digit verification
- **Medical Codes:** ICD-10/CPT code validation + version consistency
- **Dates:** Clinical timeline validation + HIPAA retention compliance
- **Dosages:** Range validation + unit consistency""",
            "customer": """
- **Customer IDs:** Uniqueness + referential integrity
- **Contact Info:** Multi-channel validation + preference management
- **Behavioral Data:** Range validation + privacy classification
- **Segmentation:** Category validation + business rule consistency""",
        }

        specific = domain_specific.get(domain.lower(), "")
        return base_strategy + specific

    def _format_schema_for_analysis(self, data_schema: dict) -> str:
        """Format schema for prompt analysis"""
        formatted = []
        for col, details in data_schema.items():
            if col != "domain" and isinstance(details, dict):
                samples = details.get("sample_values", [])
                dtype = details.get("dtype", "unknown")
                formatted.append(f"- {col}: {dtype} (samples: {samples[:3]})")
        return "\n".join(formatted)

    def _format_rules_for_analysis(self, gx_rules: list) -> str:
        """Format available rules for prompt analysis"""
        if not gx_rules:
            return "- Using default Great Expectations rule set"

        formatted = []
        for rule in gx_rules[:10]:  # Limit to prevent token overflow
            if isinstance(rule, dict):
                name = rule.get("rule_name", "Unknown")
                desc = rule.get("description", "No description")
                formatted.append(f"- {name}: {desc}")
        return "\n".join(formatted)

    def _get_schema_design_expertise(self) -> str:
        """Get schema design expertise context"""
        return """You are a Senior Data Architect and Enterprise Schema Designer with deep expertise in:

- Business domain modeling across industries (Finance, Healthcare, Retail, Manufacturing)
- Data warehouse and data lake design patterns for scalability
- Regulatory compliance requirements (GDPR, HIPAA, PCI-DSS, SOX)
- Modern data governance frameworks and metadata management
- Real-world production data challenges and performance optimization

**DESIGN PHILOSOPHY:**
Create schemas that are business-aligned, compliance-ready, scalable, performance-optimized, and rich in metadata for data discovery and governance."""

    def _get_domain_specific_patterns(self, domain: str) -> str:
        """Get domain-specific design patterns"""
        patterns = self._industry_patterns.get(domain.lower(), self._industry_patterns.get("general", {}))

        formatted = []
        for category, columns in patterns.items():
            formatted.append(f"**{category.title()}:** {', '.join(columns)}")

        return "\n".join(formatted)

    def _get_quality_standards(self) -> str:
        """Get quality standards for schema design"""
        return """Each column must demonstrate:
- **Business Clarity:** Clear, descriptive names following conventions
- **Technical Precision:** Appropriate data types for content and scale
- **Realistic Samples:** Values that business users would recognize and trust
- **Governance Readiness:** Implicit consideration for validation and compliance
- **Integration Friendly:** Compatible with downstream systems and analytics"""

    def _load_industry_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Load industry-specific column patterns"""
        return {
            "customer": {
                "identifiers": ["customer_id", "account_number", "external_id"],
                "contact": ["email", "phone", "preferred_contact_method"],
                "temporal": ["registration_date", "last_login_date", "updated_timestamp"],
                "descriptive": ["first_name", "last_name", "status", "customer_type"],
                "behavioral": ["lifetime_value", "engagement_score", "preferences"],
                "location": ["address", "city", "postal_code", "country_code"],
            },
            "financial": {
                "identifiers": ["account_id", "transaction_id", "routing_number"],
                "monetary": ["balance", "transaction_amount", "credit_limit"],
                "temporal": ["transaction_date", "settlement_date", "maturity_date"],
                "descriptive": ["account_type", "transaction_type", "currency_code"],
                "regulatory": ["regulatory_code", "compliance_flag", "audit_trail"],
                "risk": ["risk_rating", "fraud_score", "compliance_status"],
            },
            "product": {
                "identifiers": ["product_id", "sku", "barcode", "model_number"],
                "descriptive": ["product_name", "category", "brand", "description"],
                "pricing": ["price", "cost", "margin", "discount_percent"],
                "inventory": ["quantity_on_hand", "reserved_quantity", "reorder_level"],
                "temporal": ["created_date", "last_updated", "discontinuation_date"],
                "supplier": ["supplier_id", "vendor_code", "lead_time_days"],
            },
            "healthcare": {
                "identifiers": ["patient_id", "medical_record_number", "visit_id"],
                "temporal": ["admission_date", "discharge_date", "appointment_time"],
                "clinical": ["diagnosis_code", "treatment_type", "medication"],
                "administrative": ["insurance_id", "provider_id", "facility_code"],
                "compliance": ["consent_status", "privacy_flag", "retention_date"],
                "outcomes": ["treatment_outcome", "satisfaction_score", "follow_up_required"],
            },
            "general": {
                "identifiers": ["id", "external_id", "reference_number"],
                "temporal": ["created_date", "updated_date", "effective_date"],
                "descriptive": ["name", "description", "category", "status"],
                "quantitative": ["amount", "quantity", "percentage", "score"],
                "metadata": ["source_system", "data_quality_score", "version"],
            },
        }

    def _load_compliance_requirements(self) -> Dict[str, List[str]]:
        """Load compliance requirements by domain"""
        return {
            "customer": ["GDPR Article 6", "CCPA compliance", "Right to be forgotten"],
            "financial": ["SOX compliance", "PCI-DSS", "Anti-money laundering"],
            "healthcare": ["HIPAA compliance", "FDA 21 CFR Part 11", "Clinical data integrity"],
            "employee": ["Employment law", "Equal opportunity", "Privacy protection"],
        }

    def _get_business_architecture_framework(self) -> str:
        """Get business architecture framework"""
        return """
**COLUMN CATEGORIES TO INCLUDE:**
1. **Identifiers:** Primary keys, foreign keys, business reference numbers
2. **Temporal:** Created/updated timestamps, effective dates, expiry dates  
3. **Descriptive:** Names, descriptions, categories, statuses, classifications
4. **Quantitative:** Amounts, counts, rates, percentages, scores, metrics
5. **Contact/Location:** Addresses, phone, email (with PII considerations)
6. **Behavioral:** Flags, preferences, activity indicators, interaction data
7. **Metadata:** Data lineage, quality indicators, source system references"""

    def _get_column_design_patterns(self, domain: str) -> str:
        """Get specific column design patterns for domain"""
        patterns = self._industry_patterns.get(domain.lower(), {})

        formatted = []
        for category, columns in patterns.items():
            example_cols = ", ".join(columns[:5])  # Limit examples
            formatted.append(f"- **{category.title()}:** {example_cols}")

        return "\n".join(formatted) if formatted else "- Apply general business patterns"

    def _get_column_suggestion_expertise(self) -> str:
        """Get column suggestion expertise"""
        return """You are a Business Intelligence Architect specializing in domain-driven data modeling. Your expertise includes:

- Cross-industry data modeling patterns and best practices
- Business process analysis and data flow mapping
- Regulatory and compliance data requirements across sectors  
- Data governance and metadata management frameworks
- Enterprise data integration and interoperability standards"""

    def _get_domain_analysis_framework(self, domain: str) -> str:
        """Get domain analysis framework"""
        return f"""
**ANALYSIS FRAMEWORK for {domain.upper()}:**
1. **Core Entities:** What are the primary business objects and their attributes?
2. **Key Processes:** What business workflows and transactions involve this data?
3. **Regulatory Needs:** What compliance and audit data is required?
4. **Operational Metrics:** What KPIs and performance indicators are tracked?
5. **Integration Points:** What external system data and references are needed?
6. **Temporal Aspects:** What time-based tracking and lifecycle data is essential?"""

    def _get_business_intelligence_approach(self) -> str:
        """Get business intelligence approach"""
        return """
**BI-DRIVEN COLUMN SELECTION:**
- Prioritize columns that enable key business questions and analytics
- Include dimensions for segmentation, filtering, and grouping
- Ensure measures and metrics for quantitative analysis
- Consider temporal dimensions for trend analysis and reporting
- Include categorical fields for business intelligence drill-down capabilities"""

    def _get_naming_standards(self) -> str:
        """Get naming standards"""
        return """
**NAMING CONVENTIONS:**
- Use lowercase_with_underscores for technical consistency
- Choose business-friendly, self-documenting names that stakeholders understand
- Prioritize commonly understood terminology over technical jargon
- Include essential temporal fields (created_date, updated_date)
- Always include a primary identifier field
- Use consistent patterns: *_id for identifiers, *_date for dates, *_flag for booleans"""


# Global prompt manager instance
_prompt_manager = None


def get_prompt_manager(config: PromptConfig = None) -> EnhancedPromptManager:
    """Get global prompt manager instance"""
    global _prompt_manager
    if _prompt_manager is None or config is not None:
        _prompt_manager = EnhancedPromptManager(config)
    return _prompt_manager


def get_enhanced_rule_prompt(domain: str, data_schema: dict, gx_rules: list) -> str:
    """Quick access to enhanced rule generation prompt"""
    manager = get_prompt_manager()
    return manager.get_rule_generation_prompt(domain, data_schema, gx_rules)


def get_enhanced_schema_prompt(domain: str, config_params: dict) -> str:
    """Quick access to enhanced schema design prompt"""
    manager = get_prompt_manager()
    return manager.get_schema_design_prompt(domain, config_params)


def get_enhanced_column_prompt(domain: str) -> str:
    """Quick access to enhanced column suggestion prompt"""
    manager = get_prompt_manager()
    return manager.get_column_suggestion_prompt(domain)