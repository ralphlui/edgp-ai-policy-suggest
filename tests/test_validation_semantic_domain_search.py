"""
Comprehensive tests for app.validation.semantic_domain_search module
"""

import pytest
from unittest.mock import patch, MagicMock, Mock
import json
from app.validation.semantic_domain_search import (
    SemanticDomainSearch,
    DomainMatch,
    enhance_domain_validation
)


class TestDomainMatch:
    """Test DomainMatch dataclass"""
    
    def test_domain_match_creation(self):
        """Test creating a DomainMatch"""
        match = DomainMatch(
            domain="customer",
            score=0.95,
            match_type="llm_semantic",
            explanation="High confidence match"
        )
        
        assert match.domain == "customer"
        assert match.score == 0.95
        assert match.match_type == "llm_semantic"
        assert match.explanation == "High confidence match"


class TestSemanticDomainSearch:
    """Test SemanticDomainSearch class"""
    
    @pytest.fixture
    def search(self):
        """Create a SemanticDomainSearch instance"""
        return SemanticDomainSearch()
    
    @pytest.fixture
    def sample_domains(self):
        """Sample domains for testing"""
        return ["customer", "product", "order", "employee", "financial", "inventory", "analytics"]
    
    def test_init(self, search):
        """Test initialization"""
        assert search.logger is not None
        assert search._llm is None
        assert "customer" in search.domain_keywords
        assert "product" in search.domain_keywords
    
    @patch('app.validation.semantic_domain_search.require_openai_api_key')
    @patch('app.validation.semantic_domain_search.ChatOpenAI')
    def test_llm_property_initialization(self, mock_chat_openai, mock_require_key, search):
        """Test LLM lazy initialization"""
        mock_require_key.return_value = "test-key"
        mock_llm_instance = MagicMock()
        mock_chat_openai.return_value = mock_llm_instance
        
        # First access should initialize
        llm = search.llm
        assert llm == mock_llm_instance
        mock_chat_openai.assert_called_once()
        
        # Second access should use cached instance
        llm2 = search.llm
        assert llm2 == mock_llm_instance
        assert mock_chat_openai.call_count == 1
    
    @patch('app.validation.semantic_domain_search.require_openai_api_key')
    def test_llm_property_initialization_failure(self, mock_require_key, search):
        """Test LLM initialization failure"""
        mock_require_key.side_effect = Exception("API key error")
        
        llm = search.llm
        assert llm is None
    
    def test_analyze_query_with_llm_no_llm(self, search, sample_domains):
        """Test analyze_query_with_llm when LLM is not available"""
        # Mock the llm property to return None
        with patch.object(type(search), 'llm', new_callable=lambda: property(lambda self: None)):
            result = search.analyze_query_with_llm("customer data", sample_domains)
            
            assert result["success"] is False
            assert "LLM not available" in result["error"]
    
    @patch.object(SemanticDomainSearch, 'llm')
    def test_analyze_query_with_llm_success(self, mock_llm_property, search, sample_domains):
        """Test successful LLM analysis"""
        mock_llm = MagicMock()
        mock_llm_property.__get__ = MagicMock(return_value=mock_llm)
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "matches": [
                {
                    "domain": "customer",
                    "confidence": 0.95,
                    "reasoning": "Exact semantic match for customer data"
                }
            ],
            "query_intent": "Looking for customer domain",
            "keywords_extracted": ["customer", "data"]
        })
        mock_llm.invoke.return_value = mock_response
        
        result = search.analyze_query_with_llm("customer data", sample_domains)
        
        assert result["success"] is True
        assert result["method"] == "llm"
        assert len(result["result"]["matches"]) == 1
        assert result["result"]["matches"][0]["domain"] == "customer"
    
    @patch.object(SemanticDomainSearch, 'llm')
    def test_analyze_query_with_llm_invalid_json(self, mock_llm_property, search, sample_domains):
        """Test LLM analysis with invalid JSON response"""
        mock_llm = MagicMock()
        mock_llm_property.__get__ = MagicMock(return_value=mock_llm)
        
        mock_response = MagicMock()
        mock_response.content = "invalid json response"
        mock_llm.invoke.return_value = mock_response
        
        result = search.analyze_query_with_llm("customer", sample_domains)
        
        assert result["success"] is False
        assert "Invalid JSON response" in result["error"]
    
    @patch.object(SemanticDomainSearch, 'llm')
    def test_analyze_query_with_llm_exception(self, mock_llm_property, search, sample_domains):
        """Test LLM analysis with exception"""
        mock_llm = MagicMock()
        mock_llm_property.__get__ = MagicMock(return_value=mock_llm)
        mock_llm.invoke.side_effect = Exception("LLM error")
        
        result = search.analyze_query_with_llm("customer", sample_domains)
        
        assert result["success"] is False
        assert "LLM error" in result["error"]
    
    @patch.object(SemanticDomainSearch, 'analyze_query_with_llm')
    def test_search_domains_with_llm_success(self, mock_analyze, search, sample_domains):
        """Test successful LLM domain search"""
        mock_analyze.return_value = {
            "success": True,
            "result": {
                "matches": [
                    {"domain": "customer", "confidence": 0.95, "reasoning": "Exact match"},
                    {"domain": "product", "confidence": 0.8, "reasoning": "Related domain"}
                ]
            }
        }
        
        matches = search.search_domains_with_llm("customer info", sample_domains)
        
        assert len(matches) == 2
        assert matches[0].domain == "customer"
        assert matches[0].score == 0.95
        assert matches[0].match_type == "llm_semantic"
        assert matches[1].domain == "product"
        assert matches[1].score == 0.8
    
    @patch.object(SemanticDomainSearch, 'analyze_query_with_llm')
    def test_search_domains_with_llm_domain_not_in_available(self, mock_analyze, search, sample_domains):
        """Test LLM search when suggested domain is not in available list"""
        mock_analyze.return_value = {
            "success": True,
            "result": {
                "matches": [
                    {"domain": "nonexistent", "confidence": 0.95, "reasoning": "Not in list"}
                ]
            }
        }
        
        matches = search.search_domains_with_llm("customer", sample_domains)
        
        assert len(matches) == 0
    
    @patch.object(SemanticDomainSearch, 'analyze_query_with_llm')
    @patch.object(SemanticDomainSearch, 'search_domains_algorithmic')
    def test_search_domains_with_llm_fallback(self, mock_algo, mock_analyze, search, sample_domains):
        """Test fallback to algorithmic search when LLM fails"""
        mock_analyze.return_value = {"success": False, "error": "LLM failed"}
        mock_algo.return_value = [
            DomainMatch("customer", 0.8, "algorithmic", "Fuzzy match")
        ]
        
        matches = search.search_domains_with_llm("customer", sample_domains)
        
        assert len(matches) == 1
        assert matches[0].domain == "customer"
        mock_algo.assert_called_once()
    
    def test_extract_keywords(self, search):
        """Test keyword extraction"""
        query = "I need rules for customer data validation"
        keywords = search.extract_keywords(query)
        
        assert "customer" in keywords
        assert "data" in keywords
        assert "validation" in keywords
        assert "need" not in keywords  # Should be filtered as stop word
        assert "for" not in keywords  # Should be filtered as stop word
    
    def test_find_keyword_matches(self, search, sample_domains):
        """Test finding keyword matches"""
        keywords = ["customer", "data"]
        matches = search.find_keyword_matches(keywords, sample_domains)
        
        customer_matches = [m for m in matches if m.domain == "customer"]
        assert len(customer_matches) > 0
        assert customer_matches[0].match_type == "keyword"
    
    def test_find_fuzzy_matches(self, search, sample_domains):
        """Test fuzzy string matching"""
        matches = search.find_fuzzy_matches("custmer", sample_domains)  # Typo in "customer"
        
        customer_matches = [m for m in matches if m.domain == "customer"]
        assert len(customer_matches) > 0
        assert customer_matches[0].match_type == "fuzzy"
        assert customer_matches[0].score > 0.4
    
    def test_find_partial_matches(self, search, sample_domains):
        """Test partial substring matching"""
        matches = search.find_partial_matches("cust data", sample_domains)
        
        customer_matches = [m for m in matches if m.domain == "customer"]
        assert len(customer_matches) > 0
        assert customer_matches[0].match_type == "partial"
    
    def test_search_domains_algorithmic(self, search, sample_domains):
        """Test algorithmic domain search"""
        matches = search.search_domains_algorithmic("customer information", sample_domains)
        
        assert len(matches) > 0
        customer_matches = [m for m in matches if m.domain == "customer"]
        assert len(customer_matches) > 0
    
    def test_search_domains_algorithmic_deduplication(self, search, sample_domains):
        """Test that algorithmic search deduplicates and merges scores"""
        # This query should match "customer" through multiple methods
        matches = search.search_domains_algorithmic("customer", sample_domains)
        
        customer_matches = [m for m in matches if m.domain == "customer"]
        assert len(customer_matches) == 1  # Should be deduplicated
        
        # Check that match types are combined
        if customer_matches:
            match_type = customer_matches[0].match_type
            assert "+" in match_type or len(match_type) > 5  # Combined or single method
    
    @patch.object(SemanticDomainSearch, 'search_domains_with_llm')
    def test_search_domains_delegates_to_llm(self, mock_llm_search, search, sample_domains):
        """Test that search_domains delegates to LLM method"""
        expected_matches = [DomainMatch("customer", 0.9, "llm", "Test")]
        mock_llm_search.return_value = expected_matches
        
        matches = search.search_domains("customer", sample_domains)
        
        assert matches == expected_matches
        mock_llm_search.assert_called_once_with("customer", sample_domains, 5)
    
    def test_get_search_suggestions_no_matches(self, search, sample_domains):
        """Test search suggestions when no matches found"""
        with patch.object(search, 'search_domains', return_value=[]):
            suggestions = search.get_search_suggestions("xyz", sample_domains)
            
            assert suggestions["found_domains"] is False
            assert "No matching domains found" in suggestions["message"]
            assert "suggestions" in suggestions
            assert "available_keywords" in suggestions
    
    def test_get_search_suggestions_high_confidence(self, search, sample_domains):
        """Test search suggestions with high confidence matches"""
        high_conf_match = DomainMatch("customer", 0.9, "llm_semantic", "Perfect match")
        
        with patch.object(search, 'search_domains', return_value=[high_conf_match]):
            suggestions = search.get_search_suggestions("customer", sample_domains)
            
            assert suggestions["found_domains"] is True
            assert "high_confidence_matches" in suggestions
            assert len(suggestions["high_confidence_matches"]) == 1
            assert suggestions["high_confidence_matches"][0]["domain"] == "customer"
            assert "Try using domain: 'customer'" in suggestions["recommended_action"]
    
    def test_get_search_suggestions_medium_confidence(self, search, sample_domains):
        """Test search suggestions with medium confidence matches"""
        med_conf_match = DomainMatch("customer", 0.7, "keyword", "Keyword match")
        
        with patch.object(search, 'search_domains', return_value=[med_conf_match]):
            suggestions = search.get_search_suggestions("cust", sample_domains)
            
            assert suggestions["found_domains"] is True
            assert "possible_matches" in suggestions
            assert len(suggestions["possible_matches"]) == 1
            assert suggestions["possible_matches"][0]["domain"] == "customer"
    
    def test_get_search_suggestions_low_confidence_only(self, search, sample_domains):
        """Test search suggestions with only low confidence matches"""
        low_conf_match = DomainMatch("customer", 0.3, "fuzzy", "Weak match")
        
        with patch.object(search, 'search_domains', return_value=[low_conf_match]):
            suggestions = search.get_search_suggestions("xyz", sample_domains)
            
            assert suggestions["found_domains"] is True
            assert "weak_matches" in suggestions
            assert len(suggestions["weak_matches"]) == 1
            assert suggestions["weak_matches"][0]["domain"] == "customer"
    
    def test_get_search_suggestions_mixed_confidence(self, search, sample_domains):
        """Test search suggestions with mixed confidence levels"""
        matches = [
            DomainMatch("customer", 0.95, "llm_semantic", "Perfect match"),
            DomainMatch("product", 0.75, "keyword", "Good match"),
            DomainMatch("order", 0.4, "fuzzy", "Weak match")
        ]
        
        with patch.object(search, 'search_domains', return_value=matches):
            suggestions = search.get_search_suggestions("customer stuff", sample_domains)
            
            assert suggestions["found_domains"] is True
            assert "high_confidence_matches" in suggestions
            assert "possible_matches" in suggestions
            assert "weak_matches" not in suggestions  # Should not show weak when we have better
            assert suggestions["total_matches"] == 3
    
    def test_get_search_suggestions_method_detection(self, search, sample_domains):
        """Test that method is correctly detected in suggestions"""
        llm_match = DomainMatch("customer", 0.9, "llm_semantic", "LLM match")
        
        with patch.object(search, 'search_domains', return_value=[llm_match]):
            suggestions = search.get_search_suggestions("customer", sample_domains)
            
            assert suggestions["method"] == "llm_semantic_analysis"
        
        algo_match = DomainMatch("customer", 0.9, "keyword", "Keyword match")
        
        with patch.object(search, 'search_domains', return_value=[algo_match]):
            suggestions = search.get_search_suggestions("customer", sample_domains)
            
            assert suggestions["method"] == "algorithmic_fallback"


class TestEnhanceDomainValidation:
    """Test enhance_domain_validation function"""
    
    @pytest.fixture
    def sample_domains(self):
        return ["customer", "product", "order", "employee"]
    
    def test_enhance_domain_validation_exact_match(self, sample_domains):
        """Test exact domain match"""
        is_exact, suggested, results = enhance_domain_validation("customer", sample_domains)
        
        assert is_exact is True
        assert suggested == "customer"
        assert results["match_type"] == "exact"
    
    def test_enhance_domain_validation_case_insensitive(self, sample_domains):
        """Test case insensitive exact match"""
        is_exact, suggested, results = enhance_domain_validation("CUSTOMER", sample_domains)
        
        assert is_exact is True
        assert suggested == "customer"
        assert results["match_type"] == "exact"
    
    @patch('app.validation.semantic_domain_search.SemanticDomainSearch')
    def test_enhance_domain_validation_high_confidence_semantic(self, mock_search_class, sample_domains):
        """Test high confidence semantic match"""
        mock_search = MagicMock()
        mock_search_class.return_value = mock_search
        mock_search.get_search_suggestions.return_value = {
            "found_domains": True,
            "high_confidence_matches": [
                {"domain": "customer", "confidence": "85%", "reason": "Semantic match"}
            ]
        }
        
        is_exact, suggested, results = enhance_domain_validation("client", sample_domains)
        
        assert is_exact is False
        assert suggested == "customer"
        assert results["numeric_confidence"] == 0.85
        assert results["confidence_threshold_met"] is True
        assert results["llm_powered"] is True
    
    @patch('app.validation.semantic_domain_search.SemanticDomainSearch')
    def test_enhance_domain_validation_medium_confidence_accepted(self, mock_search_class, sample_domains):
        """Test medium confidence match that meets threshold"""
        mock_search = MagicMock()
        mock_search_class.return_value = mock_search
        mock_search.get_search_suggestions.return_value = {
            "found_domains": True,
            "possible_matches": [
                {"domain": "customer", "confidence": "80%", "reason": "Good match"}
            ]
        }
        
        is_exact, suggested, results = enhance_domain_validation("client", sample_domains)
        
        assert is_exact is False
        assert suggested == "customer"
        assert results["numeric_confidence"] == 0.80
        assert results["confidence_threshold_met"] is True
    
    @patch('app.validation.semantic_domain_search.SemanticDomainSearch')
    def test_enhance_domain_validation_medium_confidence_rejected(self, mock_search_class, sample_domains):
        """Test medium confidence match below threshold"""
        mock_search = MagicMock()
        mock_search_class.return_value = mock_search
        mock_search.get_search_suggestions.return_value = {
            "found_domains": True,
            "possible_matches": [
                {"domain": "customer", "confidence": "70%", "reason": "Weak match"}
            ]
        }
        
        is_exact, suggested, results = enhance_domain_validation("xyz", sample_domains)
        
        assert is_exact is False
        assert suggested is None
        # Results should have llm_powered but it might not be set to True if method doesn't start with llm
        assert "llm_powered" in results
    
    @patch('app.validation.semantic_domain_search.SemanticDomainSearch')
    def test_enhance_domain_validation_no_matches(self, mock_search_class, sample_domains):
        """Test when no matches are found"""
        mock_search = MagicMock()
        mock_search_class.return_value = mock_search
        mock_search.get_search_suggestions.return_value = {
            "found_domains": False,
            "method": "llm_semantic_analysis"
        }
        
        is_exact, suggested, results = enhance_domain_validation("nonexistent", sample_domains)
        
        assert is_exact is False
        assert suggested is None
        assert results["llm_powered"] is True
    
    @patch('app.validation.semantic_domain_search.SemanticDomainSearch')
    def test_enhance_domain_validation_confidence_parsing_error(self, mock_search_class, sample_domains):
        """Test confidence parsing with invalid format"""
        mock_search = MagicMock()
        mock_search_class.return_value = mock_search
        mock_search.get_search_suggestions.return_value = {
            "found_domains": True,
            "high_confidence_matches": [
                {"domain": "customer", "confidence": "invalid", "reason": "Match"}
            ]
        }
        
        is_exact, suggested, results = enhance_domain_validation("client", sample_domains)
        
        assert is_exact is False
        assert suggested == "customer"
        assert results["numeric_confidence"] == 0.8  # Default fallback


class TestSemanticDomainSearchEdgeCases:
    """Test edge cases and error conditions"""
    
    @pytest.fixture
    def search(self):
        return SemanticDomainSearch()
    
    def test_search_domains_empty_query(self, search):
        """Test search with empty query"""
        matches = search.search_domains("", ["customer", "product"])
        assert len(matches) == 0
    
    def test_search_domains_empty_domains(self, search):
        """Test search with empty domain list"""
        matches = search.search_domains("customer", [])
        assert len(matches) == 0
    
    def test_extract_keywords_empty_query(self, search):
        """Test keyword extraction with empty query"""
        keywords = search.extract_keywords("")
        assert len(keywords) == 0
    
    def test_extract_keywords_only_stop_words(self, search):
        """Test keyword extraction with only stop words"""
        keywords = search.extract_keywords("the for a me")
        assert len(keywords) == 0
    
    def test_find_fuzzy_matches_low_threshold(self, search):
        """Test fuzzy matching with very different strings"""
        matches = search.find_fuzzy_matches("xyz", ["customer", "product"])
        # Should have no matches or very low scores
        assert all(m.score < 0.5 for m in matches)
    
    def test_domain_keywords_coverage(self, search):
        """Test that domain keywords contain expected categories"""
        expected_categories = [
            'customer', 'product', 'order', 'employee', 
            'finance', 'inventory', 'analytics', 'marketing'
        ]
        
        for category in expected_categories:
            assert category in search.domain_keywords
            assert len(search.domain_keywords[category]) > 0
    
    def test_algorithmic_search_max_results(self, search):
        """Test that algorithmic search respects max_results parameter"""
        # Create a query that would match many domains
        domains = [f"customer_{i}" for i in range(20)]
        matches = search.search_domains_algorithmic("customer", domains, max_results=3)
        
        assert len(matches) <= 3


class TestDomainMatchSorting:
    """Test domain match sorting and scoring"""
    
    def test_domain_match_sorting(self):
        """Test that domain matches are sorted by score"""
        matches = [
            DomainMatch("product", 0.7, "keyword", "Medium match"),
            DomainMatch("customer", 0.9, "llm", "High match"),
            DomainMatch("order", 0.5, "fuzzy", "Low match")
        ]
        
        sorted_matches = sorted(matches, key=lambda x: x.score, reverse=True)
        
        assert sorted_matches[0].domain == "customer"
        assert sorted_matches[1].domain == "product"
        assert sorted_matches[2].domain == "order"
    
    def test_score_normalization(self):
        """Test that scores are properly normalized"""
        search = SemanticDomainSearch()
        
        # Test with keyword matches - should normalize based on number of keywords
        keywords = ["customer", "data"]
        matches = search.find_keyword_matches(keywords, ["customer"])
        
        if matches:
            assert 0.0 <= matches[0].score <= 1.0


class TestLLMPromptConstruction:
    """Test LLM prompt construction and validation"""
    
    @pytest.fixture
    def search(self):
        return SemanticDomainSearch()
    
    @patch.object(SemanticDomainSearch, 'llm')
    def test_analyze_query_prompt_structure(self, mock_llm_property, search):
        """Test that LLM prompts are properly structured"""
        mock_llm = MagicMock()
        mock_llm_property.__get__ = MagicMock(return_value=mock_llm)
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({"matches": [], "query_intent": "", "keywords_extracted": []})
        mock_llm.invoke.return_value = mock_response
        
        search.analyze_query_with_llm("customer", ["customer", "product"])
        
        # Verify invoke was called with messages
        mock_llm.invoke.assert_called_once()
        messages = mock_llm.invoke.call_args[0][0]
        
        assert len(messages) == 2  # System and human message
        assert any("expert" in str(msg.content).lower() for msg in messages)
        assert any("customer" in str(msg.content) for msg in messages)