"""
Specialized logger for prompt management and generation
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class PromptLogger:
    @staticmethod
    def log_prompt_generation(prompt_type: str, domain: str, context: Optional[Dict[str, Any]] = None):
        """Log prompt generation with context"""
        logger.info(f"Generating {prompt_type} prompt for domain: {domain}")
        if context:
            logger.debug(f"Context for {prompt_type} prompt:")
            for key, value in context.items():
                if isinstance(value, (list, dict)):
                    logger.debug(f"- {key}: {type(value).__name__} with {len(value)} items")
                else:
                    logger.debug(f"- {key}: {value}")

    @staticmethod
    def log_prompt_content(prompt_type: str, prompt: str, section_markers: bool = True):
        """Log the content of a prompt with optional section markers"""
        if section_markers:
            logger.info(f"=== {prompt_type.upper()} PROMPT START ===")
            logger.info(prompt)
            logger.info(f"=== {prompt_type.upper()} PROMPT END ===")
        else:
            logger.info(f"{prompt_type} prompt content: {prompt[:200]}...")
        
        logger.info(f"Prompt statistics:")
        logger.info(f"- Total length: {len(prompt)} characters")
        logger.info(f"- Section count: {prompt.count('**')//2}")  # Count markdown sections

    @staticmethod
    def log_prompt_components(prompt_type: str, components: Dict[str, str]):
        """Log individual components of a prompt"""
        logger.info(f"Components for {prompt_type} prompt:")
        for component, content in components.items():
            logger.debug(f"- {component}: {len(content)} characters")
            if len(content) > 100:
                logger.debug(f"  Preview: {content[:100]}...")

    @staticmethod
    def log_prompt_combination(base_type: str, enhancement_type: str):
        """Log when different prompt types are being combined"""
        logger.info(f"Combining {base_type} prompt with {enhancement_type}")

    @staticmethod
    def log_prompt_error(prompt_type: str, error: Exception):
        """Log errors in prompt generation or processing"""
        logger.error(f"Error in {prompt_type} prompt: {str(error)}", exc_info=True)