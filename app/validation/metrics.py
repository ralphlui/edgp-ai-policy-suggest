"""
LLM Validation Metrics and Monitoring

This module provides metrics collection, monitoring, and reporting
for LLM validation performance and results.
"""

import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from threading import Lock

from app.validation.llm_validator import ValidationResult, ValidationSeverity

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetric:
    """Single validation metric record"""
    timestamp: datetime
    domain: str
    response_type: str
    is_valid: bool
    confidence_score: float
    issue_count: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    validation_time_ms: float
    auto_corrected: bool = False


class ValidationMetricsCollector:
    """Collects and aggregates validation metrics"""
    
    def __init__(self, max_metrics: int = 10000):
        """
        Initialize metrics collector
        
        Args:
            max_metrics: Maximum number of metrics to keep in memory
        """
        self.max_metrics = max_metrics
        self.metrics: List[ValidationMetric] = []
        self._lock = Lock()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def record_validation(self, 
                         domain: str,
                         response_type: str,
                         validation_result: ValidationResult,
                         validation_time_ms: float) -> None:
        """
        Record a validation result
        
        Args:
            domain: Domain being validated
            response_type: Type of response (schema, rule, etc.)
            validation_result: ValidationResult instance
            validation_time_ms: Time taken for validation in milliseconds
        """
        # Enhanced logging for metrics recording
        logger.info(f"📊 [METRICS] Recording validation metric for domain: {domain}, type: {response_type}")
        logger.debug(f"📈 [METRICS] Valid: {validation_result.is_valid}, "
                    f"Confidence: {validation_result.confidence_score:.2f}, "
                    f"Time: {validation_time_ms:.1f}ms")
        
        # Count issues by severity
        issue_counts = Counter(issue.severity for issue in validation_result.issues)
        
        # Log issue breakdown
        if validation_result.issues:
            logger.info(f"🔍 [METRICS] Issues breakdown - Critical: {issue_counts.get(ValidationSeverity.CRITICAL, 0)}, "
                       f"High: {issue_counts.get(ValidationSeverity.HIGH, 0)}, "
                       f"Medium: {issue_counts.get(ValidationSeverity.MEDIUM, 0)}, "
                       f"Low: {issue_counts.get(ValidationSeverity.LOW, 0)}")
        
        metric = ValidationMetric(
            timestamp=datetime.now(),
            domain=domain,
            response_type=response_type,
            is_valid=validation_result.is_valid,
            confidence_score=validation_result.confidence_score,
            issue_count=len(validation_result.issues),
            critical_issues=issue_counts.get(ValidationSeverity.CRITICAL, 0),
            high_issues=issue_counts.get(ValidationSeverity.HIGH, 0),
            medium_issues=issue_counts.get(ValidationSeverity.MEDIUM, 0),
            low_issues=issue_counts.get(ValidationSeverity.LOW, 0),
            validation_time_ms=validation_time_ms,
            auto_corrected=validation_result.corrected_data is not None
        )
        
        with self._lock:
            old_count = len(self.metrics)
            self.metrics.append(metric)
            
            # Enhanced logging for metrics storage
            logger.debug(f"💾 [METRICS] Stored metric - Total metrics: {len(self.metrics)}")
            
            # Log performance thresholds
            if validation_time_ms > 1000:  # 1 second
                logger.warning(f"⏰ [METRICS] Slow validation detected: {validation_time_ms:.1f}ms > 1000ms")
            elif validation_time_ms > 500:  # 500ms
                logger.info(f"⚡ [METRICS] Moderate validation time: {validation_time_ms:.1f}ms")
            
            # Log confidence score concerns
            if validation_result.confidence_score < 0.7:
                logger.warning(f"🎯 [METRICS] Low confidence score: {validation_result.confidence_score:.2f} < 0.7")
            
            # Keep only the most recent metrics
            if len(self.metrics) > self.max_metrics:
                removed_count = len(self.metrics) - self.max_metrics
                self.metrics = self.metrics[-self.max_metrics:]
                logger.debug(f"🗑️ [METRICS] Pruned {removed_count} old metrics, kept {len(self.metrics)}")
        
        # Enhanced final logging
        logger.info(f"✅ [METRICS] Metric recorded successfully - "
                   f"Domain: {domain}, Type: {response_type}, "
                   f"Valid: {validation_result.is_valid}, "
                   f"Confidence: {validation_result.confidence_score:.3f}, "
                   f"Issues: {len(validation_result.issues)}, "
                   f"Time: {validation_time_ms:.1f}ms")
        
        self.logger.debug(f"Recorded validation metric for {domain}/{response_type}: "
                         f"valid={validation_result.is_valid}, "
                         f"confidence={validation_result.confidence_score:.3f}, "
                         f"issues={len(validation_result.issues)}")
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get summary of validation metrics for the specified time period
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dictionary with metrics summary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {
                "period_hours": hours,
                "total_validations": 0,
                "success_rate": 0.0,
                "avg_confidence": 0.0,
                "avg_validation_time_ms": 0.0
            }
        
        total_validations = len(recent_metrics)
        successful_validations = sum(1 for m in recent_metrics if m.is_valid)
        success_rate = successful_validations / total_validations if total_validations > 0 else 0.0
        
        avg_confidence = sum(m.confidence_score for m in recent_metrics) / total_validations
        avg_validation_time = sum(m.validation_time_ms for m in recent_metrics) / total_validations
        
        # Group by domain and response type
        domain_stats = defaultdict(lambda: {"count": 0, "success_count": 0, "avg_confidence": 0.0})
        response_type_stats = defaultdict(lambda: {"count": 0, "success_count": 0, "avg_confidence": 0.0})
        
        for metric in recent_metrics:
            # Domain stats
            domain_stats[metric.domain]["count"] += 1
            if metric.is_valid:
                domain_stats[metric.domain]["success_count"] += 1
            domain_stats[metric.domain]["avg_confidence"] += metric.confidence_score
            
            # Response type stats
            response_type_stats[metric.response_type]["count"] += 1
            if metric.is_valid:
                response_type_stats[metric.response_type]["success_count"] += 1
            response_type_stats[metric.response_type]["avg_confidence"] += metric.confidence_score
        
        # Calculate averages
        for stats in domain_stats.values():
            stats["success_rate"] = stats["success_count"] / stats["count"] if stats["count"] > 0 else 0.0
            stats["avg_confidence"] /= stats["count"] if stats["count"] > 0 else 1.0
        
        for stats in response_type_stats.values():
            stats["success_rate"] = stats["success_count"] / stats["count"] if stats["count"] > 0 else 0.0
            stats["avg_confidence"] /= stats["count"] if stats["count"] > 0 else 1.0
        
        # Issue distribution
        total_issues = sum(m.issue_count for m in recent_metrics)
        issue_distribution = {
            "total_issues": total_issues,
            "critical": sum(m.critical_issues for m in recent_metrics),
            "high": sum(m.high_issues for m in recent_metrics),
            "medium": sum(m.medium_issues for m in recent_metrics),
            "low": sum(m.low_issues for m in recent_metrics)
        }
        
        return {
            "period_hours": hours,
            "total_validations": total_validations,
            "success_rate": round(success_rate, 3),
            "avg_confidence": round(avg_confidence, 3),
            "avg_validation_time_ms": round(avg_validation_time, 2),
            "domain_stats": dict(domain_stats),
            "response_type_stats": dict(response_type_stats),
            "issue_distribution": issue_distribution,
            "auto_corrections": sum(1 for m in recent_metrics if m.auto_corrected)
        }
    
    def get_domain_performance(self, domain: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get validation performance for a specific domain
        
        Args:
            domain: Domain name
            hours: Number of hours to look back
            
        Returns:
            Dictionary with domain performance metrics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            domain_metrics = [m for m in self.metrics 
                            if m.domain == domain and m.timestamp >= cutoff_time]
        
        if not domain_metrics:
            return {
                "domain": domain,
                "period_hours": hours,
                "total_validations": 0,
                "success_rate": 0.0,
                "avg_confidence": 0.0
            }
        
        total = len(domain_metrics)
        successful = sum(1 for m in domain_metrics if m.is_valid)
        avg_confidence = sum(m.confidence_score for m in domain_metrics) / total
        
        # Recent trend (last vs previous half of period)
        mid_time = cutoff_time + timedelta(hours=hours/2)
        recent_metrics = [m for m in domain_metrics if m.timestamp >= mid_time]
        older_metrics = [m for m in domain_metrics if m.timestamp < mid_time]
        
        trend = "stable"
        if recent_metrics and older_metrics:
            recent_success_rate = sum(1 for m in recent_metrics if m.is_valid) / len(recent_metrics)
            older_success_rate = sum(1 for m in older_metrics if m.is_valid) / len(older_metrics)
            
            if recent_success_rate > older_success_rate + 0.1:
                trend = "improving"
            elif recent_success_rate < older_success_rate - 0.1:
                trend = "declining"
        
        return {
            "domain": domain,
            "period_hours": hours,
            "total_validations": total,
            "success_rate": round(successful / total, 3),
            "avg_confidence": round(avg_confidence, 3),
            "trend": trend,
            "recent_validations": len(recent_metrics),
            "common_issues": self._get_common_issues(domain_metrics)
        }
    
    def _get_common_issues(self, metrics: List[ValidationMetric]) -> List[str]:
        """Get list of common issue patterns for given metrics"""
        # This is a simplified version - in a real implementation,
        # you'd track specific issue types and messages
        issue_patterns = []
        
        critical_count = sum(m.critical_issues for m in metrics)
        high_count = sum(m.high_issues for m in metrics)
        medium_count = sum(m.medium_issues for m in metrics)
        
        if critical_count > len(metrics) * 0.1:
            issue_patterns.append("Frequent critical validation failures")
        
        if high_count > len(metrics) * 0.3:
            issue_patterns.append("High severity issues common")
        
        if medium_count > len(metrics) * 0.5:
            issue_patterns.append("Many medium severity issues")
        
        return issue_patterns
    
    def export_metrics(self, hours: int = 24, format: str = "json") -> str:
        """
        Export metrics data
        
        Args:
            hours: Number of hours to look back
            format: Export format ("json" or "csv")
            
        Returns:
            Exported data as string
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
        
        if format.lower() == "json":
            # Convert to JSON-serializable format
            metrics_data = []
            for metric in recent_metrics:
                metric_dict = asdict(metric)
                metric_dict["timestamp"] = metric.timestamp.isoformat()
                metrics_data.append(metric_dict)
            
            return json.dumps({
                "export_time": datetime.now().isoformat(),
                "period_hours": hours,
                "metrics": metrics_data
            }, indent=2)
        
        elif format.lower() == "csv":
            import csv
            import io
            
            output = io.StringIO()
            if recent_metrics:
                fieldnames = list(asdict(recent_metrics[0]).keys())
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                
                for metric in recent_metrics:
                    metric_dict = asdict(metric)
                    metric_dict["timestamp"] = metric.timestamp.isoformat()
                    writer.writerow(metric_dict)
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global metrics collector instance
_metrics_collector: Optional[ValidationMetricsCollector] = None
_collector_lock = Lock()


def get_metrics_collector() -> ValidationMetricsCollector:
    """Get or create the global metrics collector instance"""
    global _metrics_collector
    
    if _metrics_collector is None:
        with _collector_lock:
            if _metrics_collector is None:
                _metrics_collector = ValidationMetricsCollector()
    
    return _metrics_collector


def record_validation_metric(domain: str,
                           response_type: str, 
                           validation_result: ValidationResult,
                           validation_time_ms: float) -> None:
    """
    Convenience function to record validation metrics
    
    Args:
        domain: Domain being validated
        response_type: Type of response
        validation_result: ValidationResult instance  
        validation_time_ms: Validation time in milliseconds
    """
    collector = get_metrics_collector()
    collector.record_validation(domain, response_type, validation_result, validation_time_ms)


def get_validation_summary(hours: int = 24) -> Dict[str, Any]:
    """
    Get validation metrics summary
    
    Args:
        hours: Number of hours to look back
        
    Returns:
        Metrics summary dictionary
    """
    collector = get_metrics_collector()
    return collector.get_metrics_summary(hours)


class ValidationMonitor:
    """Monitor for validation performance and alerting"""
    
    def __init__(self, 
                 success_rate_threshold: float = 0.8,
                 confidence_threshold: float = 0.7,
                 alert_callback: Optional[callable] = None):
        """
        Initialize validation monitor
        
        Args:
            success_rate_threshold: Minimum acceptable success rate
            confidence_threshold: Minimum acceptable confidence score
            alert_callback: Function to call when thresholds are breached
        """
        self.success_rate_threshold = success_rate_threshold
        self.confidence_threshold = confidence_threshold
        self.alert_callback = alert_callback or self._default_alert
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def check_performance(self, hours: int = 1) -> Dict[str, Any]:
        """
        Check validation performance against thresholds
        
        Args:
            hours: Period to check
            
        Returns:
            Performance check results
        """
        # Enhanced logging for performance monitoring
        logger.info(f"🔍 [MONITOR] Starting performance check for last {hours} hour(s)")
        
        summary = get_validation_summary(hours)
        
        # Log current performance metrics
        logger.info(f"📊 [MONITOR] Current metrics - "
                   f"Total validations: {summary['total_validations']}, "
                   f"Success rate: {summary['success_rate']:.2f}, "
                   f"Avg confidence: {summary['avg_confidence']:.2f}")
        
        issues = []
        
        # Check success rate
        if summary["success_rate"] < self.success_rate_threshold:
            logger.warning(f"🚨 [MONITOR] SUCCESS RATE ALERT: {summary['success_rate']:.2f} < {self.success_rate_threshold}")
            issues.append({
                "type": "low_success_rate",
                "current": summary["success_rate"],
                "threshold": self.success_rate_threshold,
                "severity": "high"
            })
        else:
            logger.info(f"✅ [MONITOR] Success rate OK: {summary['success_rate']:.2f} >= {self.success_rate_threshold}")
        
        # Check confidence score
        if summary["avg_confidence"] < self.confidence_threshold:
            logger.warning(f"🎯 [MONITOR] CONFIDENCE ALERT: {summary['avg_confidence']:.2f} < {self.confidence_threshold}")
            issues.append({
                "type": "low_confidence",
                "current": summary["avg_confidence"], 
                "threshold": self.confidence_threshold,
                "severity": "medium"
            })
        else:
            logger.info(f"✅ [MONITOR] Confidence OK: {summary['avg_confidence']:.2f} >= {self.confidence_threshold}")
        
        # Check for domains with poor performance
        domain_issues = 0
        for domain, stats in summary.get("domain_stats", {}).items():
            if stats["success_rate"] < self.success_rate_threshold:
                logger.warning(f"🏷️ [MONITOR] DOMAIN ALERT: {domain} success rate {stats['success_rate']:.2f} < {self.success_rate_threshold}")
                domain_issues += 1
                issues.append({
                    "type": "domain_performance",
                    "domain": domain,
                    "current": stats["success_rate"],
                    "threshold": self.success_rate_threshold,
                    "severity": "medium"
                })
        
        if domain_issues == 0:
            logger.info(f"✅ [MONITOR] All domains performing within thresholds")
        
        # Enhanced alert logging
        if issues:
            logger.error(f"🚨 [MONITOR] PERFORMANCE ISSUES DETECTED: {len(issues)} issues found")
            for issue in issues:
                logger.error(f"   🔴 {issue['type'].upper()}: "
                           f"current={issue.get('current', 'N/A')} "
                           f"threshold={issue.get('threshold', 'N/A')} "
                           f"severity={issue['severity']}")
            
            logger.info(f"📢 [MONITOR] Triggering alert callbacks")
            self.alert_callback(issues, summary)
        else:
            logger.info(f"✅ [MONITOR] All performance checks PASSED - no issues detected")
        
        # Trigger alerts if issues found
        if issues:
            self.alert_callback(issues, summary)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "period_hours": hours,
            "issues_found": len(issues),
            "issues": issues,
            "overall_health": "good" if not issues else "degraded"
        }
    
    def _default_alert(self, issues: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
        """Default alert handler - logs warnings"""
        self.logger.warning(f"Validation performance issues detected: {len(issues)} problems")
        for issue in issues:
            self.logger.warning(f"  {issue['type']}: {issue.get('current', 'N/A')} "
                              f"(threshold: {issue.get('threshold', 'N/A')})")