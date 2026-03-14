//! Permission - Confidence-based permission system for ZeroLang
//!
//! Provides permission evaluation with confidence scores for AI assistant applications.

use crate::tensor::Tensor;

/// Result of a permission evaluation
#[derive(Debug, Clone)]
pub struct PermissionResult {
    /// Whether the permission was granted
    pub allowed: bool,
    /// Confidence score of the decision
    pub confidence: f32,
    /// Human-readable reason for the decision
    pub reason: String,
    /// Audit trail of checks performed
    pub audit_trail: Vec<PermissionAuditEntry>,
}

impl PermissionResult {
    /// Create a new allowed permission result
    pub fn allowed(confidence: f32, reason: impl Into<String>) -> Self {
        Self {
            allowed: true,
            confidence,
            reason: reason.into(),
            audit_trail: Vec::new(),
        }
    }

    /// Create a new denied permission result
    pub fn denied(confidence: f32, reason: impl Into<String>) -> Self {
        Self {
            allowed: false,
            confidence,
            reason: reason.into(),
            audit_trail: Vec::new(),
        }
    }

    /// Add an audit entry
    pub fn with_audit(mut self, entry: PermissionAuditEntry) -> Self {
        self.audit_trail.push(entry);
        self
    }
}

/// Entry in the permission audit trail
#[derive(Debug, Clone)]
pub struct PermissionAuditEntry {
    /// Timestamp of the check
    pub timestamp: u64,
    /// Check name/type
    pub check_name: String,
    /// Check result (pass/fail)
    pub passed: bool,
    /// Confidence from this check
    pub confidence: f32,
    /// Additional details
    pub details: String,
}

/// Policy configuration for permission evaluation
#[derive(Debug, Clone)]
pub struct PermissionPolicy {
    /// Minimum confidence required to grant permission
    pub threshold: f32,
    /// How to combine multiple permission checks
    pub combination: CombinationStrategy,
    /// Whether to log permission decisions for auditing
    pub audit: bool,
    /// Default action when confidence is below threshold
    pub default_action: DefaultAction,
}

impl Default for PermissionPolicy {
    fn default() -> Self {
        Self {
            threshold: 0.8,
            combination: CombinationStrategy::All,
            audit: true,
            default_action: DefaultAction::Deny,
        }
    }
}

impl PermissionPolicy {
    /// Create a new policy with custom threshold
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            threshold,
            ..Default::default()
        }
    }

    /// Create a permissive policy (any check passes)
    pub fn permissive() -> Self {
        Self {
            threshold: 0.5,
            combination: CombinationStrategy::Any,
            audit: true,
            default_action: DefaultAction::Allow,
        }
    }

    /// Create a strict policy (all checks must pass with high confidence)
    pub fn strict() -> Self {
        Self {
            threshold: 0.95,
            combination: CombinationStrategy::All,
            audit: true,
            default_action: DefaultAction::Deny,
        }
    }
}

/// Strategy for combining multiple permission checks
#[derive(Debug, Clone, PartialEq)]
pub enum CombinationStrategy {
    /// All checks must pass
    All,
    /// Any check can pass
    Any,
    /// More than half must pass
    Majority,
    /// Weighted combination of checks
    Weighted(Vec<f32>),
}

/// Default action when permission cannot be determined
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DefaultAction {
    /// Deny by default (secure)
    Deny,
    /// Allow by default (permissive)
    Allow,
    /// Escalate to fallback handler
    Escalate,
}

/// Permission evaluator for checking access
pub struct PermissionEvaluator {
    /// The policy to use for evaluation
    policy: PermissionPolicy,
}

impl PermissionEvaluator {
    /// Create a new permission evaluator with default policy
    pub fn new() -> Self {
        Self {
            policy: PermissionPolicy::default(),
        }
    }

    /// Create a permission evaluator with custom policy
    pub fn with_policy(policy: PermissionPolicy) -> Self {
        Self { policy }
    }

    /// Evaluate a permission based on subject and action tensors
    pub fn evaluate(&self, subject: &Tensor, action: &Tensor) -> PermissionResult {
        // Get base confidence from inputs
        let subject_confidence = subject.confidence;
        let action_confidence = action.confidence;

        // Combine confidences
        let combined_confidence = match &self.policy.combination {
            CombinationStrategy::All => subject_confidence.min(action_confidence),
            CombinationStrategy::Any => subject_confidence.max(action_confidence),
            CombinationStrategy::Majority => (subject_confidence + action_confidence) / 2.0,
            CombinationStrategy::Weighted(weights) => {
                if weights.len() >= 2 {
                    (subject_confidence * weights[0] + action_confidence * weights[1])
                        / (weights[0] + weights[1])
                } else {
                    (subject_confidence + action_confidence) / 2.0
                }
            }
        };

        // Create audit entry
        let audit_entry = PermissionAuditEntry {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            check_name: "confidence_check".to_string(),
            passed: combined_confidence >= self.policy.threshold,
            confidence: combined_confidence,
            details: format!(
                "Subject confidence: {}, Action confidence: {}, Combined: {}",
                subject_confidence, action_confidence, combined_confidence
            ),
        };

        // Make decision
        let allowed = combined_confidence >= self.policy.threshold;
        let reason = if allowed {
            format!(
                "Permission granted: confidence {} >= threshold {}",
                combined_confidence, self.policy.threshold
            )
        } else {
            format!(
                "Permission denied: confidence {} < threshold {}",
                combined_confidence, self.policy.threshold
            )
        };

        let mut result = if allowed {
            PermissionResult::allowed(combined_confidence, reason)
        } else {
            PermissionResult::denied(combined_confidence, reason)
        };

        if self.policy.audit {
            result = result.with_audit(audit_entry);
        }

        result
    }

    /// Evaluate multiple checks and combine results
    pub fn evaluate_multiple(&self, checks: Vec<(Tensor, Tensor)>) -> PermissionResult {
        if checks.is_empty() {
            return match self.policy.default_action {
                DefaultAction::Allow => {
                    PermissionResult::allowed(1.0, "No checks required, allowing by default")
                }
                DefaultAction::Deny => {
                    PermissionResult::denied(0.0, "No checks provided, denying by default")
                }
                DefaultAction::Escalate => PermissionResult::denied(
                    0.5,
                    "No checks provided, escalating to fallback",
                ),
            };
        }

        let results: Vec<PermissionResult> = checks
            .iter()
            .map(|(s, a)| self.evaluate(s, a))
            .collect();

        let confidences: Vec<f32> = results.iter().map(|r| r.confidence).collect();
        let all_allowed = results.iter().all(|r| r.allowed);
        let any_allowed = results.iter().any(|r| r.allowed);
        let majority_allowed = results.iter().filter(|r| r.allowed).count() > results.len() / 2;

        let (allowed, combined_confidence) = match &self.policy.combination {
            CombinationStrategy::All => {
                let min_conf = confidences.iter().cloned().fold(1.0f32, f32::min);
                (all_allowed, min_conf)
            }
            CombinationStrategy::Any => {
                let max_conf = confidences.iter().cloned().fold(0.0f32, f32::max);
                (any_allowed, max_conf)
            }
            CombinationStrategy::Majority => {
                let avg_conf: f32 = confidences.iter().sum::<f32>() / confidences.len() as f32;
                (majority_allowed, avg_conf)
            }
            CombinationStrategy::Weighted(weights) => {
                let weighted_sum: f32 = confidences
                    .iter()
                    .zip(weights.iter().cycle())
                    .map(|(c, w)| c * w)
                    .sum();
                let weight_sum: f32 = weights.iter().cycle().take(confidences.len()).sum();
                let weighted_avg = weighted_sum / weight_sum;
                (weighted_avg >= self.policy.threshold, weighted_avg)
            }
        };

        let reason = format!(
            "Multiple checks evaluated: {} of {} passed",
            results.iter().filter(|r| r.allowed).count(),
            results.len()
        );

        let mut result = if allowed {
            PermissionResult::allowed(combined_confidence, reason)
        } else {
            PermissionResult::denied(combined_confidence, reason)
        };

        // Combine audit trails
        for r in results {
            for entry in r.audit_trail {
                result = result.with_audit(entry);
            }
        }

        result
    }

    /// Get the current policy
    pub fn policy(&self) -> &PermissionPolicy {
        &self.policy
    }
}

impl Default for PermissionEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function for simple permission evaluation
pub fn evaluate_permission(
    subject: &Tensor,
    action: &Tensor,
    policy: &PermissionPolicy,
) -> PermissionResult {
    let evaluator = PermissionEvaluator::with_policy(policy.clone());
    evaluator.evaluate(subject, action)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permission_allowed() {
        let policy = PermissionPolicy::with_threshold(0.7);
        let evaluator = PermissionEvaluator::with_policy(policy);

        let subject = Tensor::scalar(1.0, 0.9);
        let action = Tensor::scalar(1.0, 0.8);

        let result = evaluator.evaluate(&subject, &action);
        assert!(result.allowed);
        assert!(result.confidence >= 0.8);
    }

    #[test]
    fn test_permission_denied() {
        let policy = PermissionPolicy::with_threshold(0.9);
        let evaluator = PermissionEvaluator::with_policy(policy);

        let subject = Tensor::scalar(1.0, 0.7);
        let action = Tensor::scalar(1.0, 0.6);

        let result = evaluator.evaluate(&subject, &action);
        assert!(!result.allowed);
    }

    #[test]
    fn test_combination_any() {
        let policy = PermissionPolicy {
            threshold: 0.7,
            combination: CombinationStrategy::Any,
            ..Default::default()
        };
        let evaluator = PermissionEvaluator::with_policy(policy);

        let subject = Tensor::scalar(1.0, 0.9);
        let action = Tensor::scalar(1.0, 0.5);

        let result = evaluator.evaluate(&subject, &action);
        assert!(result.allowed); // 0.9 >= 0.7
    }

    #[test]
    fn test_strict_policy() {
        let policy = PermissionPolicy::strict();
        let evaluator = PermissionEvaluator::with_policy(policy);

        let subject = Tensor::scalar(1.0, 0.9);
        let action = Tensor::scalar(1.0, 0.9);

        let result = evaluator.evaluate(&subject, &action);
        assert!(!result.allowed); // 0.9 < 0.95 threshold
    }

    #[test]
    fn test_audit_trail() {
        let policy = PermissionPolicy {
            audit: true,
            ..Default::default()
        };
        let evaluator = PermissionEvaluator::with_policy(policy);

        let subject = Tensor::scalar(1.0, 0.9);
        let action = Tensor::scalar(1.0, 0.85);

        let result = evaluator.evaluate(&subject, &action);
        assert!(!result.audit_trail.is_empty());
        assert_eq!(result.audit_trail[0].check_name, "confidence_check");
    }
}
