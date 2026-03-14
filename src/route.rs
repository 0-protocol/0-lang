//! Route - Multi-path routing system for ZeroLang
//!
//! Provides priority-based routing with confidence thresholds for AI assistant applications.

use crate::graph::{NodeHash, Route, RouteMetadata};
use crate::tensor::Tensor;

/// Result of a routing decision
#[derive(Debug, Clone)]
pub struct RouteResult {
    /// The selected route (None if default was used)
    pub selected_route: Option<SelectedRoute>,
    /// Whether the default route was used
    pub used_default: bool,
    /// Confidence of the routing decision
    pub confidence: f32,
    /// All routes that were evaluated
    pub evaluated_routes: Vec<RouteEvaluation>,
}

/// Information about the selected route
#[derive(Debug, Clone)]
pub struct SelectedRoute {
    /// Index of the selected route
    pub index: usize,
    /// Name of the selected route
    pub name: String,
    /// Target node hash
    pub target: NodeHash,
    /// Confidence that matched
    pub confidence: f32,
}

/// Evaluation result for a single route
#[derive(Debug, Clone)]
pub struct RouteEvaluation {
    /// Route index
    pub index: usize,
    /// Route name
    pub name: String,
    /// Condition confidence score
    pub confidence: f32,
    /// Threshold required
    pub threshold: f32,
    /// Whether this route matched
    pub matched: bool,
}

/// Router for evaluating and selecting routes
pub struct Router {
    /// Routes to evaluate
    routes: Vec<RouteConfig>,
    /// Default target if no route matches
    default_target: NodeHash,
    /// Router configuration
    config: RouterConfig,
}

/// Configuration for a single route
#[derive(Debug, Clone)]
pub struct RouteConfig {
    /// Minimum confidence to take this route
    pub threshold: f32,
    /// Target node if route matches
    pub target: NodeHash,
    /// Route metadata
    pub metadata: RouteMetadata,
}

/// Configuration for the router
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Whether to continue evaluating after finding a match (for metrics)
    pub evaluate_all: bool,
    /// Minimum confidence delta between routes to prefer one over another
    pub confidence_epsilon: f32,
    /// Whether to use the highest confidence route instead of first match
    pub use_highest_confidence: bool,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            evaluate_all: false,
            confidence_epsilon: 0.01,
            use_highest_confidence: false,
        }
    }
}

impl Router {
    /// Create a new router with routes and default target
    pub fn new(routes: Vec<RouteConfig>, default_target: NodeHash) -> Self {
        Self {
            routes,
            default_target,
            config: RouterConfig::default(),
        }
    }

    /// Create a router with custom configuration
    pub fn with_config(
        routes: Vec<RouteConfig>,
        default_target: NodeHash,
        config: RouterConfig,
    ) -> Self {
        Self {
            routes,
            default_target,
            config,
        }
    }

    /// Create a router from graph Route structures
    pub fn from_graph_routes(routes: Vec<Route>, default_target: NodeHash) -> Self {
        let route_configs: Vec<RouteConfig> = routes
            .into_iter()
            .map(|r| RouteConfig {
                threshold: r.threshold,
                target: r.target,
                metadata: r.metadata,
            })
            .collect();
        Self::new(route_configs, default_target)
    }

    /// Evaluate routes and select the best one
    ///
    /// The `condition_results` parameter should contain the confidence scores
    /// for each route's condition, in the same order as the routes.
    pub fn evaluate(&self, condition_results: &[f32]) -> RouteResult {
        let mut evaluated_routes = Vec::new();
        let mut first_match: Option<(usize, f32)> = None;
        let mut highest_match: Option<(usize, f32)> = None;

        for (index, (route, &confidence)) in
            self.routes.iter().zip(condition_results.iter()).enumerate()
        {
            let matched = confidence >= route.threshold;

            evaluated_routes.push(RouteEvaluation {
                index,
                name: route.metadata.name.clone(),
                confidence,
                threshold: route.threshold,
                matched,
            });

            if matched {
                if first_match.is_none() {
                    first_match = Some((index, confidence));
                }

                if let Some((_, highest_conf)) = highest_match {
                    if confidence > highest_conf + self.config.confidence_epsilon {
                        highest_match = Some((index, confidence));
                    }
                } else {
                    highest_match = Some((index, confidence));
                }

                // If not evaluating all and not using highest confidence, stop here
                if !self.config.evaluate_all && !self.config.use_highest_confidence {
                    break;
                }
            }
        }

        // Select route based on configuration
        let selected = if self.config.use_highest_confidence {
            highest_match
        } else {
            first_match
        };

        match selected {
            Some((index, confidence)) => {
                let route = &self.routes[index];
                RouteResult {
                    selected_route: Some(SelectedRoute {
                        index,
                        name: route.metadata.name.clone(),
                        target: route.target.clone(),
                        confidence,
                    }),
                    used_default: false,
                    confidence,
                    evaluated_routes,
                }
            }
            None => RouteResult {
                selected_route: None,
                used_default: true,
                confidence: 0.0,
                evaluated_routes,
            },
        }
    }

    /// Get the default target
    pub fn default_target(&self) -> &NodeHash {
        &self.default_target
    }

    /// Get the number of routes
    pub fn route_count(&self) -> usize {
        self.routes.len()
    }

    /// Add a new route
    pub fn add_route(&mut self, route: RouteConfig) {
        self.routes.push(route);
    }

    /// Remove a route by index
    pub fn remove_route(&mut self, index: usize) -> Option<RouteConfig> {
        if index < self.routes.len() {
            Some(self.routes.remove(index))
        } else {
            None
        }
    }
}

/// Builder for creating routes more easily
pub struct RouteBuilder {
    routes: Vec<RouteConfig>,
    default_target: Option<NodeHash>,
    config: RouterConfig,
}

impl RouteBuilder {
    /// Create a new route builder
    pub fn new() -> Self {
        Self {
            routes: Vec::new(),
            default_target: None,
            config: RouterConfig::default(),
        }
    }

    /// Add a route with threshold and target
    pub fn route(
        mut self,
        name: impl Into<String>,
        threshold: f32,
        target: NodeHash,
    ) -> Self {
        self.routes.push(RouteConfig {
            threshold,
            target,
            metadata: RouteMetadata {
                name: name.into(),
                description: String::new(),
            },
        });
        self
    }

    /// Add a route with full metadata
    pub fn route_with_description(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        threshold: f32,
        target: NodeHash,
    ) -> Self {
        self.routes.push(RouteConfig {
            threshold,
            target,
            metadata: RouteMetadata {
                name: name.into(),
                description: description.into(),
            },
        });
        self
    }

    /// Set the default target
    pub fn default(mut self, target: NodeHash) -> Self {
        self.default_target = Some(target);
        self
    }

    /// Set whether to evaluate all routes
    pub fn evaluate_all(mut self, evaluate: bool) -> Self {
        self.config.evaluate_all = evaluate;
        self
    }

    /// Set whether to use highest confidence route
    pub fn use_highest_confidence(mut self, use_highest: bool) -> Self {
        self.config.use_highest_confidence = use_highest;
        self
    }

    /// Build the router
    pub fn build(self) -> Router {
        let default_target = self.default_target.unwrap_or_else(|| vec![0]);
        Router::with_config(self.routes, default_target, self.config)
    }
}

impl Default for RouteBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to create a simple condition tensor from a confidence value
pub fn condition_tensor(confidence: f32) -> Tensor {
    Tensor::confidence_scalar(confidence)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_route(name: &str, threshold: f32) -> RouteConfig {
        RouteConfig {
            threshold,
            target: vec![name.len() as u8],
            metadata: RouteMetadata {
                name: name.to_string(),
                description: format!("Route to {}", name),
            },
        }
    }

    #[test]
    fn test_first_match_routing() {
        let routes = vec![
            make_route("high", 0.9),
            make_route("medium", 0.7),
            make_route("low", 0.5),
        ];
        let router = Router::new(routes, vec![0]);

        // Confidence 0.75 should match "medium" (first match)
        let result = router.evaluate(&[0.5, 0.75, 0.6]);
        assert!(!result.used_default);
        assert_eq!(result.selected_route.unwrap().name, "medium");
    }

    #[test]
    fn test_highest_confidence_routing() {
        let routes = vec![
            make_route("first", 0.5),
            make_route("second", 0.5),
            make_route("third", 0.5),
        ];
        let config = RouterConfig {
            use_highest_confidence: true,
            ..Default::default()
        };
        let router = Router::with_config(routes, vec![0], config);

        // Should select "third" with highest confidence
        let result = router.evaluate(&[0.6, 0.7, 0.9]);
        assert!(!result.used_default);
        assert_eq!(result.selected_route.unwrap().name, "third");
    }

    #[test]
    fn test_default_route() {
        let routes = vec![make_route("high", 0.9), make_route("very_high", 0.95)];
        let router = Router::new(routes, vec![255]);

        // No route matches
        let result = router.evaluate(&[0.5, 0.6]);
        assert!(result.used_default);
        assert!(result.selected_route.is_none());
    }

    #[test]
    fn test_route_builder() {
        let router = RouteBuilder::new()
            .route("fast", 0.8, vec![1])
            .route("slow", 0.5, vec![2])
            .default(vec![0])
            .use_highest_confidence(true)
            .build();

        assert_eq!(router.route_count(), 2);
    }

    #[test]
    fn test_evaluate_all() {
        let routes = vec![
            make_route("a", 0.5),
            make_route("b", 0.5),
            make_route("c", 0.5),
        ];
        let config = RouterConfig {
            evaluate_all: true,
            ..Default::default()
        };
        let router = Router::with_config(routes, vec![0], config);

        let result = router.evaluate(&[0.6, 0.7, 0.8]);
        // All routes should be evaluated
        assert_eq!(result.evaluated_routes.len(), 3);
        assert!(result.evaluated_routes.iter().all(|r| r.matched));
    }
}
