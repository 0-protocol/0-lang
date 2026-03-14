//! Timer - Scheduled execution system for ZeroLang
//!
//! Provides cron-like scheduling for AI assistant background tasks.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::graph::{NodeHash, OverlapPolicy};

/// Error types for timer operations
#[derive(Debug, Clone)]
pub enum TimerError {
    /// Invalid cron expression
    InvalidSchedule { expression: String, reason: String },
    /// Timer not found
    TimerNotFound { id: u64 },
    /// Timer already exists
    TimerAlreadyExists { id: u64 },
    /// Maximum concurrent executions reached
    MaxConcurrentReached { id: u64, max: u32 },
    /// Timer is paused
    TimerPaused { id: u64 },
}

impl std::fmt::Display for TimerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimerError::InvalidSchedule { expression, reason } => {
                write!(f, "Invalid schedule '{}': {}", expression, reason)
            }
            TimerError::TimerNotFound { id } => write!(f, "Timer {} not found", id),
            TimerError::TimerAlreadyExists { id } => write!(f, "Timer {} already exists", id),
            TimerError::MaxConcurrentReached { id, max } => {
                write!(f, "Timer {} reached max concurrent executions: {}", id, max)
            }
            TimerError::TimerPaused { id } => write!(f, "Timer {} is paused", id),
        }
    }
}

impl std::error::Error for TimerError {}

/// State of a timer
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimerState {
    /// Timer is active and will fire on schedule
    Active,
    /// Timer is paused
    Paused,
    /// Timer is stopped permanently
    Stopped,
}

/// Execution record for a timer
#[derive(Debug, Clone)]
pub struct TimerExecution {
    /// Execution ID
    pub id: u64,
    /// Timer ID
    pub timer_id: u64,
    /// Start timestamp
    pub started_at: u64,
    /// End timestamp (None if still running)
    pub ended_at: Option<u64>,
    /// Whether execution was successful
    pub success: Option<bool>,
    /// Error message if failed
    pub error: Option<String>,
}

/// Configuration for a timer
#[derive(Debug, Clone)]
pub struct TimerConfig {
    /// Cron expression for scheduling
    pub schedule: String,
    /// Target node to execute
    pub target: NodeHash,
    /// Maximum concurrent executions
    pub max_concurrent: u32,
    /// Overlap policy
    pub overlap_policy: OverlapPolicy,
    /// Timer name (for debugging)
    pub name: String,
    /// Timer description
    pub description: String,
}

impl TimerConfig {
    /// Create a new timer config
    pub fn new(schedule: impl Into<String>, target: NodeHash) -> Self {
        Self {
            schedule: schedule.into(),
            target,
            max_concurrent: 1,
            overlap_policy: OverlapPolicy::Skip,
            name: String::new(),
            description: String::new(),
        }
    }

    /// Set the name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set max concurrent executions
    pub fn with_max_concurrent(mut self, max: u32) -> Self {
        self.max_concurrent = max;
        self
    }

    /// Set overlap policy
    pub fn with_overlap_policy(mut self, policy: OverlapPolicy) -> Self {
        self.overlap_policy = policy;
        self
    }
}

/// Internal timer information
struct TimerInfo {
    config: TimerConfig,
    state: TimerState,
    current_executions: u32,
    queued_executions: u32,
    total_executions: u64,
    last_execution: Option<u64>,
    _next_execution: Option<u64>,
}

/// Manages all timers in the system
pub struct TimerManager {
    /// Active timers
    timers: Arc<RwLock<HashMap<u64, TimerInfo>>>,
    /// Execution history
    executions: Arc<RwLock<Vec<TimerExecution>>>,
    /// Next timer ID
    next_id: AtomicU64,
    /// Next execution ID
    next_execution_id: AtomicU64,
    /// Maximum execution history to keep
    max_history: usize,
}

impl TimerManager {
    /// Create a new timer manager
    pub fn new() -> Self {
        Self {
            timers: Arc::new(RwLock::new(HashMap::new())),
            executions: Arc::new(RwLock::new(Vec::new())),
            next_id: AtomicU64::new(1),
            next_execution_id: AtomicU64::new(1),
            max_history: 1000,
        }
    }

    /// Create a timer manager with custom history limit
    pub fn with_max_history(max_history: usize) -> Self {
        Self {
            max_history,
            ..Self::new()
        }
    }

    /// Register a new timer
    pub async fn register(&self, config: TimerConfig) -> Result<u64, TimerError> {
        // Validate schedule (basic validation)
        Self::validate_schedule(&config.schedule)?;

        let id = self.next_id.fetch_add(1, Ordering::SeqCst);

        let info = TimerInfo {
            config,
            state: TimerState::Active,
            current_executions: 0,
            queued_executions: 0,
            total_executions: 0,
            last_execution: None,
            _next_execution: None, // Would be calculated from cron expression
        };

        let mut timers = self.timers.write().await;
        timers.insert(id, info);

        Ok(id)
    }

    /// Validate a cron schedule expression
    fn validate_schedule(schedule: &str) -> Result<(), TimerError> {
        // Basic validation - check for required number of fields
        let parts: Vec<&str> = schedule.split_whitespace().collect();
        if parts.len() < 5 || parts.len() > 7 {
            return Err(TimerError::InvalidSchedule {
                expression: schedule.to_string(),
                reason: format!(
                    "Expected 5-7 fields (minute hour day month weekday [year] [seconds]), got {}",
                    parts.len()
                ),
            });
        }

        // Validate each field has valid characters
        for (i, part) in parts.iter().enumerate() {
            let valid = part.chars().all(|c| {
                c.is_ascii_digit() || c == '*' || c == '/' || c == '-' || c == ','
            });
            if !valid {
                return Err(TimerError::InvalidSchedule {
                    expression: schedule.to_string(),
                    reason: format!("Invalid characters in field {} ('{}')", i, part),
                });
            }
        }

        Ok(())
    }

    /// Trigger a timer execution (called by scheduler)
    pub async fn trigger(&self, timer_id: u64) -> Result<u64, TimerError> {
        let mut timers = self.timers.write().await;
        let info = timers
            .get_mut(&timer_id)
            .ok_or(TimerError::TimerNotFound { id: timer_id })?;

        if info.state == TimerState::Paused {
            return Err(TimerError::TimerPaused { id: timer_id });
        }

        if info.state == TimerState::Stopped {
            return Err(TimerError::TimerNotFound { id: timer_id });
        }

        // Handle overlap based on policy
        if info.current_executions >= info.config.max_concurrent {
            match info.config.overlap_policy {
                OverlapPolicy::Skip => {
                    return Err(TimerError::MaxConcurrentReached {
                        id: timer_id,
                        max: info.config.max_concurrent,
                    });
                }
                OverlapPolicy::Queue => {
                    info.queued_executions += 1;
                    // Return a placeholder execution ID for queued
                    return Ok(0);
                }
                OverlapPolicy::Parallel => {
                    // Continue to create execution
                }
            }
        }

        let exec_id = self.next_execution_id.fetch_add(1, Ordering::SeqCst);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        info.current_executions += 1;
        info.total_executions += 1;
        info.last_execution = Some(now);

        // Record execution
        let execution = TimerExecution {
            id: exec_id,
            timer_id,
            started_at: now,
            ended_at: None,
            success: None,
            error: None,
        };

        drop(timers);

        let mut executions = self.executions.write().await;
        executions.push(execution);

        // Trim history if needed
        if executions.len() > self.max_history {
            let excess = executions.len() - self.max_history;
            executions.drain(0..excess);
        }

        Ok(exec_id)
    }

    /// Complete a timer execution
    pub async fn complete(
        &self,
        execution_id: u64,
        success: bool,
        error: Option<String>,
    ) -> Result<(), TimerError> {
        let mut executions = self.executions.write().await;
        let execution = executions
            .iter_mut()
            .find(|e| e.id == execution_id)
            .ok_or(TimerError::TimerNotFound { id: execution_id })?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        execution.ended_at = Some(now);
        execution.success = Some(success);
        execution.error = error;

        let timer_id = execution.timer_id;
        drop(executions);

        // Update timer state
        let mut timers = self.timers.write().await;
        if let Some(info) = timers.get_mut(&timer_id) {
            info.current_executions = info.current_executions.saturating_sub(1);

            // Process queued executions
            if info.queued_executions > 0
                && info.current_executions < info.config.max_concurrent
            {
                info.queued_executions -= 1;
                // Would trigger another execution here
            }
        }

        Ok(())
    }

    /// Pause a timer
    pub async fn pause(&self, timer_id: u64) -> Result<(), TimerError> {
        let mut timers = self.timers.write().await;
        let info = timers
            .get_mut(&timer_id)
            .ok_or(TimerError::TimerNotFound { id: timer_id })?;
        info.state = TimerState::Paused;
        Ok(())
    }

    /// Resume a paused timer
    pub async fn resume(&self, timer_id: u64) -> Result<(), TimerError> {
        let mut timers = self.timers.write().await;
        let info = timers
            .get_mut(&timer_id)
            .ok_or(TimerError::TimerNotFound { id: timer_id })?;
        if info.state == TimerState::Paused {
            info.state = TimerState::Active;
        }
        Ok(())
    }

    /// Stop a timer permanently
    pub async fn stop(&self, timer_id: u64) -> Result<(), TimerError> {
        let mut timers = self.timers.write().await;
        let info = timers
            .get_mut(&timer_id)
            .ok_or(TimerError::TimerNotFound { id: timer_id })?;
        info.state = TimerState::Stopped;
        Ok(())
    }

    /// Remove a stopped timer
    pub async fn remove(&self, timer_id: u64) -> Result<TimerConfig, TimerError> {
        let mut timers = self.timers.write().await;
        let info = timers
            .remove(&timer_id)
            .ok_or(TimerError::TimerNotFound { id: timer_id })?;
        Ok(info.config)
    }

    /// Get timer state
    pub async fn get_state(&self, timer_id: u64) -> Result<TimerState, TimerError> {
        let timers = self.timers.read().await;
        let info = timers
            .get(&timer_id)
            .ok_or(TimerError::TimerNotFound { id: timer_id })?;
        Ok(info.state)
    }

    /// Get execution history for a timer
    pub async fn get_history(&self, timer_id: u64) -> Vec<TimerExecution> {
        let executions = self.executions.read().await;
        executions
            .iter()
            .filter(|e| e.timer_id == timer_id)
            .cloned()
            .collect()
    }

    /// Get all active timers
    pub async fn active_timers(&self) -> Vec<u64> {
        let timers = self.timers.read().await;
        timers
            .iter()
            .filter(|(_, info)| info.state == TimerState::Active)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get total timer count
    pub async fn timer_count(&self) -> usize {
        self.timers.read().await.len()
    }
}

impl Default for TimerManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating timers more easily
pub struct TimerBuilder {
    schedule: String,
    target: NodeHash,
    name: Option<String>,
    description: Option<String>,
    max_concurrent: u32,
    overlap_policy: OverlapPolicy,
}

impl TimerBuilder {
    /// Create a new timer builder
    pub fn new(schedule: impl Into<String>, target: NodeHash) -> Self {
        Self {
            schedule: schedule.into(),
            target,
            name: None,
            description: None,
            max_concurrent: 1,
            overlap_policy: OverlapPolicy::Skip,
        }
    }

    /// Set the timer name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the timer description
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set max concurrent executions
    pub fn max_concurrent(mut self, max: u32) -> Self {
        self.max_concurrent = max;
        self
    }

    /// Set overlap policy to skip
    pub fn skip_on_overlap(mut self) -> Self {
        self.overlap_policy = OverlapPolicy::Skip;
        self
    }

    /// Set overlap policy to queue
    pub fn queue_on_overlap(mut self) -> Self {
        self.overlap_policy = OverlapPolicy::Queue;
        self
    }

    /// Set overlap policy to parallel
    pub fn parallel_on_overlap(mut self) -> Self {
        self.overlap_policy = OverlapPolicy::Parallel;
        self
    }

    /// Build the timer config
    pub fn build(self) -> TimerConfig {
        TimerConfig {
            schedule: self.schedule,
            target: self.target,
            max_concurrent: self.max_concurrent,
            overlap_policy: self.overlap_policy,
            name: self.name.unwrap_or_default(),
            description: self.description.unwrap_or_default(),
        }
    }
}

/// Common cron schedule presets
pub mod schedules {
    /// Every minute
    pub const EVERY_MINUTE: &str = "* * * * *";
    /// Every 5 minutes
    pub const EVERY_5_MINUTES: &str = "*/5 * * * *";
    /// Every 15 minutes
    pub const EVERY_15_MINUTES: &str = "*/15 * * * *";
    /// Every 30 minutes
    pub const EVERY_30_MINUTES: &str = "*/30 * * * *";
    /// Every hour
    pub const HOURLY: &str = "0 * * * *";
    /// Every day at midnight
    pub const DAILY: &str = "0 0 * * *";
    /// Every week on Sunday at midnight
    pub const WEEKLY: &str = "0 0 * * 0";
    /// Every month on the 1st at midnight
    pub const MONTHLY: &str = "0 0 1 * *";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_register_timer() {
        let manager = TimerManager::new();
        let config = TimerConfig::new("*/5 * * * *", vec![1, 2, 3]);

        let id = manager.register(config).await.unwrap();
        assert_eq!(id, 1);

        let state = manager.get_state(id).await.unwrap();
        assert_eq!(state, TimerState::Active);
    }

    #[tokio::test]
    async fn test_invalid_schedule() {
        let manager = TimerManager::new();
        let config = TimerConfig::new("invalid", vec![1]);

        let result = manager.register(config).await;
        assert!(matches!(result, Err(TimerError::InvalidSchedule { .. })));
    }

    #[tokio::test]
    async fn test_trigger_and_complete() {
        let manager = TimerManager::new();
        let config = TimerConfig::new("*/5 * * * *", vec![1]);
        let timer_id = manager.register(config).await.unwrap();

        let exec_id = manager.trigger(timer_id).await.unwrap();
        assert!(exec_id > 0);

        manager.complete(exec_id, true, None).await.unwrap();

        let history = manager.get_history(timer_id).await;
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].success, Some(true));
    }

    #[tokio::test]
    async fn test_pause_resume() {
        let manager = TimerManager::new();
        let config = TimerConfig::new("*/5 * * * *", vec![1]);
        let timer_id = manager.register(config).await.unwrap();

        manager.pause(timer_id).await.unwrap();
        assert_eq!(manager.get_state(timer_id).await.unwrap(), TimerState::Paused);

        // Should fail to trigger while paused
        let result = manager.trigger(timer_id).await;
        assert!(matches!(result, Err(TimerError::TimerPaused { .. })));

        manager.resume(timer_id).await.unwrap();
        assert_eq!(manager.get_state(timer_id).await.unwrap(), TimerState::Active);
    }

    #[tokio::test]
    async fn test_max_concurrent_skip() {
        let manager = TimerManager::new();
        let config = TimerConfig::new("*/5 * * * *", vec![1])
            .with_max_concurrent(1)
            .with_overlap_policy(OverlapPolicy::Skip);
        let timer_id = manager.register(config).await.unwrap();

        // First trigger should succeed
        let exec_id = manager.trigger(timer_id).await.unwrap();
        assert!(exec_id > 0);

        // Second trigger should fail (skip)
        let result = manager.trigger(timer_id).await;
        assert!(matches!(result, Err(TimerError::MaxConcurrentReached { .. })));
    }

    #[test]
    fn test_timer_builder() {
        let config = TimerBuilder::new(schedules::EVERY_5_MINUTES, vec![1, 2, 3])
            .name("test-timer")
            .max_concurrent(3)
            .queue_on_overlap()
            .build();

        assert_eq!(config.schedule, "*/5 * * * *");
        assert_eq!(config.max_concurrent, 3);
        assert_eq!(config.overlap_policy, OverlapPolicy::Queue);
    }
}
