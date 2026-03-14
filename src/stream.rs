//! Stream - Streaming data support for ZeroLang
//!
//! Provides support for WebSocket, SSE, and channel-based streaming data sources.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::tensor::{StreamHandle, StreamSource, Tensor};

/// Error types for stream operations
#[derive(Debug, Clone)]
pub enum StreamError {
    /// Connection failed
    ConnectionFailed { url: String, reason: String },
    /// Stream not found
    StreamNotFound { id: u64 },
    /// Stream closed
    StreamClosed { id: u64 },
    /// Buffer overflow
    BufferOverflow { id: u64, max_size: usize },
    /// Invalid source type
    InvalidSourceType { reason: String },
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamError::ConnectionFailed { url, reason } => {
                write!(f, "Connection to '{}' failed: {}", url, reason)
            }
            StreamError::StreamNotFound { id } => write!(f, "Stream {} not found", id),
            StreamError::StreamClosed { id } => write!(f, "Stream {} is closed", id),
            StreamError::BufferOverflow { id, max_size } => {
                write!(f, "Stream {} buffer overflow (max: {})", id, max_size)
            }
            StreamError::InvalidSourceType { reason } => {
                write!(f, "Invalid source type: {}", reason)
            }
        }
    }
}

impl std::error::Error for StreamError {}

/// Configuration for stream behavior
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Maximum buffer size before overflow error
    pub max_buffer_size: usize,
    /// Reconnect on disconnect
    pub auto_reconnect: bool,
    /// Maximum reconnect attempts
    pub max_reconnect_attempts: u32,
    /// Reconnect delay in milliseconds
    pub reconnect_delay_ms: u64,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: 1000,
            auto_reconnect: true,
            max_reconnect_attempts: 5,
            reconnect_delay_ms: 1000,
        }
    }
}

/// Stream state tracking
#[derive(Debug, Clone, PartialEq)]
pub enum StreamState {
    /// Stream is connecting
    Connecting,
    /// Stream is connected and active
    Connected,
    /// Stream is disconnected
    Disconnected,
    /// Stream is reconnecting
    Reconnecting { attempt: u32 },
    /// Stream is closed permanently
    Closed,
}

/// Internal stream info for management
struct StreamInfo {
    handle: StreamHandle,
    state: Arc<RwLock<StreamState>>,
    config: StreamConfig,
}

/// Manages all active streams in the system
pub struct StreamManager {
    /// Active streams indexed by ID
    streams: HashMap<u64, StreamInfo>,
    /// Next stream ID counter
    next_id: AtomicU64,
    /// Default configuration for new streams
    default_config: StreamConfig,
}

impl StreamManager {
    /// Create a new stream manager
    pub fn new() -> Self {
        Self {
            streams: HashMap::new(),
            next_id: AtomicU64::new(1),
            default_config: StreamConfig::default(),
        }
    }

    /// Create a stream manager with custom default configuration
    pub fn with_config(config: StreamConfig) -> Self {
        Self {
            streams: HashMap::new(),
            next_id: AtomicU64::new(1),
            default_config: config,
        }
    }

    /// Allocate a new stream ID
    fn allocate_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Create a new stream from a WebSocket URL
    pub async fn from_websocket(&mut self, url: &str) -> Result<StreamHandle, StreamError> {
        self.from_websocket_with_config(url, self.default_config.clone())
            .await
    }

    /// Create a new stream from a WebSocket URL with custom config
    pub async fn from_websocket_with_config(
        &mut self,
        url: &str,
        config: StreamConfig,
    ) -> Result<StreamHandle, StreamError> {
        let id = self.allocate_id();
        let buffer = Arc::new(RwLock::new(VecDeque::new()));

        let handle = StreamHandle {
            id,
            source_type: StreamSource::WebSocket {
                url: url.to_string(),
            },
            buffer: buffer.clone(),
        };

        let state = Arc::new(RwLock::new(StreamState::Connecting));

        // Store stream info
        let info = StreamInfo {
            handle: handle.clone(),
            state: state.clone(),
            config,
        };
        self.streams.insert(id, info);

        // In a real implementation, we would spawn a WebSocket listener here
        // For now, we just mark it as connected
        *state.write().await = StreamState::Connected;

        Ok(handle)
    }

    /// Create a new stream from an internal channel
    pub async fn from_channel(&mut self, channel_id: &str) -> Result<StreamHandle, StreamError> {
        let id = self.allocate_id();
        let buffer = Arc::new(RwLock::new(VecDeque::new()));

        let handle = StreamHandle {
            id,
            source_type: StreamSource::Channel {
                channel_id: channel_id.to_string(),
            },
            buffer: buffer.clone(),
        };

        let state = Arc::new(RwLock::new(StreamState::Connected));

        let info = StreamInfo {
            handle: handle.clone(),
            state,
            config: self.default_config.clone(),
        };
        self.streams.insert(id, info);

        Ok(handle)
    }

    /// Create a new stream from an event source (SSE)
    pub async fn from_event_source(
        &mut self,
        event_type: &str,
    ) -> Result<StreamHandle, StreamError> {
        let id = self.allocate_id();
        let buffer = Arc::new(RwLock::new(VecDeque::new()));

        let handle = StreamHandle {
            id,
            source_type: StreamSource::Event {
                event_type: event_type.to_string(),
            },
            buffer: buffer.clone(),
        };

        let state = Arc::new(RwLock::new(StreamState::Connected));

        let info = StreamInfo {
            handle: handle.clone(),
            state,
            config: self.default_config.clone(),
        };
        self.streams.insert(id, info);

        Ok(handle)
    }

    /// Read next item from stream (non-blocking)
    pub async fn read(&self, handle: &StreamHandle) -> Option<Tensor> {
        let mut buffer = handle.buffer.write().await;
        buffer.pop_front()
    }

    /// Read next item from stream, waiting if empty
    pub async fn read_blocking(&self, handle: &StreamHandle) -> Result<Tensor, StreamError> {
        loop {
            // Check if stream exists and is connected
            let info = self
                .streams
                .get(&handle.id)
                .ok_or(StreamError::StreamNotFound { id: handle.id })?;

            let state = info.state.read().await;
            if *state == StreamState::Closed {
                return Err(StreamError::StreamClosed { id: handle.id });
            }
            drop(state);

            // Try to read from buffer
            if let Some(tensor) = self.read(handle).await {
                return Ok(tensor);
            }

            // Wait a bit before trying again
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }
    }

    /// Push a tensor to a stream's buffer (for testing or internal use)
    pub async fn push(&self, handle: &StreamHandle, tensor: Tensor) -> Result<(), StreamError> {
        let info = self
            .streams
            .get(&handle.id)
            .ok_or(StreamError::StreamNotFound { id: handle.id })?;

        let mut buffer = handle.buffer.write().await;

        // Check buffer size limit
        if buffer.len() >= info.config.max_buffer_size {
            return Err(StreamError::BufferOverflow {
                id: handle.id,
                max_size: info.config.max_buffer_size,
            });
        }

        buffer.push_back(tensor);
        Ok(())
    }

    /// Get the current state of a stream
    pub async fn get_state(&self, id: u64) -> Result<StreamState, StreamError> {
        let info = self
            .streams
            .get(&id)
            .ok_or(StreamError::StreamNotFound { id })?;
        Ok(info.state.read().await.clone())
    }

    /// Close a stream
    pub async fn close(&mut self, id: u64) -> Result<(), StreamError> {
        let info = self
            .streams
            .get(&id)
            .ok_or(StreamError::StreamNotFound { id })?;

        *info.state.write().await = StreamState::Closed;
        Ok(())
    }

    /// Remove a closed stream from the manager
    pub fn remove(&mut self, id: u64) -> Option<StreamHandle> {
        self.streams.remove(&id).map(|info| info.handle)
    }

    /// Get the number of active streams
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }

    /// Get buffer size for a stream
    pub async fn buffer_size(&self, handle: &StreamHandle) -> Result<usize, StreamError> {
        if !self.streams.contains_key(&handle.id) {
            return Err(StreamError::StreamNotFound { id: handle.id });
        }
        Ok(handle.buffer.read().await.len())
    }
}

impl Default for StreamManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_websocket_stream() {
        let mut manager = StreamManager::new();
        let handle = manager.from_websocket("wss://example.com/ws").await.unwrap();

        assert_eq!(handle.id, 1);
        assert!(matches!(
            handle.source_type,
            StreamSource::WebSocket { .. }
        ));

        let state = manager.get_state(handle.id).await.unwrap();
        assert_eq!(state, StreamState::Connected);
    }

    #[tokio::test]
    async fn test_create_channel_stream() {
        let mut manager = StreamManager::new();
        let handle = manager.from_channel("test-channel").await.unwrap();

        assert!(matches!(
            handle.source_type,
            StreamSource::Channel { channel_id } if channel_id == "test-channel"
        ));
    }

    #[tokio::test]
    async fn test_push_and_read() {
        let mut manager = StreamManager::new();
        let handle = manager.from_channel("test").await.unwrap();

        // Push some tensors
        let t1 = Tensor::scalar(1.0, 1.0);
        let t2 = Tensor::scalar(2.0, 1.0);

        manager.push(&handle, t1.clone()).await.unwrap();
        manager.push(&handle, t2.clone()).await.unwrap();

        // Read them back
        let r1 = manager.read(&handle).await.unwrap();
        let r2 = manager.read(&handle).await.unwrap();
        let r3 = manager.read(&handle).await;

        assert_eq!(r1.as_scalar(), 1.0);
        assert_eq!(r2.as_scalar(), 2.0);
        assert!(r3.is_none());
    }

    #[tokio::test]
    async fn test_buffer_overflow() {
        let config = StreamConfig {
            max_buffer_size: 2,
            ..Default::default()
        };
        let mut manager = StreamManager::with_config(config);
        let handle = manager.from_channel("test").await.unwrap();

        // Push up to limit
        manager
            .push(&handle, Tensor::scalar(1.0, 1.0))
            .await
            .unwrap();
        manager
            .push(&handle, Tensor::scalar(2.0, 1.0))
            .await
            .unwrap();

        // This should fail
        let result = manager.push(&handle, Tensor::scalar(3.0, 1.0)).await;
        assert!(matches!(result, Err(StreamError::BufferOverflow { .. })));
    }

    #[tokio::test]
    async fn test_close_stream() {
        let mut manager = StreamManager::new();
        let handle = manager.from_channel("test").await.unwrap();

        manager.close(handle.id).await.unwrap();

        let state = manager.get_state(handle.id).await.unwrap();
        assert_eq!(state, StreamState::Closed);
    }
}
