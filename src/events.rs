//! Event System for ZeroLang Trading Applications
//!
//! Provides an event-driven architecture for tick-based execution
//! of trading graphs.

use rust_decimal::Decimal;

use crate::graph::NodeHash;

/// Order status for trading events
#[derive(Debug, Clone, PartialEq)]
pub enum OrderStatus {
    /// Order is pending submission
    Pending,
    /// Order has been submitted
    Submitted,
    /// Order is partially filled
    PartiallyFilled {
        filled_quantity: Decimal,
        remaining: Decimal,
    },
    /// Order is fully filled
    Filled,
    /// Order has been cancelled
    Cancelled,
    /// Order was rejected
    Rejected { reason: String },
    /// Order expired
    Expired,
}

/// Trading events that can trigger graph execution
#[derive(Debug, Clone)]
pub enum TradingEvent {
    /// Time-based tick event
    Tick {
        /// Unix timestamp in milliseconds
        timestamp: u64,
    },
    /// Order fill notification
    Fill {
        /// Unique order identifier
        order_id: String,
        /// Fill price
        price: Decimal,
        /// Fill quantity
        quantity: Decimal,
        /// Fill side (true = buy, false = sell)
        is_buy: bool,
        /// Timestamp of fill
        timestamp: u64,
    },
    /// Order status update
    OrderUpdate {
        /// Unique order identifier
        order_id: String,
        /// New status
        status: OrderStatus,
        /// Timestamp of update
        timestamp: u64,
    },
    /// Market price update
    PriceUpdate {
        /// Trading pair (e.g., "BTC/USD")
        pair: String,
        /// Current price
        price: Decimal,
        /// Bid price
        bid: Option<Decimal>,
        /// Ask price
        ask: Option<Decimal>,
        /// Timestamp
        timestamp: u64,
    },
    /// Balance update
    BalanceUpdate {
        /// Asset symbol (e.g., "BTC", "USD")
        asset: String,
        /// Total balance
        total: Decimal,
        /// Available balance (not locked in orders)
        available: Decimal,
        /// Timestamp
        timestamp: u64,
    },
    /// Position update
    PositionUpdate {
        /// Trading pair
        pair: String,
        /// Position size (positive = long, negative = short)
        size: Decimal,
        /// Entry price
        entry_price: Decimal,
        /// Unrealized PnL
        unrealized_pnl: Decimal,
        /// Timestamp
        timestamp: u64,
    },
    /// Custom event for extensibility
    Custom {
        /// Event type identifier
        event_type: String,
        /// JSON payload
        payload: String,
        /// Timestamp
        timestamp: u64,
    },
}

impl TradingEvent {
    /// Get the timestamp of this event
    pub fn timestamp(&self) -> u64 {
        match self {
            TradingEvent::Tick { timestamp } => *timestamp,
            TradingEvent::Fill { timestamp, .. } => *timestamp,
            TradingEvent::OrderUpdate { timestamp, .. } => *timestamp,
            TradingEvent::PriceUpdate { timestamp, .. } => *timestamp,
            TradingEvent::BalanceUpdate { timestamp, .. } => *timestamp,
            TradingEvent::PositionUpdate { timestamp, .. } => *timestamp,
            TradingEvent::Custom { timestamp, .. } => *timestamp,
        }
    }

    /// Get the event type as a string
    pub fn event_type(&self) -> &str {
        match self {
            TradingEvent::Tick { .. } => "tick",
            TradingEvent::Fill { .. } => "fill",
            TradingEvent::OrderUpdate { .. } => "order_update",
            TradingEvent::PriceUpdate { .. } => "price_update",
            TradingEvent::BalanceUpdate { .. } => "balance_update",
            TradingEvent::PositionUpdate { .. } => "position_update",
            TradingEvent::Custom { event_type, .. } => event_type,
        }
    }
}

/// Trait for handling trading events
///
/// Implement this trait to create custom event handlers that
/// process trading events and determine which graph nodes to execute.
pub trait EventHandler: Send + Sync {
    /// Handle an incoming trading event
    ///
    /// Returns a list of node hashes that should be executed
    /// in response to this event.
    fn on_event(&mut self, event: TradingEvent) -> Vec<NodeHash>;

    /// Check if this handler is interested in a particular event type
    fn interested_in(&self, event_type: &str) -> bool;
}

/// Simple event handler that triggers all registered nodes on any event
pub struct SimpleEventHandler {
    /// Nodes to trigger on events
    trigger_nodes: Vec<NodeHash>,
    /// Event types this handler is interested in
    event_types: Vec<String>,
}

impl SimpleEventHandler {
    /// Create a new simple event handler
    pub fn new() -> Self {
        Self {
            trigger_nodes: Vec::new(),
            event_types: vec!["tick".to_string()],
        }
    }

    /// Add a node to trigger on events
    pub fn add_trigger_node(&mut self, node: NodeHash) {
        self.trigger_nodes.push(node);
    }

    /// Set event types to listen for
    pub fn set_event_types(&mut self, types: Vec<String>) {
        self.event_types = types;
    }
}

impl Default for SimpleEventHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl EventHandler for SimpleEventHandler {
    fn on_event(&mut self, _event: TradingEvent) -> Vec<NodeHash> {
        self.trigger_nodes.clone()
    }

    fn interested_in(&self, event_type: &str) -> bool {
        self.event_types.contains(&event_type.to_string())
    }
}

/// Event dispatcher that routes events to handlers
pub struct EventDispatcher {
    /// Registered event handlers
    handlers: Vec<Box<dyn EventHandler>>,
}

impl EventDispatcher {
    /// Create a new event dispatcher
    pub fn new() -> Self {
        Self {
            handlers: Vec::new(),
        }
    }

    /// Register an event handler
    pub fn register_handler(&mut self, handler: Box<dyn EventHandler>) {
        self.handlers.push(handler);
    }

    /// Dispatch an event to all interested handlers
    ///
    /// Returns a deduplicated list of nodes to execute
    pub fn dispatch(&mut self, event: TradingEvent) -> Vec<NodeHash> {
        let event_type = event.event_type().to_string();
        let mut nodes_to_execute = Vec::new();

        for handler in &mut self.handlers {
            if handler.interested_in(&event_type) {
                let nodes = handler.on_event(event.clone());
                for node in nodes {
                    if !nodes_to_execute.contains(&node) {
                        nodes_to_execute.push(node);
                    }
                }
            }
        }

        nodes_to_execute
    }
}

impl Default for EventDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tick_event() {
        let event = TradingEvent::Tick { timestamp: 1234567890 };
        assert_eq!(event.timestamp(), 1234567890);
        assert_eq!(event.event_type(), "tick");
    }

    #[test]
    fn test_price_update_event() {
        let event = TradingEvent::PriceUpdate {
            pair: "BTC/USD".to_string(),
            price: Decimal::new(42000, 0),
            bid: Some(Decimal::new(41990, 0)),
            ask: Some(Decimal::new(42010, 0)),
            timestamp: 1234567890,
        };
        assert_eq!(event.event_type(), "price_update");
    }

    #[test]
    fn test_simple_event_handler() {
        let mut handler = SimpleEventHandler::new();
        handler.add_trigger_node(vec![1, 2, 3]);
        handler.add_trigger_node(vec![4, 5, 6]);

        assert!(handler.interested_in("tick"));
        assert!(!handler.interested_in("fill"));

        let event = TradingEvent::Tick { timestamp: 0 };
        let nodes = handler.on_event(event);
        assert_eq!(nodes.len(), 2);
    }

    #[test]
    fn test_event_dispatcher() {
        let mut dispatcher = EventDispatcher::new();
        
        let mut handler = SimpleEventHandler::new();
        handler.add_trigger_node(vec![1, 2, 3]);
        handler.set_event_types(vec!["tick".to_string(), "price_update".to_string()]);
        
        dispatcher.register_handler(Box::new(handler));

        let event = TradingEvent::Tick { timestamp: 0 };
        let nodes = dispatcher.dispatch(event);
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0], vec![1, 2, 3]);
    }
}
