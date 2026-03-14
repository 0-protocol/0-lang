//! Agent-native Web3 helpers for ZeroLang.
//!
//! These helpers deliberately optimize for machine consumption:
//! stable field names, canonical JSON output, and exact/raw values
//! alongside normalized approximations for downstream graph math.

use std::time::Duration;

use num_bigint::BigUint;
use reqwest::blocking::Client;
use serde_json::{json, Value};

const DEFAULT_PRIORITY_FEE_GWEI: f64 = 2.0;
const DEFAULT_SAFETY_MULTIPLIER: f64 = 1.2;

pub fn get_gas_price_json(
    rpc_url: &str,
    priority_fee_gwei: Option<f32>,
    safety_multiplier: Option<f32>,
) -> Result<String, String> {
    let client = RpcClient::new(rpc_url)?;
    let legacy_hex = client.call("eth_gasPrice", json!([]))?;
    let legacy_wei = hex_quantity_to_u128(&legacy_hex)?;

    let latest_block = client.call_value("eth_getBlockByNumber", json!(["latest", false]))?;
    let base_fee_hex = latest_block
        .get("baseFeePerGas")
        .and_then(Value::as_str)
        .map(str::to_string);
    let base_fee_wei = match base_fee_hex.as_deref() {
        Some(hex) => Some(hex_quantity_to_u128(hex)?),
        None => None,
    };

    let priority_fee_hex = client.call_optional("eth_maxPriorityFeePerGas", json!([]))?;
    let configured_priority = priority_fee_gwei
        .map(|value| value.max(0.0) as f64)
        .unwrap_or(DEFAULT_PRIORITY_FEE_GWEI);
    let priority_fee_wei = match priority_fee_hex.as_deref() {
        Some(hex) => hex_quantity_to_u128(hex)?,
        None => gwei_to_wei(configured_priority),
    };

    let multiplier = safety_multiplier
        .map(|value| value.max(1.0) as f64)
        .unwrap_or(DEFAULT_SAFETY_MULTIPLIER);
    let max_fee_wei = match base_fee_wei {
        Some(base_fee) => ((base_fee as f64 * multiplier).round() as u128).saturating_add(priority_fee_wei),
        None => legacy_wei.max(priority_fee_wei),
    };

    serde_json::to_string(&json!({
        "source": "evm.gas",
        "rpc_url": rpc_url,
        "network_style": if base_fee_wei.is_some() { "eip1559" } else { "legacy" },
        "legacy_wei": legacy_wei.to_string(),
        "legacy_gwei": wei_to_gwei(legacy_wei),
        "base_fee_wei": base_fee_wei.map(|value| value.to_string()),
        "base_fee_gwei": base_fee_wei.map(wei_to_gwei),
        "priority_fee_wei": priority_fee_wei.to_string(),
        "priority_fee_gwei": wei_to_gwei(priority_fee_wei),
        "max_fee_wei": max_fee_wei.to_string(),
        "max_fee_gwei": wei_to_gwei(max_fee_wei),
    }))
    .map_err(|err| format!("failed to serialize gas quote: {err}"))
}

pub fn oracle_read_json(
    rpc_url: &str,
    address: &str,
    call_data: &str,
    decimals: u32,
) -> Result<String, String> {
    let client = RpcClient::new(rpc_url)?;
    let raw_hex = client.call(
        "eth_call",
        json!([
            {
                "to": address,
                "data": call_data,
            },
            "latest"
        ]),
    )?;

    let raw_value = hex_data_to_biguint(&raw_hex)?;
    let integer_value = raw_value.to_str_radix(10);
    let normalized = format_units(&raw_value, decimals);
    let normalized_f32 = normalized.parse::<f32>().ok();

    serde_json::to_string(&json!({
        "source": "evm.oracle",
        "rpc_url": rpc_url,
        "address": address,
        "call_data": call_data,
        "decimals": decimals,
        "raw_hex": raw_hex,
        "value": integer_value,
        "normalized": normalized,
        "normalized_f32": normalized_f32,
    }))
    .map_err(|err| format!("failed to serialize oracle read: {err}"))
}

struct RpcClient {
    rpc_url: String,
    client: Client,
}

impl RpcClient {
    fn new(rpc_url: &str) -> Result<Self, String> {
        let client = Client::builder()
            .timeout(Duration::from_secs(15))
            .build()
            .map_err(|err| format!("failed to build RPC client: {err}"))?;
        Ok(Self {
            rpc_url: rpc_url.to_string(),
            client,
        })
    }

    fn call(&self, method: &str, params: Value) -> Result<String, String> {
        let response = self.raw_call(method, params)?;
        response
            .get("result")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| format!("RPC method {method} returned non-string result: {response}"))
    }

    fn call_optional(&self, method: &str, params: Value) -> Result<Option<String>, String> {
        let response = self.raw_call(method, params)?;
        match response.get("result") {
            Some(Value::String(value)) => Ok(Some(value.clone())),
            Some(Value::Null) | None => Ok(None),
            Some(other) => Err(format!("RPC method {method} returned unexpected result: {other}")),
        }
    }

    fn call_value(&self, method: &str, params: Value) -> Result<Value, String> {
        let response = self.raw_call(method, params)?;
        response
            .get("result")
            .cloned()
            .ok_or_else(|| format!("RPC method {method} returned no result"))
    }

    fn raw_call(&self, method: &str, params: Value) -> Result<Value, String> {
        let body = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        });

        let response = self
            .client
            .post(&self.rpc_url)
            .json(&body)
            .send()
            .map_err(|err| format!("RPC method {method} failed: {err}"))?;

        let status = response.status();
        let json: Value = response
            .json()
            .map_err(|err| format!("RPC method {method} returned invalid JSON: {err}"))?;

        if !status.is_success() {
            return Err(format!("RPC method {method} returned HTTP {status}: {json}"));
        }

        if let Some(error) = json.get("error") {
            return Err(format!("RPC method {method} returned error: {error}"));
        }

        Ok(json)
    }
}

fn hex_quantity_to_u128(hex: &str) -> Result<u128, String> {
    let sanitized = hex
        .strip_prefix("0x")
        .ok_or_else(|| format!("hex quantity must start with 0x: {hex}"))?;
    if sanitized.is_empty() {
        return Ok(0);
    }
    u128::from_str_radix(sanitized, 16).map_err(|err| format!("invalid hex quantity {hex}: {err}"))
}

fn hex_data_to_biguint(hex: &str) -> Result<BigUint, String> {
    let sanitized = hex
        .strip_prefix("0x")
        .ok_or_else(|| format!("hex data must start with 0x: {hex}"))?;
    if sanitized.is_empty() {
        return Ok(BigUint::default());
    }
    BigUint::parse_bytes(sanitized.as_bytes(), 16)
        .ok_or_else(|| format!("invalid hex data {hex}"))
}

fn gwei_to_wei(value: f64) -> u128 {
    (value * 1_000_000_000.0).round() as u128
}

fn wei_to_gwei(value: u128) -> f64 {
    value as f64 / 1_000_000_000.0
}

fn format_units(value: &BigUint, decimals: u32) -> String {
    if decimals == 0 {
        return value.to_str_radix(10);
    }

    let digits = value.to_str_radix(10);
    let decimals = decimals as usize;
    if digits.len() <= decimals {
        let zero_padding = "0".repeat(decimals - digits.len());
        let mut out = format!("0.{}{}", zero_padding, digits);
        trim_fractional_zeros(&mut out);
        return out;
    }

    let split = digits.len() - decimals;
    let mut out = format!("{}.{}", &digits[..split], &digits[split..]);
    trim_fractional_zeros(&mut out);
    out
}

fn trim_fractional_zeros(value: &mut String) {
    while value.ends_with('0') {
        value.pop();
    }
    if value.ends_with('.') {
        value.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_units_integer() {
        let value = BigUint::parse_bytes(b"123450000", 10).unwrap();
        assert_eq!(format_units(&value, 6), "123.45");
    }

    #[test]
    fn test_format_units_subunit() {
        let value = BigUint::parse_bytes(b"42", 10).unwrap();
        assert_eq!(format_units(&value, 4), "0.0042");
    }

    #[test]
    fn test_hex_quantity_to_u128() {
        assert_eq!(hex_quantity_to_u128("0x2540be400").unwrap(), 10_000_000_000);
    }
}
