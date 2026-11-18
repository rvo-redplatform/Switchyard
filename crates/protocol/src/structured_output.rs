// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Structured output expressed as one tool definition.
//!
//! Some upstreams (and gateways in front of them) ignore `response_format`, but
//! every tool-capable model honours a forced tool call. These helpers turn a
//! Chat-shaped JSON `response_format` into the tool whose parameters are the
//! requested schema, so a host can force that tool instead.

use serde_json::{Value, json};

use crate::{LlmClientError, ToolDefinition};

/// Name of the tool that carries a structured response.
pub const STRUCTURED_OUTPUT_TOOL: &str = "respond";

const TOOL_DESCRIPTION: &str = "Give your complete final answer by calling this tool. \
     The arguments are the whole response and must match the parameter schema.";

/// Builds the [`STRUCTURED_OUTPUT_TOOL`] definition for a JSON `response_format`.
///
/// `format` is Chat-shaped: `{"type": "json_schema", "json_schema": {...}}` or
/// `{"type": "json_object"}`. Returns `Ok(None)` for any other type (`text`, or
/// an unrecognized one). Returns an invalid-request error when the schema cannot
/// be a tool's parameters, because tool arguments are always an object.
pub fn structured_output_tool(format: &Value) -> Result<Option<ToolDefinition>, LlmClientError> {
    let (parameters, description, strict) = match format.get("type").and_then(Value::as_str) {
        Some("json_schema") => json_schema_tool_parts(format)?,
        Some("json_object") => (
            permissive_object_schema(),
            TOOL_DESCRIPTION.to_string(),
            None,
        ),
        _ => return Ok(None),
    };
    Ok(Some(ToolDefinition {
        name: STRUCTURED_OUTPUT_TOOL.to_string(),
        description: Some(description),
        parameters,
        strict,
    }))
}

// Reads the tool parameters, description and strictness from a Chat-shaped
// `{"type": "json_schema", "json_schema": {...}}` format.
fn json_schema_tool_parts(format: &Value) -> Result<(Value, String, Option<bool>), LlmClientError> {
    let spec = format.get("json_schema").and_then(Value::as_object);
    let parameters = match spec.and_then(|spec| spec.get("schema")) {
        None | Some(Value::Null) => permissive_object_schema(),
        Some(Value::Object(schema)) => {
            if let Some(kind) = schema
                .get("type")
                .and_then(Value::as_str)
                .filter(|kind| *kind != "object")
            {
                return Err(invalid_request(format!(
                    "response_format json_schema must describe an object (got type {kind}): \
                     structured output is served through a tool call, whose arguments are an object"
                )));
            }
            Value::Object(schema.clone())
        }
        Some(_) => {
            return Err(invalid_request(
                "response_format json_schema.schema must be a JSON object",
            ));
        }
    };
    let name = spec
        .and_then(|spec| spec.get("name"))
        .and_then(Value::as_str);
    let description = match (
        spec.and_then(|spec| spec.get("description"))
            .and_then(Value::as_str),
        name,
    ) {
        (Some(description), _) => format!("{TOOL_DESCRIPTION} {description}"),
        (None, Some(name)) => format!("{TOOL_DESCRIPTION} The response is `{name}`."),
        (None, None) => TOOL_DESCRIPTION.to_string(),
    };
    let strict = spec
        .and_then(|spec| spec.get("strict"))
        .and_then(Value::as_bool);
    Ok((parameters, description, strict))
}

// `json_object` asks for any JSON object.
fn permissive_object_schema() -> Value {
    json!({"type": "object", "additionalProperties": true})
}

fn invalid_request(message: impl Into<String>) -> LlmClientError {
    LlmClientError::InvalidRequest {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_schema_becomes_the_tool_parameters() {
        let schema = json!({"type": "object", "properties": {"answer": {"type": "integer"}}});
        let tool = structured_output_tool(&json!({
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": schema, "strict": true}
        }))
        .expect("valid")
        .expect("a json_schema format has a tool");
        assert_eq!(tool.name, STRUCTURED_OUTPUT_TOOL);
        assert_eq!(tool.parameters, schema);
        assert_eq!(tool.strict, Some(true));
        assert!(
            tool.description
                .as_deref()
                .is_some_and(|description| description.contains("`answer`"))
        );
    }

    #[test]
    fn json_object_uses_a_permissive_object_schema() {
        let tool = structured_output_tool(&json!({"type": "json_object"}))
            .expect("valid")
            .expect("a json_object format has a tool");
        assert_eq!(tool.parameters, permissive_object_schema());
        assert_eq!(tool.strict, None);
    }

    #[test]
    fn other_formats_have_no_tool() {
        for format in [json!({"type": "text"}), json!({}), json!({"type": "xml"})] {
            assert_eq!(structured_output_tool(&format).ok(), Some(None));
        }
    }

    #[test]
    fn non_object_schemas_are_refused() {
        for schema in [json!({"type": "array"}), json!("object")] {
            let format = json!({"type": "json_schema", "json_schema": {"schema": schema}});
            assert!(matches!(
                structured_output_tool(&format),
                Err(LlmClientError::InvalidRequest { .. })
            ));
        }
    }
}
