// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Structured output served through one forced tool call.
//!
//! A request whose `response_format` asks for JSON (`json_schema` or
//! `json_object`) is rewritten before routing: one function tool whose
//! parameters are the requested schema is added, `tool_choice` forces it, and
//! `response_format` is removed. Every tool-capable target can then serve
//! structured output, including ones without a native JSON mode. On the way
//! back the forced call's arguments become the assistant's message content,
//! buffered or streamed, and the stop reason becomes a normal stop.
//!
//! The IR is Chat-shaped for every inbound dialect: Chat `response_format`,
//! Responses `text.format`, and Anthropic `output_format` all decode to
//! `{"type": "json_schema", "json_schema": {...}}`, so one rewrite covers all.

use futures_util::StreamExt;
use serde_json::{Value, json};
use switchyard_protocol::{
    AggLlmResponse, ContentBlock, LlmClientError, LlmRequest, LlmResponse, LlmResponseChunk,
    LlmResponseStreamEvent, PreservationMetadata, Response, StopReason, ToolChoice, ToolDefinition,
};

/// Name of the tool that carries a structured response.
pub const STRUCTURED_OUTPUT_TOOL: &str = "respond";

const TOOL_DESCRIPTION: &str = "Give your complete final answer by calling this tool. \
     The arguments are the whole response and must match the parameter schema.";

/// Marks a request whose structured output was turned into a forced tool call.
/// Use it to unwrap the response the route returns.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForcedStructuredOutput {
    tool: String,
}

impl ForcedStructuredOutput {
    /// Name of the forced tool.
    pub fn tool_name(&self) -> &str {
        &self.tool
    }

    /// Turns the forced tool call back into JSON message content.
    pub fn unwrap_response(&self, response: Response) -> Response {
        let Response {
            llm_response,
            metadata,
        } = response;
        let llm_response = match llm_response {
            LlmResponse::Agg(agg) => LlmResponse::Agg(unwrap_agg(agg)),
            LlmResponse::Stream(stream) => {
                let mut state = StreamUnwrap::default();
                LlmResponse::Stream(
                    stream
                        .map(move |item| item.map(|event| state.event(event)))
                        .boxed(),
                )
            }
        };
        Response {
            llm_response,
            metadata,
        }
    }
}

/// Rewrites a JSON `response_format` into one forced tool call.
///
/// Returns `Ok(None)` and leaves the request untouched when it asks for no
/// structured output (no format, `text`, or an unrecognized type). Returns an
/// invalid-request error when the caller also sent tools, because the model
/// could not tell the caller's tools from the response tool, or when the schema
/// cannot be a tool's parameters.
pub fn force_structured_output_tool(
    request: &mut LlmRequest,
) -> Result<Option<ForcedStructuredOutput>, LlmClientError> {
    let Some(format) = request.output.response_format.as_ref() else {
        return Ok(None);
    };
    let format_type = match format.get("type").and_then(Value::as_str) {
        Some(kind @ ("json_schema" | "json_object")) => kind.to_string(),
        _ => return Ok(None),
    };
    let (parameters, description, strict) = if format_type == "json_schema" {
        json_schema_tool_parts(format)?
    } else {
        (
            permissive_object_schema(),
            TOOL_DESCRIPTION.to_string(),
            None,
        )
    };
    if !request.tools.is_empty() {
        return Err(invalid_request(format!(
            "response_format {format_type} cannot be combined with tools: structured output is \
             served through a forced tool call, so send either tools or response_format, not both"
        )));
    }

    request.tools.push(ToolDefinition {
        name: STRUCTURED_OUTPUT_TOOL.to_string(),
        description: Some(description),
        parameters,
        strict,
    });
    request.tool_choice = Some(ToolChoice::Tool {
        name: STRUCTURED_OUTPUT_TOOL.to_string(),
    });
    request.output.response_format = None;
    // A preserved same-format body would replay the original response_format
    // and no tool, so the codecs must encode the rewritten IR instead.
    request.preservation.requests.clear();
    Ok(Some(ForcedStructuredOutput {
        tool: STRUCTURED_OUTPUT_TOOL.to_string(),
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

// The forced call's arguments as the JSON text the caller asked for.
fn arguments_text(arguments: &Value) -> String {
    match arguments {
        // Arguments that did not parse as JSON are kept as the raw string.
        Value::String(raw) => raw.clone(),
        other => other.to_string(),
    }
}

// The stop reason after unwrapping: a tool-use stop is a normal stop, while a
// truncation or a content filter is still reported as such.
fn unwrapped_stop_reason(reason: Option<StopReason>) -> Option<StopReason> {
    match reason {
        Some(reason @ (StopReason::MaxTokens | StopReason::ContentFilter)) => Some(reason),
        _ => Some(StopReason::EndTurn),
    }
}

fn unwrap_agg(mut agg: AggLlmResponse) -> AggLlmResponse {
    for output in &mut agg.outputs {
        let Some(text) = output.content.iter().find_map(|block| match block {
            ContentBlock::ToolCall(call) => Some(arguments_text(&call.arguments)),
            _ => None,
        }) else {
            // The model answered without calling the tool; pass that through.
            continue;
        };
        // Any preamble text is dropped so the content is exactly the JSON value.
        output
            .content
            .retain(|block| matches!(block, ContentBlock::Reasoning { .. }));
        output.content.push(ContentBlock::Text { text });
        output.stop_reason = unwrapped_stop_reason(output.stop_reason);
    }
    // The preserved provider body still describes a tool call.
    agg.preservation = PreservationMetadata::default();
    agg
}

/// Rewrites streamed chunks: argument fragments of the first tool call become
/// text deltas, and text the model emits before or after that call is dropped.
/// When no tool call arrives, the held text is released at the stop chunk.
#[derive(Default)]
struct StreamUnwrap {
    tool_index: Option<usize>,
    held_text: Vec<LlmResponseChunk>,
}

impl StreamUnwrap {
    fn event(&mut self, event: LlmResponseStreamEvent) -> LlmResponseStreamEvent {
        // Raw provider events describe the tool call, so they are never replayed.
        let (_, chunks) = event.into_parts();
        let mut out = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            self.push(chunk, &mut out);
        }
        LlmResponseStreamEvent::new(out)
    }

    fn push(&mut self, chunk: LlmResponseChunk, out: &mut Vec<LlmResponseChunk>) {
        match chunk {
            LlmResponseChunk::ToolCallDelta {
                index,
                arguments_delta,
                ..
            } => {
                let first = *self.tool_index.get_or_insert(index);
                if first != index {
                    // A second call to the response tool cannot be merged into one value.
                    return;
                }
                self.held_text.clear();
                if let Some(text) = arguments_delta.filter(|delta| !delta.is_empty()) {
                    out.push(LlmResponseChunk::TextDelta { index: 0, text });
                }
            }
            LlmResponseChunk::TextDelta { .. } => {
                if self.tool_index.is_none() {
                    self.held_text.push(chunk);
                }
            }
            LlmResponseChunk::MessageStop { reason } => {
                if self.tool_index.is_some() {
                    let keep = matches!(
                        reason.as_deref(),
                        Some("length" | "max_tokens" | "content_filter")
                    );
                    let reason = if keep {
                        reason
                    } else {
                        Some("stop".to_string())
                    };
                    out.push(LlmResponseChunk::MessageStop { reason });
                } else {
                    out.append(&mut self.held_text);
                    out.push(LlmResponseChunk::MessageStop { reason });
                }
            }
            other => out.push(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use switchyard_protocol::{FormatId, ResponseOutput, Role, ToolCall, WireFormat, text_request};

    fn schema() -> Value {
        json!({
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
            "additionalProperties": false
        })
    }

    fn json_schema_request() -> LlmRequest {
        let mut request = text_request(Some("route".to_string()), "what is six times seven?");
        request.output.response_format = Some(json!({
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": schema(), "strict": true}
        }));
        request.preservation.requests.insert(
            FormatId::from(WireFormat::OpenAiChat),
            json!({"model": "route"}),
        );
        request
    }

    fn tool_call(arguments: Value) -> ContentBlock {
        ContentBlock::ToolCall(ToolCall {
            id: "call_1".to_string(),
            name: STRUCTURED_OUTPUT_TOOL.to_string(),
            arguments,
        })
    }

    fn agg_response(content: Vec<ContentBlock>, stop_reason: StopReason) -> Response {
        let mut agg = AggLlmResponse {
            outputs: vec![ResponseOutput {
                role: Role::Assistant,
                content,
                stop_reason: Some(stop_reason),
            }],
            ..AggLlmResponse::default()
        };
        agg.preservation
            .responses
            .insert(FormatId::from(WireFormat::OpenAiChat), json!({"id": "raw"}));
        Response {
            llm_response: LlmResponse::Agg(agg),
            metadata: None,
        }
    }

    #[test]
    fn json_schema_becomes_one_forced_tool() {
        let mut request = json_schema_request();
        let forced = force_structured_output_tool(&mut request)
            .expect("a json_schema request is valid")
            .expect("a json_schema request is rewritten");
        assert_eq!(forced.tool_name(), STRUCTURED_OUTPUT_TOOL);
        assert_eq!(request.tools.len(), 1);
        let tool = &request.tools[0];
        assert_eq!(tool.name, STRUCTURED_OUTPUT_TOOL);
        assert_eq!(tool.parameters, schema());
        assert_eq!(tool.strict, Some(true));
        assert!(
            tool.description
                .as_deref()
                .is_some_and(|description| description.contains("`answer`"))
        );
        assert_eq!(
            request.tool_choice,
            Some(ToolChoice::Tool {
                name: STRUCTURED_OUTPUT_TOOL.to_string()
            })
        );
        assert_eq!(request.output.response_format, None);
        assert!(request.preservation.requests.is_empty());
    }

    #[test]
    fn json_object_uses_a_permissive_object_schema() {
        let mut request = text_request(None, "reply in JSON");
        request.output.response_format = Some(json!({"type": "json_object"}));
        request.tool_choice = Some(ToolChoice::Auto);
        let forced = force_structured_output_tool(&mut request).expect("json_object is valid");
        assert!(forced.is_some());
        assert_eq!(request.tools[0].parameters, permissive_object_schema());
        assert_eq!(request.tools[0].strict, None);
        assert_eq!(
            request.tool_choice,
            Some(ToolChoice::Tool {
                name: STRUCTURED_OUTPUT_TOOL.to_string()
            })
        );
    }

    #[test]
    fn text_and_absent_formats_are_left_alone() {
        let mut plain = text_request(None, "hi");
        assert_eq!(force_structured_output_tool(&mut plain).ok(), Some(None));

        let mut text = text_request(None, "hi");
        text.output.response_format = Some(json!({"type": "text"}));
        let before = text.clone();
        assert_eq!(force_structured_output_tool(&mut text).ok(), Some(None));
        assert_eq!(text, before);
    }

    #[test]
    fn caller_tools_with_a_json_format_are_refused() {
        let mut request = json_schema_request();
        request.tools.push(ToolDefinition {
            name: "lookup".to_string(),
            description: None,
            parameters: json!({"type": "object"}),
            strict: None,
        });
        let before = request.clone();
        match force_structured_output_tool(&mut request) {
            Err(LlmClientError::InvalidRequest { message }) => {
                assert!(message.contains("response_format"), "{message}");
                assert!(message.contains("tools"), "{message}");
            }
            other => panic!("expected an invalid request, got {other:?}"),
        }
        assert_eq!(request, before, "a refused request is not modified");
    }

    #[test]
    fn non_object_schemas_are_refused() {
        let mut array = text_request(None, "list");
        array.output.response_format = Some(json!({
            "type": "json_schema",
            "json_schema": {"name": "items", "schema": {"type": "array"}}
        }));
        assert!(matches!(
            force_structured_output_tool(&mut array),
            Err(LlmClientError::InvalidRequest { .. })
        ));

        let mut malformed = text_request(None, "list");
        malformed.output.response_format = Some(json!({
            "type": "json_schema",
            "json_schema": {"name": "items", "schema": "object"}
        }));
        assert!(matches!(
            force_structured_output_tool(&mut malformed),
            Err(LlmClientError::InvalidRequest { .. })
        ));
    }

    fn forced() -> ForcedStructuredOutput {
        let mut request = json_schema_request();
        force_structured_output_tool(&mut request)
            .expect("valid")
            .expect("rewritten")
    }

    #[test]
    fn buffered_tool_call_becomes_json_content() {
        let response = agg_response(
            vec![
                ContentBlock::Text {
                    text: "Sure, here it is.".to_string(),
                },
                tool_call(json!({"answer": 42})),
            ],
            StopReason::ToolUse,
        );
        let unwrapped = forced().unwrap_response(response);
        let agg = unwrapped.llm_response.as_agg().expect("still buffered");
        assert_eq!(
            agg.outputs[0].content,
            vec![ContentBlock::Text {
                text: r#"{"answer":42}"#.to_string()
            }]
        );
        assert_eq!(agg.outputs[0].stop_reason, Some(StopReason::EndTurn));
        assert!(agg.preservation.responses.is_empty());
    }

    #[test]
    fn buffered_truncation_keeps_its_stop_reason() {
        let response = agg_response(
            vec![tool_call(Value::String(r#"{"answer":"#.to_string()))],
            StopReason::MaxTokens,
        );
        let unwrapped = forced().unwrap_response(response);
        let agg = unwrapped.llm_response.as_agg().expect("still buffered");
        assert_eq!(
            agg.outputs[0].content,
            vec![ContentBlock::Text {
                text: r#"{"answer":"#.to_string()
            }]
        );
        assert_eq!(agg.outputs[0].stop_reason, Some(StopReason::MaxTokens));
    }

    async fn unwrap_stream(chunks: Vec<LlmResponseChunk>) -> Vec<LlmResponseChunk> {
        let events = chunks
            .into_iter()
            .map(|chunk| {
                Ok(LlmResponseStreamEvent::preserved(
                    WireFormat::OpenAiChat,
                    json!({"raw": true}),
                    vec![chunk],
                ))
            })
            .collect::<Vec<Result<LlmResponseStreamEvent, LlmClientError>>>();
        let response = Response {
            llm_response: LlmResponse::Stream(futures_util::stream::iter(events).boxed()),
            metadata: None,
        };
        let LlmResponse::Stream(stream) = forced().unwrap_response(response).llm_response else {
            panic!("a stream stays a stream");
        };
        let mut out = Vec::new();
        for item in stream.collect::<Vec<_>>().await {
            let event = item.expect("no stream errors");
            assert!(
                event.preservation().is_none(),
                "raw events are not replayed"
            );
            out.extend(event.normalized().iter().cloned());
        }
        out
    }

    #[tokio::test]
    async fn streamed_tool_call_becomes_json_text_deltas() {
        let out = unwrap_stream(vec![
            LlmResponseChunk::MessageStart {
                id: Some("chatcmpl-1".to_string()),
                model: Some("model/nova".to_string()),
            },
            LlmResponseChunk::TextDelta {
                index: 0,
                text: "Sure, ".to_string(),
            },
            LlmResponseChunk::ToolCallDelta {
                index: 0,
                id: Some("call_1".to_string()),
                name: Some(STRUCTURED_OUTPUT_TOOL.to_string()),
                arguments_delta: Some(String::new()),
            },
            LlmResponseChunk::ToolCallDelta {
                index: 0,
                id: None,
                name: None,
                arguments_delta: Some(r#"{"answer":"#.to_string()),
            },
            LlmResponseChunk::ToolCallDelta {
                index: 0,
                id: None,
                name: None,
                arguments_delta: Some("42}".to_string()),
            },
            LlmResponseChunk::MessageStop {
                reason: Some("tool_calls".to_string()),
            },
        ])
        .await;
        assert_eq!(
            out,
            vec![
                LlmResponseChunk::MessageStart {
                    id: Some("chatcmpl-1".to_string()),
                    model: Some("model/nova".to_string()),
                },
                LlmResponseChunk::TextDelta {
                    index: 0,
                    text: r#"{"answer":"#.to_string(),
                },
                LlmResponseChunk::TextDelta {
                    index: 0,
                    text: "42}".to_string(),
                },
                LlmResponseChunk::MessageStop {
                    reason: Some("stop".to_string()),
                },
            ]
        );
    }

    #[tokio::test]
    async fn streamed_answer_without_a_tool_call_is_released_at_stop() {
        let out = unwrap_stream(vec![
            LlmResponseChunk::TextDelta {
                index: 0,
                text: "I cannot ".to_string(),
            },
            LlmResponseChunk::TextDelta {
                index: 0,
                text: "answer that.".to_string(),
            },
            LlmResponseChunk::MessageStop {
                reason: Some("stop".to_string()),
            },
        ])
        .await;
        assert_eq!(
            out,
            vec![
                LlmResponseChunk::TextDelta {
                    index: 0,
                    text: "I cannot ".to_string(),
                },
                LlmResponseChunk::TextDelta {
                    index: 0,
                    text: "answer that.".to_string(),
                },
                LlmResponseChunk::MessageStop {
                    reason: Some("stop".to_string()),
                },
            ]
        );
    }
}
