// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared LLM judge primitives.
//!
//! [`Judge`] owns algorithm-specific request construction and verdict parsing.
//! [`JudgeClassifier`] owns the judge model call and hands its verdict to a policy that chooses
//! the route.

use std::marker::PhantomData;

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde_json::Value;
use switchyard_protocol::{
    AggLlmResponse, ContentBlock, InstructionBlock, LlmRequest, Message, ModelId, OutputParams,
    Role, STRUCTURED_OUTPUT_TOOL, ToolChoice, completion_text, structured_output_tool,
};

use super::robustness::{safe_client_error, safe_error_summary};

use super::classifier_contract::ClassifierContract;
use crate::core::algorithm::Driver;
use crate::core::classifier::{Classification, Classifier};
use crate::core::state::State;
use crate::observability::record_routing;
use crate::{LibsyError, Result};
use switchyard_protocol::{LlmClientError, Request, Response, RoutingDecisionKeys as Keys};

/// Builds the classifier-specific message view presented to a structured judge.
pub(crate) trait ClassifierInput: Send + Sync {
    fn build_messages(&self, state: &State, request: &Request) -> Vec<Message>;
}

/// Converts one structured model response into the verdict type consumed by a policy.
pub(crate) trait VerdictDecoder: Send + Sync {
    type Verdict: DeserializeOwned + Send + Sync;

    fn decode(
        &self,
        response: &AggLlmResponse,
        contract: &ClassifierContract,
    ) -> Result<Self::Verdict>;
}

/// Deserializes a structured response directly into a typed verdict.
pub(crate) struct SerdeDecoder<V> {
    verdict: PhantomData<fn() -> V>,
}

impl<V> SerdeDecoder<V> {
    pub(crate) const fn new() -> Self {
        Self {
            verdict: PhantomData,
        }
    }
}

impl<V> VerdictDecoder for SerdeDecoder<V>
where
    V: DeserializeOwned + Send + Sync,
{
    type Verdict = V;

    fn decode(
        &self,
        response: &AggLlmResponse,
        contract: &ClassifierContract,
    ) -> Result<Self::Verdict> {
        if !contract.validates_locally() {
            return parse_json_verdict(response);
        }
        let verdict = parse_json_verdict::<Value>(response)?;
        contract.validate_verdict(&verdict)?;
        serde_json::from_value(verdict).map_err(|error| LibsyError::AlgorithmError {
            message: format!(
                "judge reply did not parse as {}: {error}",
                std::any::type_name::<Self::Verdict>()
            ),
        })
    }
}

/// Parses a JSON value and enforces the custom contract's compiled response schema.
pub(crate) struct JsonSchemaDecoder;

impl JsonSchemaDecoder {
    pub(crate) const fn new() -> Self {
        Self
    }
}

impl VerdictDecoder for JsonSchemaDecoder {
    type Verdict = Value;

    fn decode(
        &self,
        response: &AggLlmResponse,
        contract: &ClassifierContract,
    ) -> Result<Self::Verdict> {
        let verdict = parse_json_verdict(response)?;
        contract.validate_verdict(&verdict)?;
        Ok(verdict)
    }
}

/// Runtime limits shared by structured classifier judges.
pub(crate) struct JudgeRuntimeConfig {
    max_output_tokens: u64,
}

impl JudgeRuntimeConfig {
    pub(crate) fn new(max_output_tokens: u64) -> Result<Self> {
        if max_output_tokens == 0 {
            return Err(LibsyError::AlgorithmError {
                message: "max_output_tokens must be at least 1".to_string(),
            });
        }
        Ok(Self { max_output_tokens })
    }
}

/// Reusable structured judge assembled from an input view, contract, and verdict decoder.
pub(crate) struct StructuredJudge<I, D> {
    input: I,
    contract: ClassifierContract,
    decoder: D,
    runtime: JudgeRuntimeConfig,
}

impl<I, D> StructuredJudge<I, D> {
    pub(crate) fn new(
        input: I,
        contract: ClassifierContract,
        decoder: D,
        runtime: JudgeRuntimeConfig,
    ) -> Self {
        Self {
            input,
            contract,
            decoder,
            runtime,
        }
    }

    #[cfg(test)]
    pub(crate) fn contract(&self) -> &ClassifierContract {
        &self.contract
    }
}

impl<I, D> Judge for StructuredJudge<I, D>
where
    I: ClassifierInput,
    D: VerdictDecoder,
{
    type Verdict = D::Verdict;

    fn build_request(&self, state: &State, request: &Request) -> Request {
        let messages = self.input.build_messages(state, request);
        let mut llm_request = LlmRequest {
            model: request.llm_request.model.clone(),
            instructions: vec![InstructionBlock {
                role: Role::System,
                content: Message::text(Role::System, self.contract.system_prompt().to_string())
                    .content,
            }],
            messages,
            output: OutputParams {
                max_output_tokens: Some(self.runtime.max_output_tokens),
                response_format: Some(self.contract.response_format().clone()),
            },
            ..LlmRequest::default()
        };
        // Gateways that translate to other providers can drop `response_format` (Envoy AI
        // Gateway's Chat-to-Converse path does), but a forced tool call survives translation.
        // The verdict schema is therefore sent as the only tool, and the judge must call it.
        // A contract whose schema cannot be tool parameters keeps the plain `response_format`.
        if let Ok(Some(tool)) = structured_output_tool(self.contract.response_format()) {
            llm_request.tools = vec![tool];
            llm_request.tool_choice = Some(ToolChoice::Tool {
                name: STRUCTURED_OUTPUT_TOOL.to_string(),
            });
            llm_request.output.response_format = None;
        }
        Request {
            llm_request,
            raw_request: None,
            metadata: request.metadata.clone(),
        }
    }

    fn parse(&self, response: &AggLlmResponse) -> Result<Self::Verdict> {
        self.decoder.decode(response, &self.contract)
    }
}

/// Builds and parses requests for one algorithm-specific LLM judge.
pub trait Judge: Send + Sync {
    type Verdict: DeserializeOwned + Send + Sync;

    fn build_request(&self, state: &State, request: &Request) -> Request;

    fn parse(&self, response: &AggLlmResponse) -> Result<Self::Verdict> {
        parse_json_verdict(response)
    }
}

/// Converts a parsed verdict, or an unavailable verdict, into a routing classification.
/// Consider this as a deterministic policy which can act on the signals predicted from the classifier
/// and choose the route based on the verdict.
pub trait JudgePolicy: Send + Sync {
    type Verdict: Send + Sync;

    fn to_classification(&self, verdict: Option<&Self::Verdict>) -> Classification;
}

/// A classifier that calls one judge target and routes through its verdict policy.
pub struct JudgeClassifier<J, P> {
    judge: J,
    target: ModelId,
    policy: P,
}

impl<J, P> JudgeClassifier<J, P>
where
    J: Judge,
    P: JudgePolicy<Verdict = J::Verdict>,
{
    /// Combines a judge target with a verdict policy.
    pub fn new(judge: J, target: ModelId, policy: P) -> Self {
        Self {
            judge,
            target,
            policy,
        }
    }

    /// Consults the judge, yielding `None` when it is unavailable or unintelligible.
    ///
    /// A judge is an optimization, not a dependency: failing the caller's request because the
    /// judge is down would be worse than routing without it, so every failure — transport,
    /// mid-stream, or unparseable reply — is logged and folded into `None` for the policy's
    /// fallback branch. A closed driver stream is folded too; the algorithm's next driver
    /// call surfaces it, so nothing is masked.
    async fn verdict(
        &self,
        state: &mut State,
        request: &Request,
        driver: &Driver,
    ) -> Option<J::Verdict> {
        let judge_model = self.target.as_str();

        tracing::info!(target = judge_model, "consulting llm judge");
        let response = driver
            .call_model(
                self.judge.build_request(state, request),
                vec![self.target.clone()],
            )
            .await
            .inspect_err(|error| {
                report_fail_open(
                    judge_model,
                    safe_error_summary(error),
                    libsy_error_reason(error),
                )
            })
            .ok()?;
        let aggregate = response
            .llm_response
            .into_agg()
            .await
            .inspect_err(|error| {
                report_fail_open(
                    judge_model,
                    safe_client_error(error),
                    client_error_reason(error),
                )
            })
            .ok()?;
        self.judge
            .parse(&aggregate)
            .inspect_err(|error| report_parse_failure(judge_model, error, &aggregate))
            .ok()
    }
}

/// Logs and counts a judge reply that did not parse into a verdict.
///
/// The log carries the reply's shape and byte length so a parse failure can be told apart
/// from an empty reply or a model that ignored the forced tool, but never the reply itself:
/// it can quote the conversation back.
fn report_parse_failure(judge_model: &str, error: &LibsyError, response: &AggLlmResponse) {
    let (reply_shape, reply_len) = reply_shape(response);
    tracing::warn!(
        target: "libsy",
        judge_model,
        reason = "parse_error",
        reply_shape,
        reply_len,
        error = %safe_error_summary(error),
        "judge verdict unavailable; routing without one"
    );
    crate::observability::record_classifier_fail_open(judge_model, "parse_error");
}

/// Classifies a judge reply as `tool_call`, `text` or `empty`, with the byte length of the
/// part the parser reads.
fn reply_shape(response: &AggLlmResponse) -> (&'static str, usize) {
    if let Some(arguments) = forced_tool_arguments(response) {
        return ("tool_call", arguments_text(arguments).len());
    }
    let text = completion_text(response);
    if text.trim().is_empty() {
        ("empty", text.len())
    } else {
        ("text", text.len())
    }
}

/// Logs and counts a judge failure with a bounded label that excludes message content.
/// `error` must already be redacted: `LlmClientError::UpstreamHttp`'s `Display` interpolates the
/// raw upstream body, which can quote the conversation back. Callers pass a
/// `robustness::safe_*` summary rather than the error itself.
fn report_fail_open(judge_model: &str, error: String, reason: &'static str) {
    tracing::warn!(
        target: "libsy",
        judge_model,
        reason,
        error = %error,
        "judge verdict unavailable; routing without one"
    );
    crate::observability::record_classifier_fail_open(judge_model, reason);
}

/// Returns a bounded reason for a judge call that failed at the libsy layer.
pub(crate) fn libsy_error_reason(error: &LibsyError) -> &'static str {
    match error {
        LibsyError::ClientCall { source, .. } => client_error_reason(source),
        _ => "call_error",
    }
}

/// Returns a bounded reason from the error kind and HTTP status only.
fn client_error_reason(error: &LlmClientError) -> &'static str {
    match error {
        LlmClientError::Timeout { .. } => "timeout",
        LlmClientError::Transport { .. } => "transport",
        LlmClientError::UpstreamHttp { status, .. } if status.is_server_error() => "upstream_5xx",
        LlmClientError::UpstreamHttp { .. } => "upstream_non_5xx",
        LlmClientError::InvalidResponse { .. } | LlmClientError::ResponseTranslation(_) => {
            "invalid_response"
        }
        _ => "client_error",
    }
}

#[async_trait]
impl<J, P> Classifier<State> for JudgeClassifier<J, P>
where
    J: Judge,
    P: JudgePolicy<Verdict = J::Verdict>,
{
    async fn score(
        &self,
        state: &mut State,
        request: &mut Request,
        driver: Option<&Driver>,
    ) -> Result<(Classification, Option<Response>)> {
        // A missing driver is a broken composition, not an unavailable judge.
        let Some(driver) = driver else {
            return Err(LibsyError::AlgorithmError {
                message: format!(
                    "judge classifier for target {:?} requires a driver to call it",
                    self.target
                ),
            });
        };
        let verdict = self.verdict(state, request, driver).await;
        record_routing(Keys::JUDGE_MODEL, self.target.to_string());
        // Recorded before the policy runs, so a policy that rejects the verdict can
        // overwrite the status and give its own fallback reason.
        if verdict.is_some() {
            record_routing(Keys::JUDGE_VERDICT_STATUS, "ok");
        } else {
            record_routing(Keys::JUDGE_VERDICT_STATUS, "unavailable");
            record_routing(Keys::FALLBACK_REASON, "verdict_unavailable");
        }
        let classification = self.policy.to_classification(verdict.as_ref());
        if matches!(&classification, Classification::Scores(scores) if !scores.is_empty()) {
            record_routing(Keys::DECIDED_BY, "judge");
        }
        // A judge consultation is a side call, never the turn's answer.
        Ok((classification, None))
    }
}

/// Parses the verdict from the forced response tool's arguments, or else from the reply text.
///
/// The text path serves providers that honour `response_format` or ignore the forced tool.
fn parse_json_verdict<T: DeserializeOwned>(response: &AggLlmResponse) -> Result<T> {
    let parsed = match forced_tool_arguments(response) {
        // Arguments that did not parse as JSON upstream are kept as the raw string.
        Some(Value::String(raw)) => serde_json::from_str(strip_json_fence(raw.trim())),
        Some(arguments) => serde_json::from_value(arguments.clone()),
        None => {
            // Providers sometimes wrap otherwise valid JSON in a Markdown fence.
            let reply = completion_text(response);
            serde_json::from_str(strip_json_fence(reply.trim()))
        }
    };
    parsed.map_err(|err| LibsyError::AlgorithmError {
        message: format!(
            "judge reply did not parse as {}: {err}",
            std::any::type_name::<T>()
        ),
    })
}

/// Returns the arguments of the first call to the response tool in the first output.
fn forced_tool_arguments(response: &AggLlmResponse) -> Option<&Value> {
    response
        .outputs
        .first()?
        .content
        .iter()
        .find_map(|block| match block {
            ContentBlock::ToolCall(call) if call.name == STRUCTURED_OUTPUT_TOOL => {
                Some(&call.arguments)
            }
            _ => None,
        })
}

// The tool arguments as the JSON text the parser reads.
fn arguments_text(arguments: &Value) -> String {
    match arguments {
        Value::String(raw) => raw.clone(),
        other => other.to_string(),
    }
}

fn strip_json_fence(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("```") else {
        return text;
    };
    let rest = rest.strip_prefix("json").unwrap_or(rest);
    let rest = rest.trim_start_matches(['\n', '\r']);
    rest.strip_suffix("```").map(str::trim).unwrap_or(rest)
}

#[cfg(test)]
mod tests {
    use super::*;

    use futures::StreamExt;
    use http::StatusCode;
    use serde::Deserialize;
    use switchyard_protocol::{
        LlmClientError, ResponseOutput, ToolCall, text_request, text_response,
    };

    use crate::core::algorithm::Step;
    use crate::core::classifier::Score;
    use switchyard_protocol::{LlmResponse, LlmResponseChunk, Response};

    const VERDICT: &str = r#"{"ok":true}"#;

    #[derive(Debug, Deserialize, PartialEq)]
    struct TestVerdict {
        ok: bool,
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct ScoreVerdict {
        score: f64,
    }

    struct TestJudge;

    impl Judge for TestJudge {
        type Verdict = TestVerdict;

        fn build_request(&self, _state: &State, request: &Request) -> Request {
            request.clone()
        }
    }

    /// Reports only whether a verdict arrived.
    struct TestPolicy;

    impl JudgePolicy for TestPolicy {
        type Verdict = TestVerdict;

        fn to_classification(&self, verdict: Option<&Self::Verdict>) -> Classification {
            let target = if verdict.is_some() {
                "verdict"
            } else {
                "no-verdict"
            };
            Classification::Scores(vec![Score {
                target: ModelId::from(target),
                confidence: 1.0,
            }])
        }
    }

    fn classifier() -> JudgeClassifier<TestJudge, TestPolicy> {
        JudgeClassifier::new(TestJudge, ModelId::from("judge"), TestPolicy)
    }

    fn request() -> Request {
        Request {
            llm_request: text_request(Some("auto".to_string()), "judge this"),
            raw_request: None,
            metadata: None,
        }
    }

    #[test]
    fn the_verdict_is_read_from_the_completion() -> Result<()> {
        // A judge's reasoning is not its answer: only `content` carries the verdict, so a
        // reply that never reached one — a run truncated mid-thought — is an error rather
        // than a guess.
        let mut response = text_response(None, VERDICT);
        if let Some(output) = response.outputs.first_mut() {
            output.content.insert(
                0,
                ContentBlock::Reasoning {
                    text: r#"{"ok":false}"#.to_string(),
                    signature: None,
                    details: Vec::new(),
                },
            );
        }
        let parsed: TestVerdict = parse_json_verdict(&response)?;
        assert_eq!(parsed, TestVerdict { ok: true });

        assert!(parse_json_verdict::<TestVerdict>(&text_response(None, "still thinking")).is_err());
        Ok(())
    }

    #[test]
    fn typed_decoder_enforces_a_json_object_contract_locally() -> Result<()> {
        use super::super::classifier_contract::{
            ClassifierContractConfig, ClassifierResponseFormat,
        };

        let config = ClassifierContractConfig::default()
            .with_response_format_type(ClassifierResponseFormat::JsonObject);
        let contract = ClassifierContract::from_config(
            &config,
            "Return one JSON score.",
            r#"{
                "type": "json_schema",
                "json_schema": {
                    "name": "ScoreVerdict",
                    "schema": {
                        "type": "object",
                        "properties": {"score": {"type": "number"}},
                        "required": ["score"],
                        "additionalProperties": false
                    }
                }
            }"#,
        )?;
        let decoder = SerdeDecoder::<ScoreVerdict>::new();

        let error = decoder
            .decode(
                &text_response(None, r#"{"score":0.5,"unexpected":true}"#),
                &contract,
            )
            .expect_err("an extra property should fail the local schema");

        assert!(error.to_string().contains("did not match response_schema"));
        assert_eq!(
            decoder.decode(&text_response(None, r#"{"score":0.5}"#), &contract)?,
            ScoreVerdict { score: 0.5 }
        );
        Ok(())
    }

    fn buffered(completion: &str) -> Response {
        Response {
            llm_response: LlmResponse::Agg(text_response(None, completion)),
            metadata: None,
        }
    }

    fn streamed(chunks: Vec<LlmResponseChunk>) -> Response {
        Response {
            llm_response: LlmResponse::Stream(
                futures::stream::iter(chunks.into_iter().map(|chunk| Ok(chunk.into()))).boxed(),
            ),
            metadata: None,
        }
    }

    fn streamed_then_failing(chunk: LlmResponseChunk) -> Response {
        let items = futures::stream::iter([
            Ok(chunk.into()),
            Err(LlmClientError::Timeout {
                source: Box::new(std::io::Error::other("stream died")),
            }),
        ]);
        Response {
            llm_response: LlmResponse::Stream(items.boxed()),
            metadata: None,
        }
    }

    fn selected(classification: Classification) -> Result<ModelId> {
        classification
            .argmax(false)?
            .map(|score| score.target)
            .ok_or_else(|| LibsyError::AlgorithmError {
                message: "policy abstained".to_string(),
            })
    }

    /// Serves the single offloaded judge call with `reply` through a standalone step receiver.
    async fn score_served_with(reply: Result<Response>) -> Result<ModelId> {
        let (driver, step_rx) = Driver::new("test");
        let mut steps = tokio_stream::wrappers::ReceiverStream::new(step_rx);
        let classifier = classifier();
        let mut state = State::default();
        let mut request = request();

        let serve = async {
            if let Some(Ok(Step::CallModel(call))) = steps.next().await {
                let _ = call.respond(reply);
            }
        };
        let (classification, ()) = tokio::join!(
            classifier.score(&mut state, &mut request, Some(&driver)),
            serve
        );
        let (classification, _) = classification?;
        selected(classification)
    }

    #[tokio::test]
    async fn a_buffered_verdict_reaches_the_policy() -> Result<()> {
        assert_eq!(score_served_with(Ok(buffered(VERDICT))).await?, "verdict");
        Ok(())
    }

    #[tokio::test]
    async fn a_streamed_verdict_is_drained_before_parsing() -> Result<()> {
        let chunks = VERDICT
            .chars()
            .map(|character| LlmResponseChunk::TextDelta {
                index: 0,
                text: character.to_string(),
            })
            .collect();
        assert_eq!(score_served_with(Ok(streamed(chunks))).await?, "verdict");
        Ok(())
    }

    #[tokio::test]
    async fn an_in_band_stream_error_falls_back_to_the_policy() -> Result<()> {
        let chunks = vec![
            LlmResponseChunk::TextDelta {
                index: 0,
                text: "{\"ok\":".to_string(),
            },
            LlmResponseChunk::StreamError {
                message: "upstream exploded".to_string(),
            },
        ];
        assert_eq!(score_served_with(Ok(streamed(chunks))).await?, "no-verdict");
        Ok(())
    }

    #[tokio::test]
    async fn a_transport_failure_mid_stream_falls_back_to_the_policy() -> Result<()> {
        let partial = LlmResponseChunk::TextDelta {
            index: 0,
            text: "{\"ok\":".to_string(),
        };
        assert_eq!(
            score_served_with(Ok(streamed_then_failing(partial))).await?,
            "no-verdict"
        );
        Ok(())
    }

    #[tokio::test]
    async fn an_unparseable_reply_falls_back_to_the_policy() -> Result<()> {
        assert_eq!(
            score_served_with(Ok(buffered("sorry, I can't help with that"))).await?,
            "no-verdict"
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_failed_judge_call_falls_back_to_the_policy() -> Result<()> {
        let error = LibsyError::client_call(
            "judge",
            LlmClientError::Timeout {
                source: Box::new(std::io::Error::other("judge unreachable")),
            },
        );
        assert_eq!(score_served_with(Err(error)).await?, "no-verdict");
        Ok(())
    }

    #[test]
    fn client_errors_map_to_bounded_fail_open_reasons() {
        let cases = vec![
            (
                LlmClientError::Timeout {
                    source: "deadline exceeded".into(),
                },
                "timeout",
            ),
            (
                LlmClientError::Transport {
                    source: "connection refused".into(),
                },
                "transport",
            ),
            (
                LlmClientError::UpstreamHttp {
                    status: StatusCode::INTERNAL_SERVER_ERROR,
                    body: "server error".to_string(),
                },
                "upstream_5xx",
            ),
            (
                LlmClientError::UpstreamHttp {
                    status: StatusCode::FOUND,
                    body: "redirect".to_string(),
                },
                "upstream_non_5xx",
            ),
            (
                LlmClientError::InvalidResponse {
                    source: "invalid JSON".into(),
                },
                "invalid_response",
            ),
            (
                LlmClientError::General("unexpected client failure".to_string()),
                "client_error",
            ),
        ];
        for (error, expected) in cases {
            assert_eq!(client_error_reason(&error), expected);
        }

        let error = LibsyError::AlgorithmError {
            message: "driver failed".to_string(),
        };
        assert_eq!(libsy_error_reason(&error), "call_error");
    }

    #[tokio::test]
    async fn a_missing_driver_is_an_error_not_a_fallback() -> Result<()> {
        let mut request = request();
        let error = classifier()
            .score(&mut State::default(), &mut request, None)
            .await
            .err()
            .ok_or_else(|| LibsyError::AlgorithmError {
                message: "expected a missing-driver error".to_string(),
            })?;

        assert!(
            matches!(&error, LibsyError::AlgorithmError { message } if message.contains("judge")),
            "unexpected error: {error}"
        );
        Ok(())
    }

    /// Passes the caller's messages through unchanged.
    struct EchoInput;

    impl ClassifierInput for EchoInput {
        fn build_messages(&self, _state: &State, request: &Request) -> Vec<Message> {
            request.llm_request.messages.clone()
        }
    }

    fn verdict_schema() -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": false
        })
    }

    fn structured_judge() -> Result<StructuredJudge<EchoInput, SerdeDecoder<TestVerdict>>> {
        use super::super::classifier_contract::ClassifierContractConfig;

        let response_format = serde_json::json!({
            "type": "json_schema",
            "json_schema": {"name": "TestVerdict", "strict": true, "schema": verdict_schema()}
        });
        let contract = ClassifierContract::from_config(
            &ClassifierContractConfig::default(),
            "Judge the task.",
            &response_format.to_string(),
        )?;
        Ok(StructuredJudge::new(
            EchoInput,
            contract,
            SerdeDecoder::new(),
            JudgeRuntimeConfig::new(256)?,
        ))
    }

    fn tool_call_response(arguments: Value) -> AggLlmResponse {
        AggLlmResponse {
            outputs: vec![ResponseOutput {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolCall(ToolCall {
                    id: "call_1".to_string(),
                    name: STRUCTURED_OUTPUT_TOOL.to_string(),
                    arguments,
                })],
                stop_reason: None,
            }],
            ..AggLlmResponse::default()
        }
    }

    #[test]
    fn a_structured_judge_forces_the_verdict_tool() -> Result<()> {
        // The schema travels as the one forced tool, because gateways may drop
        // `response_format` on the way to the provider.
        let built = structured_judge()?.build_request(&State::default(), &request());
        let llm_request = &built.llm_request;

        assert_eq!(llm_request.tools.len(), 1);
        assert_eq!(llm_request.tools[0].name, STRUCTURED_OUTPUT_TOOL);
        assert_eq!(llm_request.tools[0].parameters, verdict_schema());
        assert_eq!(llm_request.tools[0].strict, Some(true));
        assert_eq!(
            llm_request.tool_choice,
            Some(ToolChoice::Tool {
                name: STRUCTURED_OUTPUT_TOOL.to_string()
            })
        );
        assert_eq!(llm_request.output.response_format, None);
        assert_eq!(llm_request.output.max_output_tokens, Some(256));
        Ok(())
    }

    #[test]
    fn a_forced_tool_call_parses_into_the_verdict() -> Result<()> {
        let judge = structured_judge()?;
        // Decoded arguments, and arguments the codec kept as raw JSON text.
        for arguments in [
            serde_json::json!({"ok": true}),
            Value::String(r#"{"ok":true}"#.to_string()),
        ] {
            assert_eq!(
                judge.parse(&tool_call_response(arguments))?,
                TestVerdict { ok: true }
            );
        }
        Ok(())
    }

    #[test]
    fn a_plain_json_text_reply_still_parses() -> Result<()> {
        // Providers that honour `response_format`, or ignore the forced tool, answer in text.
        let judge = structured_judge()?;
        assert_eq!(
            judge.parse(&text_response(None, VERDICT))?,
            TestVerdict { ok: true }
        );
        Ok(())
    }

    /// Collects formatted log output in memory.
    #[derive(Clone, Default)]
    struct LogBuffer(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for LogBuffer {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if let Ok(mut buffer) = self.0.lock() {
                buffer.extend_from_slice(bytes);
            }
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl LogBuffer {
        fn contents(&self) -> String {
            self.0
                .lock()
                .map(|buffer| String::from_utf8_lossy(&buffer).into_owned())
                .unwrap_or_default()
        }
    }

    #[test]
    fn a_free_text_reply_logs_its_shape_but_not_its_content() -> Result<()> {
        // The reply can quote the conversation, so the log names its shape and length only.
        const REPLY: &str = "Here is my private analysis of the task";
        let response = text_response(None, REPLY);
        let error = structured_judge()?
            .parse(&response)
            .expect_err("free text is not a verdict");

        let logs = LogBuffer::default();
        let writer = logs.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .finish();
        tracing::subscriber::with_default(subscriber, || {
            report_parse_failure("judge", &error, &response);
        });

        let logs = logs.contents();
        assert!(logs.contains("parse_error"), "{logs}");
        assert!(logs.contains("reply_shape=\"text\""), "{logs}");
        assert!(
            logs.contains(&format!("reply_len={}", REPLY.len())),
            "{logs}"
        );
        assert!(!logs.contains("private analysis"), "{logs}");
        Ok(())
    }

    #[test]
    fn reply_shapes_are_bounded_labels() {
        assert_eq!(
            reply_shape(&tool_call_response(serde_json::json!({"ok": true}))),
            ("tool_call", r#"{"ok":true}"#.len())
        );
        assert_eq!(reply_shape(&text_response(None, "hi")), ("text", 2));
        assert_eq!(reply_shape(&text_response(None, "  ")), ("empty", 2));
        assert_eq!(reply_shape(&AggLlmResponse::default()), ("empty", 0));
    }

    #[test]
    fn fenced_replies_parse_as_verdicts() -> Result<()> {
        let judge = TestJudge;
        for reply in ["```json\n{\"ok\":true}\n```", "```\n{\"ok\":true}\n```"] {
            assert!(judge.parse(&text_response(None, reply))?.ok);
        }
        Ok(())
    }
}
