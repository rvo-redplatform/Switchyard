// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-target capabilities declared in a deployment's `[targets.<name>] capabilities` list.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use libsy::{Algorithm, Passthrough, Random};
use serde_json::Value;
use switchyard_protocol::{ContentBlock, LlmClientError, LlmRequest, ModelId};

use crate::RunnerError;

/// One capability a target can declare.
///
/// The wire names match the keys an operator (or a config generator) writes in
/// `capabilities = [...]`.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Capability {
    Text,
    Image,
    Video,
    Tools,
    Reasoning,
    StructuredOutput,
    Caching,
    Streaming,
}

impl Capability {
    /// Every capability, in the order the wire format lists them.
    pub const ALL: [Self; 8] = [
        Self::Text,
        Self::Image,
        Self::Video,
        Self::Tools,
        Self::Reasoning,
        Self::StructuredOutput,
        Self::Caching,
        Self::Streaming,
    ];

    /// Returns the wire name used in `capabilities = [...]`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Image => "image",
            Self::Video => "video",
            Self::Tools => "tools",
            Self::Reasoning => "reasoning",
            Self::StructuredOutput => "structured_output",
            Self::Caching => "caching",
            Self::Streaming => "streaming",
        }
    }

    /// Parses a wire name. Unknown names return `None`.
    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|capability| capability.as_str() == value)
    }
}

/// What one target supports end to end.
///
/// An undeclared target (no `capabilities` key) is unknown: it supports every
/// capability as far as routing is concerned, so nothing is filtered or refused
/// because of it. A declared list is exhaustive.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TargetCapabilities {
    declared: Option<BTreeSet<Capability>>,
}

impl TargetCapabilities {
    /// A target whose capabilities are unknown.
    pub fn undeclared() -> Self {
        Self::default()
    }

    /// A target that supports exactly `capabilities`.
    pub fn declared(capabilities: impl IntoIterator<Item = Capability>) -> Self {
        Self {
            declared: Some(capabilities.into_iter().collect()),
        }
    }

    /// Whether the target declared a capability list.
    pub fn is_declared(&self) -> bool {
        self.declared.is_some()
    }

    /// Whether the target may serve a request that needs `capability`.
    ///
    /// Always true for an undeclared target. Structured output is also
    /// supported wherever tools are, because it is served as a forced tool call.
    pub fn supports(&self, capability: Capability) -> bool {
        self.declared.as_ref().is_none_or(|declared| {
            declared.contains(&capability)
                || (capability == Capability::StructuredOutput
                    && declared.contains(&Capability::Tools))
        })
    }

    /// The capabilities every target supports, or `None` when any target is
    /// undeclared or there are no targets.
    pub fn intersection<'a>(
        targets: impl IntoIterator<Item = &'a TargetCapabilities>,
    ) -> Option<BTreeSet<Capability>> {
        let mut result: Option<BTreeSet<Capability>> = None;
        for target in targets {
            let declared = target.declared.as_ref()?;
            result = Some(match result {
                None => declared.clone(),
                Some(acc) => acc.intersection(declared).copied().collect(),
            });
        }
        result
    }
}

/// The capabilities one request needs from the target that serves it.
///
/// Only features a target can declare are tracked. Text is always assumed, and
/// prompt caching markers are hints a target may ignore, so neither filters.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RequestFeatures {
    required: BTreeSet<Capability>,
}

impl RequestFeatures {
    /// Extracts the needed capabilities from a normalized request.
    pub fn of(request: &LlmRequest) -> Self {
        let mut required = BTreeSet::new();
        let blocks = request
            .instructions
            .iter()
            .flat_map(|instruction| instruction.content.iter())
            .chain(
                request
                    .messages
                    .iter()
                    .flat_map(|message| message.content.iter()),
            );
        for block in blocks {
            require_block(block, &mut required);
        }
        if !request.tools.is_empty() {
            required.insert(Capability::Tools);
        }
        if wants_reasoning(request) {
            required.insert(Capability::Reasoning);
        }
        if wants_structured_output(request.output.response_format.as_ref()) {
            required.insert(Capability::StructuredOutput);
        }
        if request.stream {
            required.insert(Capability::Streaming);
        }
        Self { required }
    }

    /// Builds a feature set directly, mainly for tests.
    pub fn from_capabilities(capabilities: impl IntoIterator<Item = Capability>) -> Self {
        Self {
            required: capabilities.into_iter().collect(),
        }
    }

    /// Whether the request needs nothing beyond text.
    pub fn is_empty(&self) -> bool {
        self.required.is_empty()
    }

    /// The needed capabilities in wire order.
    pub fn required(&self) -> impl Iterator<Item = Capability> + '_ {
        self.required.iter().copied()
    }

    /// Whether `target` supports every needed capability.
    pub fn fit(&self, target: &TargetCapabilities) -> bool {
        self.required()
            .all(|capability| target.supports(capability))
    }
}

// Records the capability a content block needs, looking inside tool results.
fn require_block(block: &ContentBlock, required: &mut BTreeSet<Capability>) {
    match block {
        ContentBlock::Image { .. } => {
            required.insert(Capability::Image);
        }
        ContentBlock::Video { .. } => {
            required.insert(Capability::Video);
        }
        ContentBlock::ToolCall(_) => {
            required.insert(Capability::Tools);
        }
        ContentBlock::ToolResult(result) => {
            required.insert(Capability::Tools);
            for inner in &result.content {
                require_block(inner, required);
            }
        }
        _ => {}
    }
}

// Reasoning is requested by a non-`none` effort or by an Anthropic-style
// `thinking` object that turns thinking on. `{"type": "disabled"}` does not count.
fn wants_reasoning(request: &LlmRequest) -> bool {
    if request
        .reasoning
        .effort
        .as_deref()
        .is_some_and(|effort| !effort.eq_ignore_ascii_case("none"))
    {
        return true;
    }
    matches!(
        request
            .reasoning
            .raw
            .as_ref()
            .and_then(|raw| raw.get("type"))
            .and_then(Value::as_str),
        Some("enabled" | "adaptive")
    )
}

// Any response format other than plain text asks for structured output.
fn wants_structured_output(format: Option<&Value>) -> bool {
    format.is_some_and(|format| format.get("type").and_then(Value::as_str) != Some("text"))
}

/// How a route narrows to the targets that fit a request.
#[derive(Clone, Copy, Debug)]
pub enum FilterStrategy {
    /// Pick randomly among the fitting targets, keeping their relative weights.
    Random {
        /// Seed for the narrowed random choice.
        seed: Option<u64>,
    },
    /// Send the request to the first fitting target in configured order. Used
    /// by single-target and judge routes, where a judge cannot choose a target
    /// that does not fit.
    FirstEligible,
}

struct FilterTarget {
    model: ModelId,
    capabilities: TargetCapabilities,
    weight: f64,
}

/// Narrows a route's serving targets to the ones whose declared capabilities
/// cover a request.
///
/// When every target fits, the route's own algorithm runs unchanged. When none
/// fits, the request is refused with an invalid-request error (HTTP 400). In
/// between, a narrowed algorithm over the fitting targets serves the request;
/// narrowed algorithms are cached per fitting set so a seeded random route keeps
/// its sequence.
pub struct TargetFilter {
    route: String,
    targets: Vec<FilterTarget>,
    strategy: FilterStrategy,
    narrowed: Mutex<HashMap<Vec<usize>, Arc<dyn Algorithm>>>,
}

impl TargetFilter {
    /// Creates a filter over `targets` in routing order. `weights` follow
    /// target order and default to one each.
    pub fn new(
        route: impl Into<String>,
        targets: Vec<(ModelId, TargetCapabilities)>,
        weights: Option<Vec<f64>>,
        strategy: FilterStrategy,
    ) -> Self {
        let weights = weights.unwrap_or_default();
        let targets = targets
            .into_iter()
            .enumerate()
            .map(|(index, (model, capabilities))| FilterTarget {
                model,
                capabilities,
                weight: weights.get(index).copied().unwrap_or(1.0),
            })
            .collect();
        Self {
            route: route.into(),
            targets,
            strategy,
            narrowed: Mutex::new(HashMap::new()),
        }
    }

    /// Returns the algorithm that should serve a request with `features`, or
    /// `None` when the route's configured algorithm can serve it unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`LlmClientError::InvalidRequest`] when no target fits.
    pub fn select(
        &self,
        features: &RequestFeatures,
    ) -> Result<Option<Arc<dyn Algorithm>>, RunnerError> {
        if features.is_empty() {
            return Ok(None);
        }
        let eligible = self
            .targets
            .iter()
            .enumerate()
            .filter(|(_, target)| features.fit(&target.capabilities))
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if eligible.len() == self.targets.len() {
            return Ok(None);
        }
        if eligible.is_empty() {
            tracing::info!(route = %self.route, "no route target supports the request");
            return Err(self.no_fit(features));
        }
        let mut narrowed = match self.narrowed.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let cached = narrowed.get(&eligible).cloned();
        let algorithm = match cached {
            Some(algorithm) => algorithm,
            None => {
                let algorithm = self.narrow(&eligible)?;
                narrowed.insert(eligible.clone(), Arc::clone(&algorithm));
                algorithm
            }
        };
        tracing::info!(
            route = %self.route,
            eligible = ?self.models(&eligible),
            "route narrowed to targets that support the request"
        );
        Ok(Some(algorithm))
    }

    fn models(&self, indices: &[usize]) -> Vec<ModelId> {
        indices
            .iter()
            .filter_map(|index| self.targets.get(*index))
            .map(|target| target.model.clone())
            .collect()
    }

    // Builds the algorithm that serves only the fitting targets.
    fn narrow(&self, eligible: &[usize]) -> Result<Arc<dyn Algorithm>, RunnerError> {
        let models = self.models(eligible);
        let Some(first) = models.first().cloned() else {
            return Err(RunnerError::configuration("no eligible route target"));
        };
        if let FilterStrategy::Random { seed } = self.strategy
            && models.len() > 1
        {
            let weights = eligible
                .iter()
                .filter_map(|index| self.targets.get(*index))
                .map(|target| target.weight)
                .collect::<Vec<_>>();
            if weights.iter().any(|weight| *weight > 0.0) {
                return Ok(Arc::new(Random::new(models, Some(weights), seed)?));
            }
        }
        Ok(Arc::new(Passthrough::new(first)))
    }

    // Names the needed capabilities that no target supports, or all needed
    // ones when each is supported somewhere but never all together.
    fn no_fit(&self, features: &RequestFeatures) -> RunnerError {
        let mut missing = features
            .required()
            .filter(|capability| {
                !self
                    .targets
                    .iter()
                    .any(|target| target.capabilities.supports(*capability))
            })
            .map(Capability::as_str)
            .collect::<Vec<_>>();
        if missing.is_empty() {
            missing = features.required().map(Capability::as_str).collect();
        }
        RunnerError::Client(LlmClientError::InvalidRequest {
            message: format!(
                "model {} cannot serve this request: no target supports {}",
                self.route,
                missing.join(" + ")
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use switchyard_protocol::{ImageSource, Message, Role, ToolDefinition, text_request};

    #[test]
    fn wire_names_round_trip() {
        for capability in Capability::ALL {
            assert_eq!(Capability::parse(capability.as_str()), Some(capability));
        }
        assert_eq!(Capability::parse("telepathy"), None);
    }

    #[test]
    fn undeclared_supports_everything_and_blocks_intersection() {
        let undeclared = TargetCapabilities::undeclared();
        assert!(!undeclared.is_declared());
        assert!(Capability::ALL.into_iter().all(|c| undeclared.supports(c)));

        let declared = TargetCapabilities::declared([Capability::Text, Capability::Tools]);
        assert_eq!(
            TargetCapabilities::intersection([&declared, &undeclared]),
            None
        );
        assert_eq!(
            TargetCapabilities::intersection(std::iter::empty::<&TargetCapabilities>()),
            None
        );
    }

    #[test]
    fn intersection_keeps_only_shared_capabilities() {
        let haiku = TargetCapabilities::declared([
            Capability::Text,
            Capability::Image,
            Capability::Tools,
            Capability::Reasoning,
        ]);
        let nova =
            TargetCapabilities::declared([Capability::Text, Capability::Image, Capability::Tools]);
        assert!(!nova.supports(Capability::Reasoning));
        assert_eq!(
            TargetCapabilities::intersection([&haiku, &nova]),
            Some(BTreeSet::from([
                Capability::Text,
                Capability::Image,
                Capability::Tools
            ]))
        );
    }

    #[test]
    fn tools_imply_structured_output() {
        let tools = TargetCapabilities::declared([Capability::Text, Capability::Tools]);
        assert!(tools.supports(Capability::StructuredOutput));
        let text = TargetCapabilities::declared([Capability::Text]);
        assert!(!text.supports(Capability::StructuredOutput));
        let native = TargetCapabilities::declared([Capability::Text, Capability::StructuredOutput]);
        assert!(native.supports(Capability::StructuredOutput));
    }

    fn image_request() -> LlmRequest {
        let mut request = text_request(Some("route".to_string()), "describe this");
        request.messages.push(Message {
            role: Role::User,
            content: vec![ContentBlock::Image {
                source: ImageSource::Url {
                    url: "data:image/png;base64,AAAA".to_string(),
                    detail: None,
                },
            }],
        });
        request
    }

    #[test]
    fn features_cover_media_tools_reasoning_and_output_shape() {
        assert!(RequestFeatures::of(&text_request(None, "hi")).is_empty());

        let mut request = image_request();
        request.tools.push(ToolDefinition {
            name: "lookup".to_string(),
            description: None,
            parameters: json!({"type": "object"}),
            strict: None,
        });
        request.reasoning.raw = Some(json!({"type": "adaptive"}));
        request.output.response_format = Some(json!({"type": "json_schema"}));
        request.stream = true;
        assert_eq!(
            RequestFeatures::of(&request).required().collect::<Vec<_>>(),
            [
                Capability::Image,
                Capability::Tools,
                Capability::Reasoning,
                Capability::StructuredOutput,
                Capability::Streaming,
            ]
        );

        let mut quiet = text_request(None, "hi");
        quiet.reasoning.raw = Some(json!({"type": "disabled"}));
        quiet.reasoning.effort = Some("none".to_string());
        quiet.output.response_format = Some(json!({"type": "text"}));
        assert!(RequestFeatures::of(&quiet).is_empty());
    }

    fn filter(strategy: FilterStrategy) -> TargetFilter {
        TargetFilter::new(
            "rp-fast",
            vec![
                (
                    ModelId::from("nova-lite"),
                    TargetCapabilities::declared([Capability::Text, Capability::Image]),
                ),
                (
                    ModelId::from("haiku"),
                    TargetCapabilities::declared([
                        Capability::Text,
                        Capability::Image,
                        Capability::Reasoning,
                    ]),
                ),
            ],
            None,
            strategy,
        )
    }

    #[test]
    fn filter_keeps_the_route_algorithm_when_every_target_fits() {
        let filter = filter(FilterStrategy::Random { seed: Some(7) });
        let image = RequestFeatures::from_capabilities([Capability::Image]);
        assert!(matches!(filter.select(&image), Ok(None)));
        assert!(matches!(
            filter.select(&RequestFeatures::default()),
            Ok(None)
        ));
    }

    #[test]
    fn filter_narrows_to_the_fitting_target() {
        let filter = filter(FilterStrategy::Random { seed: Some(7) });
        let reasoning = RequestFeatures::from_capabilities([Capability::Reasoning]);
        match filter.select(&reasoning) {
            Ok(Some(algorithm)) => assert_eq!(algorithm.name(), "passthrough"),
            Ok(None) => panic!("a partial fit must narrow the route"),
            Err(error) => panic!("a partial fit must not be refused: {error}"),
        }
    }

    #[test]
    fn filter_refuses_when_no_target_fits() {
        let filter = filter(FilterStrategy::FirstEligible);
        let video = RequestFeatures::from_capabilities([Capability::Video, Capability::Image]);
        match filter.select(&video) {
            Err(RunnerError::Client(LlmClientError::InvalidRequest { message, .. })) => {
                assert!(message.contains("rp-fast"), "{message}");
                assert!(message.contains("video"), "{message}");
                assert!(!message.contains("image"), "{message}");
            }
            Err(error) => panic!("expected an invalid request, got {error}"),
            Ok(_) => panic!("no target supports video"),
        }
    }
}
