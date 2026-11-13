// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-neutral telemetry vocabulary shared across Switchyard crates.
//!
//! This module defines the stable string contract for route-selection and
//! Langfuse observations: generic OpenTelemetry attribute keys, Langfuse-native
//! metadata keys, and the allowed values for response origin and call role. It
//! carries only names and values: no instrumentation, exporter, or metric
//! behavior, so any crate can emit or read the same vocabulary without
//! depending on a telemetry backend.

use std::fmt;

/// Generic OpenTelemetry attribute keys for route-selection telemetry.
///
/// Backend-neutral keys an exporter maps onto its own attribute namespace.
/// They are associated `const`s so callers may use them in attribute tables
/// and span metadata without allocation.
pub struct GenericKeys;

impl GenericKeys {
    /// Route the request was sent for.
    pub const ROUTE_ID: &str = "switchyard.route.id";
    /// Routing algorithm that produced the selection.
    pub const ALGORITHM: &str = "switchyard.routing.algorithm";
    /// Target the routing algorithm selected.
    pub const SELECTED_TARGET: &str = "switchyard.routing.selected_target";
    /// Target that actually served the response.
    pub const SERVED_TARGET: &str = "switchyard.response.served_target";
    /// Origin of the served response, one of [`ResponseOrigin`].
    pub const RESPONSE_ORIGIN: &str = "switchyard.response.origin";
    /// Role of the call within the route, one of [`CallRole`].
    pub const CALL_ROLE: &str = "switchyard.call.role";
    /// Target the call was directed at.
    pub const CALL_TARGET: &str = "switchyard.call.target";
}

/// Attribute keys explaining why a routing algorithm picked its target.
///
/// Every value is a bounded label, number, or boolean, except the keys in
/// [`RoutingDecisionKeys::CONTENT`]: those carry judge-written free text derived from the
/// request and must be handled like prompt and completion content.
pub struct RoutingDecisionKeys;

impl RoutingDecisionKeys {
    /// Classifier mode: `capability`, `escalation`, or `custom`.
    pub const MODE: &str = "switchyard.routing.mode";
    /// Tier of the selected target, such as `weak` or `strong`.
    pub const TIER: &str = "switchyard.routing.tier";
    /// Component that decided the turn: `judge`, `affinity`, `latch`, `fallback`, or a stage
    /// decision source such as `dimensions`.
    pub const DECIDED_BY: &str = "switchyard.routing.decided_by";
    /// Why the turn fell back to a default, such as `verdict_unavailable`.
    pub const FALLBACK_REASON: &str = "switchyard.routing.fallback_reason";
    /// Target that produced the judge verdict.
    pub const JUDGE_MODEL: &str = "switchyard.routing.judge.model";
    /// Judge verdict status: `ok`, `unavailable`, or `invalid`.
    pub const JUDGE_VERDICT_STATUS: &str = "switchyard.routing.judge.verdict_status";
    /// Capability judge's solve probability.
    pub const JUDGE_P_SOLVE: &str = "switchyard.routing.judge.p_solve";
    /// Solve probability the capability verdict had to reach for the efficient target.
    pub const JUDGE_THRESHOLD: &str = "switchyard.routing.judge.threshold";
    /// Capability boundary of a valid capability verdict.
    pub const JUDGE_CAPABILITY_BOUNDARY: &str = "switchyard.routing.judge.capability_boundary";
    /// Primary rule of a valid capability verdict.
    pub const JUDGE_PRIMARY_RULE: &str = "switchyard.routing.judge.primary_rule";
    /// Capability judge's one-line reason. Content.
    pub const JUDGE_CRUX: &str = "switchyard.routing.judge.crux";
    /// Whether the escalation judge asked to escalate.
    pub const ESCALATION_ESCALATE: &str = "switchyard.routing.escalation.escalate";
    /// Consecutive escalate verdicts after this turn.
    pub const ESCALATION_STREAK: &str = "switchyard.routing.escalation.streak";
    /// Escalate verdicts required to latch the session to the capable target.
    pub const ESCALATION_CONFIRMATIONS: &str = "switchyard.routing.escalation.confirmations";
    /// Whether the session is latched to the capable target.
    pub const ESCALATION_LATCHED: &str = "switchyard.routing.escalation.latched";
    /// Escalation judge's reason. Content.
    pub const ESCALATION_REASON: &str = "switchyard.routing.escalation.reason";
    /// Configured target label a custom verdict selected.
    pub const CUSTOM_LABEL: &str = "switchyard.routing.custom.label";
    /// Stage signal scorer's probability for a turn the signals resolved.
    pub const STAGE_PROBABILITY: &str = "switchyard.routing.stage.probability";
    /// Stage signal scorer's confidence for a turn the signals resolved.
    pub const STAGE_CONFIDENCE: &str = "switchyard.routing.stage.confidence";

    /// Keys that carry judge-written free text.
    pub const CONTENT: &[&str] = &[Self::JUDGE_CRUX, Self::ESCALATION_REASON];
}

/// Langfuse-native metadata keys for route-selection observations.
///
/// The literal key strings Langfuse expects on an observation's `metadata`
/// object. `const`s so callers can build observation metadata without
/// allocation.
pub struct LangfuseKeys;

impl LangfuseKeys {
    /// Session identifier that groups an observation.
    pub const SESSION_ID: &str = "langfuse.session.id";
    /// User identifier attributed to an observation.
    pub const USER_ID: &str = "langfuse.user.id";
    /// Observation metadata key for the route id.
    pub const ROUTE_ID: &str = "langfuse.observation.metadata.switchyard.route_id";
    /// Observation metadata key for the routing algorithm.
    pub const ALGORITHM: &str = "langfuse.observation.metadata.switchyard.algorithm";
    /// Observation metadata key for the selected target.
    pub const SELECTED_TARGET: &str = "langfuse.observation.metadata.switchyard.selected_target";
    /// Observation metadata key for the served target.
    pub const SERVED_TARGET: &str = "langfuse.observation.metadata.switchyard.served_target";
    /// Observation metadata key for the response origin.
    pub const RESPONSE_ORIGIN: &str = "langfuse.observation.metadata.switchyard.response_origin";
    /// Observation metadata key for the call role.
    pub const CALL_ROLE: &str = "langfuse.observation.metadata.switchyard.call_role";
    /// Observation metadata key for the call target.
    pub const CALL_TARGET: &str = "langfuse.observation.metadata.switchyard.call_target";
}

/// Origin of a served response.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResponseOrigin {
    /// Served by a backing target model.
    Target,
    /// Generated by Switchyard's routing layer rather than a target model.
    RoutingGenerated,
}

impl ResponseOrigin {
    /// Returns the stable contract value for the origin.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Target => "target",
            Self::RoutingGenerated => "routing_generated",
        }
    }
}

impl fmt::Display for ResponseOrigin {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Role a call plays within a routed request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CallRole {
    /// A call Switchyard makes while selecting a route, not for the answer.
    RoutingDependency,
    /// The primary model serving the answer.
    AnswerPrimary,
    /// A fallback model serving the answer.
    AnswerFallback,
}

impl CallRole {
    /// Returns the stable contract value for the role.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RoutingDependency => "routing_dependency",
            Self::AnswerPrimary => "answer_primary",
            Self::AnswerFallback => "answer_fallback",
        }
    }
}

impl fmt::Display for CallRole {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_keys_match_the_contract() {
        assert_eq!(GenericKeys::ROUTE_ID, "switchyard.route.id");
        assert_eq!(GenericKeys::ALGORITHM, "switchyard.routing.algorithm");
        assert_eq!(
            GenericKeys::SELECTED_TARGET,
            "switchyard.routing.selected_target"
        );
        assert_eq!(
            GenericKeys::SERVED_TARGET,
            "switchyard.response.served_target"
        );
        assert_eq!(GenericKeys::RESPONSE_ORIGIN, "switchyard.response.origin");
        assert_eq!(GenericKeys::CALL_ROLE, "switchyard.call.role");
        assert_eq!(GenericKeys::CALL_TARGET, "switchyard.call.target");
    }

    #[test]
    fn routing_decision_keys_stay_in_the_routing_namespace() {
        assert_eq!(
            RoutingDecisionKeys::DECIDED_BY,
            "switchyard.routing.decided_by"
        );
        assert_eq!(
            RoutingDecisionKeys::JUDGE_P_SOLVE,
            "switchyard.routing.judge.p_solve"
        );
        assert_eq!(
            RoutingDecisionKeys::CONTENT,
            [
                "switchyard.routing.judge.crux",
                "switchyard.routing.escalation.reason"
            ]
        );
    }

    #[test]
    fn langfuse_keys_match_the_contract() {
        assert_eq!(LangfuseKeys::SESSION_ID, "langfuse.session.id");
        assert_eq!(LangfuseKeys::USER_ID, "langfuse.user.id");
        assert_eq!(
            LangfuseKeys::ROUTE_ID,
            "langfuse.observation.metadata.switchyard.route_id"
        );
        assert_eq!(
            LangfuseKeys::ALGORITHM,
            "langfuse.observation.metadata.switchyard.algorithm"
        );
        assert_eq!(
            LangfuseKeys::SELECTED_TARGET,
            "langfuse.observation.metadata.switchyard.selected_target"
        );
        assert_eq!(
            LangfuseKeys::SERVED_TARGET,
            "langfuse.observation.metadata.switchyard.served_target"
        );
        assert_eq!(
            LangfuseKeys::RESPONSE_ORIGIN,
            "langfuse.observation.metadata.switchyard.response_origin"
        );
        assert_eq!(
            LangfuseKeys::CALL_ROLE,
            "langfuse.observation.metadata.switchyard.call_role"
        );
        assert_eq!(
            LangfuseKeys::CALL_TARGET,
            "langfuse.observation.metadata.switchyard.call_target"
        );
    }

    #[test]
    fn allowed_values_match_the_contract() {
        assert_eq!(ResponseOrigin::Target.as_str(), "target");
        assert_eq!(
            ResponseOrigin::RoutingGenerated.as_str(),
            "routing_generated"
        );
        assert_eq!(CallRole::RoutingDependency.as_str(), "routing_dependency");
        assert_eq!(CallRole::AnswerPrimary.as_str(), "answer_primary");
        assert_eq!(CallRole::AnswerFallback.as_str(), "answer_fallback");
    }
}
