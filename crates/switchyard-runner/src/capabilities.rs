// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-target capabilities declared in a deployment's `[targets.<name>] capabilities` list.

use std::collections::BTreeSet;

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
    /// Always true for an undeclared target.
    pub fn supports(&self, capability: Capability) -> bool {
        self.declared
            .as_ref()
            .is_none_or(|declared| declared.contains(&capability))
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
