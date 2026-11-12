// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenTelemetry metrics plus `tracing` spans and structured logs for the
//! algorithm layer.
//!
//! [`Algorithm::run_stream`](crate::Algorithm::run_stream) and [`Driver`] call these
//! helpers around the routing outcome and the offload boundary, so every algorithm is
//! instrumented from the outside and carries no telemetry code of its own. The provider
//! call on the other side of the offload belongs to the host, and is instrumented by
//! whoever makes it. Metrics record through the
//! OpenTelemetry **global** meter provider under the `switchyard` scope — the host
//! installs an SDK provider and exporters; with none installed, recording is a
//! no-op. Spans and logs use the `tracing` facade (the async-native surface the
//! OpenTelemetry ecosystem bridges with `tracing-opentelemetry` /
//! `opentelemetry-appender-tracing`), so the host's subscriber decides where
//! they go. Method spans use `#[tracing::instrument]`; the `libsy.run` span is
//! attached to the spawned run task with [`tracing::Instrument`]. Neither holds
//! a [`Span::enter`] guard across an `.await` — a suspended task would leave
//! the span entered on its executor thread, mis-parenting every span other
//! tasks create there (see the `tracing` docs on spans in asynchronous code).
//!
//! Instrument names use the OTel dotted form with the unit baked into the name
//! (`switchyard.run_duration_ms`), matching the switchyard metric surface; a
//! Prometheus exporter sanitizes them to `switchyard_run_duration_ms`. Attribute
//! cardinality is bounded: `algorithm` and `selected_model` are small
//! configured sets and `outcome` is `ok`/`error`. Nothing per-request becomes a
//! metric attribute — correlation ids ride on the `libsy.run` span instead.
//!
//! Instruments are resolved from the global provider on every record (an
//! instrument-cache lookup inside the SDK) so recording follows a meter
//! provider installed at any point in the process lifetime; the cost is
//! negligible next to a model call.

use std::future::Future;
use std::time::{Duration, Instant};

use opentelemetry::metrics::Meter;
use opentelemetry::{KeyValue, global};
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt;

use crate::Result;
use switchyard_protocol::{CallRole, GenericKeys, LangfuseKeys, ModelId, Request, Response};

const METRICS_SCOPE: &str = "switchyard";
const TRACING_TARGET: &str = "libsy";

/// The `libsy`-scoped meter from the globally installed provider.
pub(crate) fn meter() -> Meter {
    global::meter(METRICS_SCOPE)
}

/// `outcome` attribute value for a result: `ok` or `error`.
pub(crate) fn outcome_value<T>(result: &Result<T>) -> &'static str {
    match result {
        Ok(_) => "ok",
        Err(_) => "error",
    }
}

/// Span covering one algorithm run (the whole `route` execution).
///
/// Correlation ids from the request [`switchyard_protocol::Metadata`] are recorded as span fields
/// when present. `tracing` spans cannot grow field names at runtime, so
/// arbitrary host labels ride in via [`switchyard_protocol::Metadata::extra_metadata`], recorded
/// whole into the `extra_metadata` field — they are never promoted into their
/// own attribute names. `outcome` and `error` are filled in by [`record_run`]
/// when the run ends. Selected-target attributes are filled by [`record_decision`].
pub(crate) fn run_span(algorithm: &str, request: &Request) -> Span {
    let span = tracing::info_span!(
        target: TRACING_TARGET,
        "libsy.run",
        algorithm,
        switchyard.algorithm = algorithm,
        openinference.span.kind = "CHAIN",
        switchyard.route = tracing::field::Empty,
        session_id = tracing::field::Empty,
        session.id = tracing::field::Empty,
        agent_id = tracing::field::Empty,
        task_id = tracing::field::Empty,
        task_kind = tracing::field::Empty,
        agent_role = tracing::field::Empty,
        correlation_id = tracing::field::Empty,
        extra_metadata = tracing::field::Empty,
        outcome = tracing::field::Empty,
        error = tracing::field::Empty,
        // Shared route-selection vocabulary. Names must match GenericKeys / LangfuseKeys.
        switchyard.route.id = tracing::field::Empty,
        switchyard.routing.algorithm = tracing::field::Empty,
        switchyard.routing.selected_target = tracing::field::Empty,
        langfuse.session.id = tracing::field::Empty,
        langfuse.observation.metadata.switchyard.route_id = tracing::field::Empty,
        langfuse.observation.metadata.switchyard.algorithm = tracing::field::Empty,
        langfuse.observation.metadata.switchyard.selected_target = tracing::field::Empty,
    );
    if let Some(route) = request.model_id() {
        span.record("switchyard.route", route.as_ref());
    }
    if let Some(metadata) = &request.metadata {
        for (field, value) in [
            ("session_id", &metadata.session_id),
            ("agent_id", &metadata.agent_id),
            ("task_id", &metadata.task_id),
            ("task_kind", &metadata.task_kind),
            ("agent_role", &metadata.agent_role),
            ("correlation_id", &metadata.correlation_id),
        ] {
            if let Some(value) = value {
                span.record(field, value.as_str());
            }
        }
        if let Some(session_id) = &metadata.session_id {
            span.record("session.id", session_id.as_str());
        }
        if let Some(extra) = &metadata.extra_metadata {
            span.record("extra_metadata", tracing::field::debug(extra));
        }
    }
    record_route_context(&span, algorithm, request);
    span
}

/// Records route, algorithm, selected target, call role, and call target on an
/// internal `libsy.llm_call` span. Internal calls are always
/// [`CallRole::RoutingDependency`]. Must run before the driver stamps a
/// candidate onto `request.model`.
pub(crate) fn record_internal_call(
    span: &Span,
    algorithm: &str,
    request: &Request,
    call_target: &str,
) {
    record_route_context(span, algorithm, request);
    record_pair(
        span,
        GenericKeys::SELECTED_TARGET,
        LangfuseKeys::SELECTED_TARGET,
        call_target,
    );
    record_pair(
        span,
        GenericKeys::CALL_ROLE,
        LangfuseKeys::CALL_ROLE,
        CallRole::RoutingDependency.as_str(),
    );
    record_pair(
        span,
        GenericKeys::CALL_TARGET,
        LangfuseKeys::CALL_TARGET,
        call_target,
    );
}

/// Records the inbound route, algorithm, and non-empty session onto `span`.
///
/// Empty sessions stay off `langfuse.session.id`. Request content and extra
/// metadata keys are not attributes.
fn record_route_context(span: &Span, algorithm: &str, request: &Request) {
    record_pair(
        span,
        GenericKeys::ALGORITHM,
        LangfuseKeys::ALGORITHM,
        algorithm,
    );
    if let Some(route) = request.model_id() {
        record_pair(
            span,
            GenericKeys::ROUTE_ID,
            LangfuseKeys::ROUTE_ID,
            route.as_ref(),
        );
    }
    let session = request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.session_id.as_deref())
        .filter(|session| !session.is_empty());
    if let Some(session) = session {
        span.record(LangfuseKeys::SESSION_ID, session);
    }
}

/// Records one [`RoutingDecisionKeys`](switchyard_protocol::RoutingDecisionKeys) attribute
/// on the current `libsy.run` span.
///
/// Algorithms call it while routing, outside any `libsy.llm_call` await, so the current
/// span is the run span. It writes the OpenTelemetry attribute directly rather than a
/// `tracing` field: the run span is near the field limit, and the stderr log layer never
/// prints these values. Values are bounded labels, numbers, or booleans, except for the
/// `RoutingDecisionKeys::CONTENT` keys.
pub(crate) fn record_routing(key: &'static str, value: impl Into<opentelemetry::Value>) {
    Span::current().set_attribute(key, value);
}

/// Writes the same value to a generic key and its Langfuse observation metadata key.
fn record_pair(span: &Span, generic_key: &'static str, langfuse_key: &'static str, value: &str) {
    span.record(generic_key, value);
    span.record(langfuse_key, value);
}

/// Runs one algorithm task to completion, recording the run counter, duration
/// histogram, span outcome, and failure log when it resolves.
/// Executes inside the `libsy.run` span its caller instruments the task with.
pub(crate) async fn observe_run<T>(
    algorithm: &str,
    run: impl Future<Output = Result<T>>,
) -> Result<T> {
    let started = Instant::now();
    let result = run.await;
    let duration = started.elapsed();
    record_run(algorithm, duration, &result, &Span::current());
    result
}

/// Records the end of one algorithm run: the run counter and duration
/// histogram, the `outcome`/`error` fields on `span`, and a warn log when the
/// run failed.
fn record_run<T>(algorithm: &str, duration: Duration, result: &Result<T>, span: &Span) {
    let outcome = outcome_value(result);
    span.record("outcome", outcome);
    if let Err(error) = result {
        span.record("error", tracing::field::display(error));
        tracing::warn!(
            target: TRACING_TARGET,
            algorithm,
            error = %error,
            "algorithm run failed"
        );
    }

    let attributes = [
        KeyValue::new("algorithm", algorithm.to_string()),
        KeyValue::new("outcome", outcome),
    ];
    let meter = meter();
    meter
        .u64_counter("switchyard.runs")
        .build()
        .add(1, &attributes);
    meter
        .f64_histogram("switchyard.run_duration_ms")
        .build()
        .record(duration.as_secs_f64() * 1000.0, &attributes);
}

/// Records a judge failure that made the classifier route without a verdict.
pub(crate) fn record_classifier_fail_open(judge_model: &str, reason: &'static str) {
    meter()
        .u64_counter("switchyard.classifier_fail_open")
        .build()
        .add(
            1,
            &[
                KeyValue::new("judge_model", judge_model.to_string()),
                KeyValue::new("reason", reason),
            ],
        );
}

/// Records the resolution of one offloaded model call: the call counter and
/// latency histogram, the `outcome`/`error`/token fields on `span`, and a warn
/// log when the call failed.
pub(crate) fn record_llm_call(
    algorithm: &str,
    selected_model: &str,
    duration: Duration,
    result: &Result<Response>,
    span: &Span,
) {
    let outcome = outcome_value(result);
    span.record("outcome", outcome);

    let meter = meter();
    let call_attributes = [
        KeyValue::new("algorithm", algorithm.to_string()),
        KeyValue::new("selected_model", selected_model.to_string()),
        KeyValue::new("outcome", outcome),
    ];
    meter
        .u64_counter("switchyard.llm_calls")
        .build()
        .add(1, &call_attributes);
    meter
        .f64_histogram("switchyard.llm_call_duration_ms")
        .build()
        .record(duration.as_secs_f64() * 1000.0, &call_attributes);

    match result {
        Ok(response) => {
            // Token usage exists only once a response is buffered; a streamed
            // response resolves before its usage is known, so none is recorded.
            let Some(usage) = response.llm_response.as_agg().map(|agg| &agg.usage) else {
                return;
            };
            for (field, value) in [
                ("input_tokens", usage.input_tokens),
                ("output_tokens", usage.output_tokens),
                ("total_tokens", usage.total_tokens),
                ("reasoning_tokens", usage.reasoning_tokens),
            ] {
                if let Some(value) = value {
                    span.record(field, value);
                }
            }
        }
        Err(error) => {
            span.record("error", tracing::field::display(error));
            tracing::warn!(
                target: TRACING_TARGET,
                algorithm,
                selected_model,
                error = %error,
                "model call failed"
            );
        }
    }
}

/// Records one published routing decision: the decision counter plus a structured debug event.
///
/// Also writes the selected target onto the current `libsy.run` span. Metric
/// labels stay `algorithm` and `selected_model` — the shared vocabulary is
/// span-only.
pub(crate) fn record_decision(algorithm: &str, selected_model: &ModelId) {
    record_pair(
        &Span::current(),
        GenericKeys::SELECTED_TARGET,
        LangfuseKeys::SELECTED_TARGET,
        selected_model.as_str(),
    );
    tracing::debug!(
        target: TRACING_TARGET,
        algorithm,
        selected_model = %selected_model,
        "routing decision"
    );
    meter().u64_counter("switchyard.decisions").build().add(
        1,
        &[
            KeyValue::new("algorithm", algorithm.to_string()),
            KeyValue::new("selected_model", selected_model.to_string()),
        ],
    );
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fmt;
    use std::sync::Arc;

    use async_trait::async_trait;
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider, SpanData};
    use parking_lot::Mutex;
    use switchyard_protocol::{LlmClientError, RoutingDecisionKeys as Keys};
    use tracing::Subscriber;
    use tracing::field::{Field, Visit};
    use tracing::span::{Attributes, Id, Record};
    use tracing_subscriber::layer::{Context as LayerContext, SubscriberExt};
    use tracing_subscriber::registry::LookupSpan;

    use super::*;
    use crate::core::algorithm::{Algorithm, Driver, RoutingOutcome};
    use crate::core::testing::{Serve, echo, reply, test_drive};
    use crate::{LibsyError, Result};
    use switchyard_protocol::{Metadata, Request, completion_text, text_request};
    use tracing_subscriber::Layer;

    // tracing's subscriber is process-global; install a capture layer once so
    // the global subscriber is never changed mid-run. Tests use unique
    // algorithm names and filter spans by algorithm, so a shared store does not
    // mix their results.
    static CAPTURE_STORE: std::sync::LazyLock<CaptureStore> =
        std::sync::LazyLock::new(CaptureStore::default);
    // Routing-decision attributes are written straight to OpenTelemetry, so the shared
    // subscriber also exports spans to memory for the tests that assert on them.
    static OTEL_SPANS: std::sync::LazyLock<InMemorySpanExporter> =
        std::sync::LazyLock::new(InMemorySpanExporter::default);
    static _SUBSCRIBER: std::sync::LazyLock<()> = std::sync::LazyLock::new(|| {
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(OTEL_SPANS.clone())
            .build();
        let subscriber = tracing_subscriber::registry()
            .with(CaptureLayer {
                store: CAPTURE_STORE.clone(),
            })
            .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("libsy-test")));
        // set_global_default is process-global and cannot be undone, so it
        // must be called exactly once.
        tracing::subscriber::set_global_default(subscriber)
            .expect("failed to set global default subscriber");
    });

    const PROMPT: &str = "SECRET_PROMPT_DO_NOT_PROMOTE";
    const EXTRA_KEY: &str = "secret_key";
    const EXTRA_VALUE: &str = "secret-value";

    #[derive(Clone, Debug, Default)]
    struct SpanRecord {
        name: String,
        fields: BTreeMap<String, String>,
    }

    #[derive(Clone, Default)]
    struct CaptureStore {
        spans: Arc<Mutex<BTreeMap<u64, SpanRecord>>>,
    }

    impl CaptureStore {
        fn spans(&self) -> Vec<SpanRecord> {
            self.spans.lock().values().cloned().collect()
        }
    }

    struct FieldVisitor<'a>(&'a mut BTreeMap<String, String>);

    impl Visit for FieldVisitor<'_> {
        fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
            self.0
                .insert(field.name().to_string(), format!("{value:?}"));
        }

        fn record_str(&mut self, field: &Field, value: &str) {
            self.0.insert(field.name().to_string(), value.to_string());
        }
    }

    struct CaptureLayer {
        store: CaptureStore,
    }

    impl<S> Layer<S> for CaptureLayer
    where
        S: Subscriber + for<'a> LookupSpan<'a>,
    {
        fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, _ctx: LayerContext<'_, S>) {
            let mut fields = BTreeMap::new();
            attrs.record(&mut FieldVisitor(&mut fields));
            self.store.spans.lock().insert(
                id.into_u64(),
                SpanRecord {
                    name: attrs.metadata().name().to_string(),
                    fields,
                },
            );
        }

        fn on_record(&self, id: &Id, values: &Record<'_>, _ctx: LayerContext<'_, S>) {
            if let Some(record) = self.store.spans.lock().get_mut(&id.into_u64()) {
                values.record(&mut FieldVisitor(&mut record.fields));
            }
        }
    }

    fn find_span<'a>(spans: &'a [SpanRecord], name: &str, algorithm: &str) -> &'a SpanRecord {
        spans
            .iter()
            .find(|span| {
                span.name == name
                    && span.fields.get("algorithm").map(String::as_str) == Some(algorithm)
            })
            .unwrap_or_else(|| panic!("missing {name} span for {algorithm} in {spans:?}"))
    }

    fn assert_pair(span: &SpanRecord, generic: &str, langfuse: &str, value: &str) {
        assert_eq!(
            span.fields.get(generic).map(String::as_str),
            Some(value),
            "{generic}"
        );
        assert_eq!(
            span.fields.get(langfuse).map(String::as_str),
            Some(value),
            "{langfuse}"
        );
    }

    fn assert_content_not_promoted(span: &SpanRecord) {
        assert!(
            !span.fields.contains_key(EXTRA_KEY),
            "extra metadata key promoted on {}",
            span.name
        );
        assert!(
            !span.fields.contains_key("gen_ai.prompt"),
            "prompt promoted on {}",
            span.name
        );
        assert!(
            !span.fields.contains_key("gen_ai.completion"),
            "completion promoted on {}",
            span.name
        );
        for (name, value) in &span.fields {
            if name != "extra_metadata" {
                assert!(
                    !value.contains(PROMPT),
                    "{name} on {} leaked request content",
                    span.name
                );
            }
        }
    }

    fn request_with(session: Option<&str>, extra: Option<BTreeMap<String, String>>) -> Request {
        Request {
            llm_request: text_request(Some("auto".to_string()), PROMPT),
            raw_request: None,
            metadata: Some(Metadata {
                session_id: session.map(str::to_string),
                extra_metadata: extra,
                ..Metadata::default()
            }),
        }
    }

    async fn capture_drive(
        algorithm: Arc<dyn Algorithm>,
        request: Request,
    ) -> Result<Vec<SpanRecord>> {
        // Use the one shared subscriber; spans are filtered by algorithm, so
        // concurrent tests don't mix results.
        let _ = &*_SUBSCRIBER;
        test_drive(algorithm, request, echo()).await?;
        Ok(CAPTURE_STORE.spans())
    }

    /// Calls one model during routing and returns that model as the answer.
    struct AnswerAlgo {
        name: &'static str,
        target: ModelId,
    }

    #[async_trait]
    impl Algorithm for AnswerAlgo {
        fn name(&self) -> &str {
            self.name
        }

        async fn route(
            self: Arc<Self>,
            driver: Driver,
            request: Request,
        ) -> Result<RoutingOutcome> {
            let response = driver
                .call_model(request.clone(), vec![self.target.clone()])
                .await?;
            Ok(RoutingOutcome::answered(
                self.target.clone(),
                request,
                response,
            ))
        }
    }

    /// Calls a judge, then publishes a selected target plus a distinct fallback.
    struct JudgeThenFallback {
        name: &'static str,
        judge: ModelId,
        selected: ModelId,
        fallback: ModelId,
    }

    #[async_trait]
    impl Algorithm for JudgeThenFallback {
        fn name(&self) -> &str {
            self.name
        }

        async fn route(
            self: Arc<Self>,
            driver: Driver,
            request: Request,
        ) -> Result<RoutingOutcome> {
            let _ = driver
                .call_model(request.clone(), vec![self.judge.clone()])
                .await?;
            Ok(RoutingOutcome::route_to(
                self.selected.clone(),
                vec![self.fallback.clone()],
                request,
            ))
        }
    }

    /// Normal success: both spans carry the shared vocabulary, existing fields
    /// remain, and neither extra metadata keys nor request content become attributes.
    #[tokio::test(flavor = "current_thread")]
    async fn successful_run_emits_shared_vocabulary_without_promoting_content() -> Result<()> {
        const ALGO: &str = "vocab-success";
        const TARGET: &str = "vocab-success-target";
        let extra = BTreeMap::from([(EXTRA_KEY.to_string(), EXTRA_VALUE.to_string())]);
        let algorithm: Arc<dyn Algorithm> = Arc::new(AnswerAlgo {
            name: ALGO,
            target: TARGET.into(),
        });
        let spans = capture_drive(algorithm, request_with(Some("session-1"), Some(extra))).await?;

        let run = find_span(&spans, "libsy.run", ALGO);
        assert_eq!(run.fields.get("algorithm").map(String::as_str), Some(ALGO));
        assert_eq!(
            run.fields.get("switchyard.algorithm").map(String::as_str),
            Some(ALGO)
        );
        assert_eq!(
            run.fields.get("switchyard.route").map(String::as_str),
            Some("auto")
        );
        assert_eq!(
            run.fields.get("session_id").map(String::as_str),
            Some("session-1")
        );
        assert_eq!(
            run.fields.get("session.id").map(String::as_str),
            Some("session-1")
        );
        assert_eq!(run.fields.get("outcome").map(String::as_str), Some("ok"));
        assert_pair(run, GenericKeys::ROUTE_ID, LangfuseKeys::ROUTE_ID, "auto");
        assert_pair(run, GenericKeys::ALGORITHM, LangfuseKeys::ALGORITHM, ALGO);
        assert_pair(
            run,
            GenericKeys::SELECTED_TARGET,
            LangfuseKeys::SELECTED_TARGET,
            TARGET,
        );
        assert_eq!(
            run.fields.get(LangfuseKeys::SESSION_ID).map(String::as_str),
            Some("session-1")
        );
        assert!(
            run.fields
                .get("extra_metadata")
                .is_some_and(|extra| extra.contains(EXTRA_KEY) && extra.contains(EXTRA_VALUE))
        );
        assert_content_not_promoted(run);

        let call = find_span(&spans, "libsy.llm_call", ALGO);
        assert_eq!(
            call.fields.get("selected_model").map(String::as_str),
            Some(TARGET)
        );
        assert_eq!(call.fields.get("outcome").map(String::as_str), Some("ok"));
        assert_pair(call, GenericKeys::ROUTE_ID, LangfuseKeys::ROUTE_ID, "auto");
        assert_pair(call, GenericKeys::ALGORITHM, LangfuseKeys::ALGORITHM, ALGO);
        assert_pair(
            call,
            GenericKeys::SELECTED_TARGET,
            LangfuseKeys::SELECTED_TARGET,
            TARGET,
        );
        assert_pair(
            call,
            GenericKeys::CALL_ROLE,
            LangfuseKeys::CALL_ROLE,
            CallRole::RoutingDependency.as_str(),
        );
        assert_pair(
            call,
            GenericKeys::CALL_TARGET,
            LangfuseKeys::CALL_TARGET,
            TARGET,
        );
        assert_eq!(
            call.fields
                .get(LangfuseKeys::SESSION_ID)
                .map(String::as_str),
            Some("session-1")
        );
        assert_content_not_promoted(call);
        Ok(())
    }

    /// Fallback success: the run span keeps the selected target after the host
    /// serves a different fallback, while the internal call is tagged as a
    /// routing dependency against the judge.
    #[tokio::test(flavor = "current_thread")]
    async fn fallback_success_keeps_selected_target_distinct_from_served() -> Result<()> {
        // Use the one shared subscriber; spans are filtered by algorithm, so
        // concurrent tests don't mix results.
        let _ = &*_SUBSCRIBER;
        const ALGO: &str = "vocab-fallback";
        const JUDGE: &str = "vocab-judge";
        const SELECTED: &str = "vocab-selected";
        const FALLBACK: &str = "vocab-fallback-target";
        let algorithm: Arc<dyn Algorithm> = Arc::new(JudgeThenFallback {
            name: ALGO,
            judge: JUDGE.into(),
            selected: SELECTED.into(),
            fallback: FALLBACK.into(),
        });
        let outcome = crate::drive(
            algorithm,
            request_with(Some("session-2"), None),
            |call| async {
                let target = call.models.first().cloned().ok_or(LibsyError::NoTargets)?;
                call.respond(Ok(reply(target)))?;
                Ok(())
            },
        )
        .await?;
        let served_target = outcome
            .fallback_models
            .first()
            .cloned()
            .expect("missing fallback");
        let served = echo()
            .serve(served_target.clone(), outcome.request)
            .await
            .map_err(|source| LibsyError::client_call(served_target.clone(), source))?;

        assert_eq!(outcome.selected_model_id.as_str(), SELECTED);
        assert_eq!(served_target.as_str(), FALLBACK);
        assert_eq!(
            served
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            FALLBACK
        );

        let spans = CAPTURE_STORE.spans();
        let run = find_span(&spans, "libsy.run", ALGO);
        assert_pair(
            run,
            GenericKeys::SELECTED_TARGET,
            LangfuseKeys::SELECTED_TARGET,
            SELECTED,
        );
        assert_ne!(
            run.fields
                .get(GenericKeys::SELECTED_TARGET)
                .map(String::as_str),
            Some(FALLBACK)
        );

        let call = find_span(&spans, "libsy.llm_call", ALGO);
        assert_pair(
            call,
            GenericKeys::CALL_ROLE,
            LangfuseKeys::CALL_ROLE,
            CallRole::RoutingDependency.as_str(),
        );
        assert_pair(
            call,
            GenericKeys::CALL_TARGET,
            LangfuseKeys::CALL_TARGET,
            JUDGE,
        );
        Ok(())
    }

    /// An empty session is kept off langfuse.session.id; existing session fields
    /// still record the empty string.
    #[tokio::test(flavor = "current_thread")]
    async fn empty_session_is_not_promoted() -> Result<()> {
        const ALGO: &str = "vocab-empty-session";
        const TARGET: &str = "vocab-empty-target";
        let algorithm: Arc<dyn Algorithm> = Arc::new(AnswerAlgo {
            name: ALGO,
            target: TARGET.into(),
        });
        let spans = capture_drive(algorithm, request_with(Some(""), None)).await?;

        let run = find_span(&spans, "libsy.run", ALGO);
        assert_eq!(run.fields.get("session_id").map(String::as_str), Some(""));
        assert_eq!(run.fields.get(LangfuseKeys::SESSION_ID), None);

        let call = find_span(&spans, "libsy.llm_call", ALGO);
        assert_eq!(call.fields.get(LangfuseKeys::SESSION_ID), None);
        Ok(())
    }

    const CRUX: &str = "SECRET_CRUX_FROM_JUDGE";

    /// Answers the `judge` target with `verdict`, or fails it when `verdict` is `None`, and
    /// echoes every other target.
    fn judge_replies(verdict: Option<&'static str>) -> impl Serve {
        move |target: ModelId, _request: Request| async move {
            if target != "judge" {
                return Ok(reply(target));
            }
            match verdict {
                Some(verdict) => Ok(reply(verdict)),
                None => Err(LlmClientError::Timeout {
                    source: Box::new(std::io::Error::other("judge unreachable")),
                }),
            }
        }
    }

    /// Routes one request on `route` and returns its exported `libsy.run` attributes, plus
    /// every `tracing` field recorded on any span of the run (what a log layer could print).
    async fn routed_run(
        algorithm: Arc<dyn Algorithm>,
        route: &str,
        serve: impl Serve,
    ) -> Result<(BTreeMap<String, opentelemetry::Value>, Vec<SpanRecord>)> {
        let _ = &*_SUBSCRIBER;
        let request = Request {
            llm_request: text_request(Some(route.to_string()), PROMPT),
            raw_request: None,
            metadata: None,
        };
        test_drive(algorithm, request, serve).await?;
        let is_run = |span: &SpanData| {
            span.name == "libsy.run"
                && span.attributes.iter().any(|kv| {
                    kv.key.as_str() == GenericKeys::ROUTE_ID && kv.value.as_str() == route
                })
        };
        // The run span closes when the spawned run task ends, just after the outcome is sent.
        for _ in 0..200 {
            let spans = OTEL_SPANS.get_finished_spans().unwrap_or_default();
            if let Some(run) = spans.iter().find(|span| is_run(span)) {
                let attributes = run
                    .attributes
                    .iter()
                    .map(|kv| (kv.key.to_string(), kv.value.clone()))
                    .collect();
                let fields = CAPTURE_STORE
                    .spans()
                    .into_iter()
                    .filter(|span| {
                        span.fields.get("switchyard.route").map(String::as_str) == Some(route)
                            || span.fields.get(GenericKeys::ROUTE_ID).map(String::as_str)
                                == Some(route)
                    })
                    .collect();
                return Ok((attributes, fields));
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("libsy.run span for {route} was never exported");
    }

    fn text(attributes: &BTreeMap<String, opentelemetry::Value>, key: &str) -> Option<String> {
        attributes.get(key).map(|value| value.as_str().into_owned())
    }

    /// Fails when `secret` appears in any routing attribute outside the content keys, or in
    /// any `tracing` field a log layer could print.
    fn assert_only_in_content(
        attributes: &BTreeMap<String, opentelemetry::Value>,
        fields: &[SpanRecord],
        secret: &str,
    ) {
        for (key, value) in attributes {
            if !Keys::CONTENT.contains(&key.as_str()) {
                assert!(!value.as_str().contains(secret), "{key} carries free text");
            }
        }
        for span in fields {
            for (name, value) in &span.fields {
                assert!(
                    !value.contains(secret),
                    "{name} on {} carries free text",
                    span.name
                );
            }
        }
    }

    fn capability_router() -> Result<Arc<dyn Algorithm>> {
        use crate::algorithms::llm_class::{
            LlmClassifierConfig, LlmTaskClassifier, TaskClassifierConfig,
        };
        Ok(Arc::new(LlmTaskClassifier::new(
            LlmClassifierConfig::Capability {
                judge_target: "judge".into(),
                efficient_target: "efficient".into(),
                capable_target: "capable".into(),
                config: TaskClassifierConfig {
                    base_threshold: 0.6,
                    threshold_step: 0.1,
                    ..TaskClassifierConfig::default()
                },
            },
        )?))
    }

    /// A valid capability verdict explains the pick: p_solve below the boundary-adjusted
    /// threshold sends the turn to the strong tier. The crux stays in its content key.
    #[tokio::test(flavor = "current_thread")]
    async fn capability_verdict_records_score_threshold_and_rule() -> Result<()> {
        let verdict = r#"{"crux":"SECRET_CRUX_FROM_JUDGE","primary_rule":"UNC-1","capability_boundary":"uncertain","p_solve":0.62}"#;
        let (run, fields) = routed_run(
            capability_router()?,
            "route-capability-ok",
            judge_replies(Some(verdict)),
        )
        .await?;

        assert_eq!(text(&run, Keys::MODE).as_deref(), Some("capability"));
        assert_eq!(text(&run, Keys::DECIDED_BY).as_deref(), Some("judge"));
        assert_eq!(text(&run, Keys::TIER).as_deref(), Some("strong"));
        assert_eq!(text(&run, Keys::JUDGE_MODEL).as_deref(), Some("judge"));
        assert_eq!(
            text(&run, Keys::JUDGE_VERDICT_STATUS).as_deref(),
            Some("ok")
        );
        assert_eq!(
            text(&run, Keys::JUDGE_PRIMARY_RULE).as_deref(),
            Some("UNC-1")
        );
        assert_eq!(
            text(&run, Keys::JUDGE_CAPABILITY_BOUNDARY).as_deref(),
            Some("uncertain")
        );
        assert_eq!(
            run.get(Keys::JUDGE_P_SOLVE),
            Some(&opentelemetry::Value::F64(0.62))
        );
        let Some(opentelemetry::Value::F64(threshold)) = run.get(Keys::JUDGE_THRESHOLD) else {
            panic!("threshold missing: {run:?}");
        };
        assert!((threshold - 0.7).abs() < 1e-9);
        assert_eq!(text(&run, Keys::JUDGE_CRUX).as_deref(), Some(CRUX));
        assert_eq!(run.get(Keys::FALLBACK_REASON), None);
        assert_only_in_content(&run, &fields, CRUX);
        Ok(())
    }

    /// No verdict, or one whose labels are not the enumerated ones, is labelled as a
    /// fallback, and the judge's free text is never copied into a routing attribute.
    #[tokio::test(flavor = "current_thread")]
    async fn missing_or_invalid_verdicts_are_labelled_fallbacks() -> Result<()> {
        let (run, _) = routed_run(
            capability_router()?,
            "route-capability-unavailable",
            judge_replies(None),
        )
        .await?;
        assert_eq!(text(&run, Keys::DECIDED_BY).as_deref(), Some("fallback"));
        assert_eq!(
            text(&run, Keys::FALLBACK_REASON).as_deref(),
            Some("verdict_unavailable")
        );
        assert_eq!(
            text(&run, Keys::JUDGE_VERDICT_STATUS).as_deref(),
            Some("unavailable")
        );
        assert_eq!(text(&run, Keys::TIER).as_deref(), Some("strong"));

        let invalid = r#"{"crux":"SECRET_CRUX_FROM_JUDGE","primary_rule":"SECRET_CRUX_FROM_JUDGE","capability_boundary":"supported","p_solve":0.9}"#;
        let (run, fields) = routed_run(
            capability_router()?,
            "route-capability-invalid",
            judge_replies(Some(invalid)),
        )
        .await?;
        assert_eq!(text(&run, Keys::DECIDED_BY).as_deref(), Some("fallback"));
        assert_eq!(
            text(&run, Keys::FALLBACK_REASON).as_deref(),
            Some("verdict_invalid")
        );
        assert_eq!(
            text(&run, Keys::JUDGE_VERDICT_STATUS).as_deref(),
            Some("invalid")
        );
        for key in [
            Keys::JUDGE_PRIMARY_RULE,
            Keys::JUDGE_CAPABILITY_BOUNDARY,
            Keys::JUDGE_P_SOLVE,
            Keys::JUDGE_CRUX,
        ] {
            assert_eq!(run.get(key), None, "{key}");
        }
        assert_only_in_content(&run, &fields, CRUX);
        Ok(())
    }

    /// A confirmed escalation records the verdict, streak, and latch; the judge's reason
    /// stays in its content key.
    #[tokio::test(flavor = "current_thread")]
    async fn escalation_records_streak_and_latch() -> Result<()> {
        use crate::algorithms::llm_class::{LlmClassifierConfig, LlmTaskClassifier};
        use crate::algorithms::util::DEFAULT_JUDGE_MAX_OUTPUT_TOKENS;
        use crate::algorithms::util::classifier_contract::ClassifierContractConfig;
        use crate::algorithms::util::escalation::EscalationJudgeConfig;

        let router = Arc::new(LlmTaskClassifier::new(LlmClassifierConfig::Escalation {
            judge_target: "judge".into(),
            efficient_target: "efficient".into(),
            capable_target: "capable".into(),
            contract: ClassifierContractConfig::default(),
            config: EscalationJudgeConfig {
                confirmations: 1,
                ..EscalationJudgeConfig::default()
            },
            max_output_tokens: DEFAULT_JUDGE_MAX_OUTPUT_TOKENS,
        })?);
        let verdict = r#"{"escalate":true,"reason":"SECRET_CRUX_FROM_JUDGE"}"#;
        let (run, fields) =
            routed_run(router, "route-escalation", judge_replies(Some(verdict))).await?;

        assert_eq!(text(&run, Keys::MODE).as_deref(), Some("escalation"));
        assert_eq!(text(&run, Keys::DECIDED_BY).as_deref(), Some("judge"));
        assert_eq!(text(&run, Keys::TIER).as_deref(), Some("strong"));
        assert_eq!(
            run.get(Keys::ESCALATION_ESCALATE),
            Some(&opentelemetry::Value::Bool(true))
        );
        assert_eq!(
            run.get(Keys::ESCALATION_STREAK),
            Some(&opentelemetry::Value::I64(1))
        );
        assert_eq!(
            run.get(Keys::ESCALATION_CONFIRMATIONS),
            Some(&opentelemetry::Value::I64(1))
        );
        assert_eq!(
            run.get(Keys::ESCALATION_LATCHED),
            Some(&opentelemetry::Value::Bool(true))
        );
        assert_eq!(text(&run, Keys::ESCALATION_REASON).as_deref(), Some(CRUX));
        assert_only_in_content(&run, &fields, CRUX);
        Ok(())
    }

    /// A custom verdict records only a configured label; an unknown label is a fallback.
    #[tokio::test(flavor = "current_thread")]
    async fn custom_label_is_recorded_only_when_configured() -> Result<()> {
        use crate::algorithms::llm_class::{
            CustomClassifierConfig, CustomClassifierPolicy, LlmClassifierConfig, LlmTaskClassifier,
        };
        let router = || -> Result<Arc<dyn Algorithm>> {
            Ok(Arc::new(LlmTaskClassifier::new(
                LlmClassifierConfig::Custom {
                    judge_target: "judge".into(),
                    targets: vec![
                        ("fast".to_string(), "efficient".into()),
                        ("deep".to_string(), "capable".into()),
                    ],
                    default_target: "deep".to_string(),
                    config: CustomClassifierConfig::new(
                        "Pick a label.",
                        serde_json::json!({
                            "type": "object",
                            "properties": {"label": {"type": "string"}},
                            "required": ["label"],
                            "additionalProperties": false
                        }),
                        CustomClassifierPolicy::target_selector("/label"),
                    ),
                },
            )?))
        };

        let (run, _) = routed_run(
            router()?,
            "route-custom-known",
            judge_replies(Some(r#"{"label":"fast"}"#)),
        )
        .await?;
        assert_eq!(text(&run, Keys::MODE).as_deref(), Some("custom"));
        assert_eq!(text(&run, Keys::CUSTOM_LABEL).as_deref(), Some("fast"));
        assert_eq!(text(&run, Keys::DECIDED_BY).as_deref(), Some("judge"));

        let (run, fields) = routed_run(
            router()?,
            "route-custom-unknown",
            judge_replies(Some(r#"{"label":"SECRET_CRUX_FROM_JUDGE"}"#)),
        )
        .await?;
        assert_eq!(run.get(Keys::CUSTOM_LABEL), None);
        assert_eq!(
            text(&run, Keys::FALLBACK_REASON).as_deref(),
            Some("label_unknown")
        );
        assert_eq!(text(&run, Keys::DECIDED_BY).as_deref(), Some("fallback"));
        assert_only_in_content(&run, &fields, CRUX);
        Ok(())
    }
}
