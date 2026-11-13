// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures_util::StreamExt;
use switchyard_llm_client::{ClientRouter, RunObservation};
use switchyard_protocol::{
    ContentBlock, LlmClientError, LlmResponse, MediaSource, ModelId, Request, Response,
    RoutedLlmClient, text_request, text_response,
};
use switchyard_runner::{AlgorithmSpec, ModelCapabilities, Route, Runner, RunnerError};

struct StubClient;

#[async_trait]
impl RoutedLlmClient for StubClient {
    async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
        Ok(Response {
            llm_response: LlmResponse::Agg(text_response(
                request.llm_request.model.clone(),
                "plugin response",
            )),
            metadata: None,
        })
    }
}

fn plugin_route(client: Arc<dyn RoutedLlmClient>) -> Route {
    let spec = AlgorithmSpec::Passthrough {
        target: "semantic-target".to_string(),
        subagents: None,
    };
    let targets = BTreeMap::from([(
        "semantic-target".to_string(),
        ModelId::from("semantic-target"),
    )]);
    let algorithm = spec
        .build("switchyard", &targets)
        .expect("identity target map should build");
    let clients = ClientRouter::new(
        BTreeMap::from([(ModelId::from("semantic-target"), client)])
            .into_iter()
            .collect(),
    );
    Route::new(
        algorithm,
        clients,
        None,
        ModelCapabilities::default(),
        None,
        Vec::new(),
    )
}

#[tokio::test]
async fn plugin_shaped_route_executes_without_runner_model_or_toml() {
    let route = plugin_route(Arc::new(StubClient));
    let observations = Arc::new(Mutex::new(Vec::new()));
    let observer = {
        let observations = Arc::clone(&observations);
        Arc::new(move |observation| observations.lock().unwrap().push(observation))
    };
    let request = Request {
        llm_request: text_request(Some("arbitrary-upstream-model".to_string()), "hello"),
        ..Request::default()
    };

    let output = route
        .execute(request, Some(observer))
        .await
        .expect("route should execute");

    assert_eq!(output.selected_model, "semantic-target");
    assert_eq!(
        output
            .response
            .llm_response
            .as_agg()
            .unwrap()
            .model
            .as_deref(),
        Some("semantic-target")
    );
    assert!(
        observations
            .lock()
            .unwrap()
            .iter()
            .any(|observation| { matches!(observation, RunObservation::AnswerCall(_)) })
    );
    assert!(
        observations
            .lock()
            .unwrap()
            .iter()
            .any(|observation| { matches!(observation, RunObservation::RoutingOverhead(_)) })
    );
}

struct LazyStreamClient {
    polls: Arc<AtomicUsize>,
}

#[async_trait]
impl RoutedLlmClient for LazyStreamClient {
    async fn call(&self, _request: Request) -> Result<Response, LlmClientError> {
        let polls = Arc::clone(&self.polls);
        let stream = futures_util::stream::poll_fn(move |_| {
            polls.fetch_add(1, Ordering::SeqCst);
            std::task::Poll::Ready(None)
        })
        .boxed();
        Ok(Response {
            llm_response: LlmResponse::Stream(stream),
            metadata: None,
        })
    }
}

#[tokio::test]
async fn route_returns_stream_without_polling_it() {
    let polls = Arc::new(AtomicUsize::new(0));
    let route = plugin_route(Arc::new(LazyStreamClient {
        polls: Arc::clone(&polls),
    }));
    let request = Request {
        llm_request: text_request(None, "hello"),
        ..Request::default()
    };

    let output = route
        .execute(request, None)
        .await
        .expect("stream handle should be returned");

    assert!(matches!(
        output.response.llm_response,
        LlmResponse::Stream(_)
    ));
    assert_eq!(polls.load(Ordering::SeqCst), 0);
}

#[test]
fn algorithm_build_reports_unknown_configured_target() {
    let spec = AlgorithmSpec::Random {
        targets: vec!["missing".to_string()],
        weights: None,
        seed: None,
    };

    let error = match spec.build("plugin", &BTreeMap::new()) {
        Ok(_) => panic!("unknown target should fail"),
        Err(error) => error,
    };

    assert_eq!(
        error.to_string(),
        "route plugin references unknown target missing"
    );
}

// Two gateway targets that differ only in reasoning, as Roundhouse stamps them.
const CAPABILITY_ROUTES: &str = r#"
schema_version = 1

[llm_clients.gateway]
format = "openai_chat"
base_url = "http://gateway.invalid/v1"

[targets.nova]
id = "nova-lite"
llm_client = "gateway"
capabilities = ["text", "image", "tools", "caching", "streaming"]

[targets.haiku]
id = "haiku"
llm_client = "gateway"
capabilities = ["text", "image", "tools", "reasoning", "caching", "streaming"]

[targets.judge]
id = "judge"
llm_client = "gateway"

[routes.fast]
id = "rp-fast"
type = "random"
targets = ["nova", "haiku"]

[routes.cheap]
id = "rp-cheap"
type = "passthrough"
target = "nova"

[routes.auto]
id = "rp-auto"
type = "llm_classifier"
classifier_target = "judge"
strong_target = "haiku"
weak_target = "nova"
base_threshold = 0.5
"#;

fn reasoning_request() -> Request {
    let mut llm_request = text_request(Some("rp-fast".to_string()), "think hard");
    llm_request.reasoning.effort = Some("high".to_string());
    Request {
        llm_request,
        ..Request::default()
    }
}

#[tokio::test]
async fn routed_targets_are_filtered_by_request_features() {
    let runner = Runner::from_toml(CAPABILITY_ROUTES).expect("capability routes should load");

    // Random never picks the target that lacks reasoning.
    let fast = runner.route("rp-fast").expect("rp-fast is configured");
    for _ in 0..20 {
        let outcome = fast
            .decide(reasoning_request())
            .await
            .expect("haiku supports reasoning");
        assert_eq!(outcome.selected_model_id, "haiku");
    }

    // The classifier skips its judge when only one tier fits; no judge call is made.
    let auto = runner.route("rp-auto").expect("rp-auto is configured");
    let outcome = auto
        .decide(reasoning_request())
        .await
        .expect("the strong tier supports reasoning");
    assert_eq!(outcome.selected_model_id, "haiku");
}

#[tokio::test]
async fn request_no_target_supports_is_an_invalid_request() {
    let runner = Runner::from_toml(CAPABILITY_ROUTES).expect("capability routes should load");
    let cheap = runner.route("rp-cheap").expect("rp-cheap is configured");

    match cheap.execute(reasoning_request(), None).await {
        Err(RunnerError::Client(LlmClientError::InvalidRequest { message, .. })) => {
            assert!(message.contains("rp-cheap"), "{message}");
            assert!(message.contains("reasoning"), "{message}");
        }
        Err(error) => panic!("expected an invalid request, got {error}"),
        Ok(_) => panic!("nova-lite does not support reasoning"),
    }

    let mut video = text_request(Some("rp-fast".to_string()), "watch");
    video.messages[0].content.push(ContentBlock::Video {
        source: MediaSource::Url {
            url: "https://example.test/clip.mp4".to_string(),
            media_type: None,
        },
    });
    let fast = runner.route("rp-fast").expect("rp-fast is configured");
    let refused = fast
        .decide(Request {
            llm_request: video,
            ..Request::default()
        })
        .await;
    assert!(matches!(
        refused,
        Err(RunnerError::Client(LlmClientError::InvalidRequest { .. }))
    ));
}
