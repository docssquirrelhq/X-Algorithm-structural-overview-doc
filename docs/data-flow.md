---
title: Data Flow and Integration
slug: data-flow
description: End-to-end data flow in the recommendation pipeline, from Rust query processing to Python model inference, including caching, bloom filters, and deduplication.
sidebar_label: Data Flow
sidebar_position: 12
---

# Data Flow and Integration in the Recommendation Pipeline

This page describes the end-to-end data flow and integration between components in the recommendation system. It starts from the user query in Rust (`ScoredPostsQuery`) and traces through the pipeline to Python model inference (e.g., via `PhoenixSource` for retrieval and `PhoenixScorer` for ranking). Key aspects covered include side effects like `CacheRequestInfoSideEffect`, bloom filters for efficient deduplication, and handling of `seen_ids`/`served_ids` to avoid repetition. The pipeline is orchestrated by the `PhoenixCandidatePipeline` in Rust, blending efficient Rust orchestration with Python-based ML for scalable recommendations.

For an overview of the pipeline structure, see [Candidate Pipeline](../candidate-pipeline.md). Related topics include [Rust Home Mixer Overview](../rust-home-mixer-overview.md), [Python Phoenix Overview](../python-phoenix-overview.md), [Sources](../sources.md), [Scorers](../scorers.md), and [Filters](../filters.md).

## High-Level Pipeline Overview

The recommendation pipeline processes a `ScoredPostsQuery` to generate a ranked feed of post candidates for the "For You" timeline. It involves:

- **Query Initiation**: gRPC entry point in Rust.
- **Hydration**: Enrich query with user data.
- **Sourcing**: Fetch candidates (parallel, including Python retrieval).
- **Candidate Hydration**: Add metadata to candidates.
- **Filtering**: Remove ineligible candidates (using bloom filters and ID lists).
- **Scoring**: Compute relevance scores (including Python ranking).
- **Selection**: Top-K by score.
- **Post-Selection Processing**: Final hydration and filtering.
- **Side Effects**: Caching and logging.
- **Response**: Protobuf output.

The flow emphasizes parallelism (e.g., async fetches), error resilience (e.g., fallbacks on ML failures), and optimizations like bloom filters for O(1) probabilistic checks on large `seen_ids`/`served_ids` sets. Total latency targets <100ms via gRPC to Python services.

### Visual Flow Diagram

```mermaid
graph TD
    A[User Query: ScoredPostsQuery<br/>(viewer_id, seen_ids, served_ids, bloom_filter_entries)] --> B[HomeMixerServer<br/>Validate & Construct Query]
    B --> C[Query Hydration<br/>(UserActionSeq, UserFeatures)]
    C --> D[Candidate Sourcing<br/>Parallel Sources<br/>(PhoenixSource → Python Retrieval)]
    D --> E[Candidate Hydration<br/>(Core Data, Author Info)]
    E --> F[Pre-Scoring Filters<br/>(Deduplication, Seen/Served via Bloom Filters)]
    F --> G[Scoring<br/>Sequential Scorers<br/>(PhoenixScorer → Python Ranking)]
    G --> H[Selection<br/>Top-K by Weighted Score]
    H --> I[Post-Selection<br/>Hydration & Filters<br/>(VF, Dedup)]
    I --> J[Side Effects<br/>(CacheRequestInfoSideEffect)]
    J --> K[Response: ScoredPostsResponse<br/>(Ranked Feed)]
    L[Bloom Filters & ID Lists<br/>Probabilistic Dedup] -.-> F
    M[Cache Store<br/>(StratoClient)] -.-> J
    N[gRPC Clients<br/>(PhoenixRetrievalClient, PhoenixPredictionClient)] -.-> D
    N -.-> G
```

## Detailed Data Flow

### 1. Query Initiation and Server Entry Point
The process begins in the `HomeMixerServer` ([Rust Home Mixer Overview](../rust-home-mixer-overview.md)), implementing the gRPC `ScoredPostsService`. An incoming `ScoredPostsQuery` protobuf (from the client app) contains:

- Core metadata: `viewer_id`, `client_app_id`, `country_code`, `language_code`.
- Deduplication: `seen_ids` (viewed posts), `served_ids` (recently shown), `bloom_filter_entries` (pre-computed bloom filter bits for efficient checks).
- Flags: `in_network_only` (limit to followed users), `is_bottom_request` (pagination).

Validation ensures `viewer_id` > 0. The server builds a Rust `ScoredPostsQuery` struct (in `home-mixer/candidate_pipeline/query.rs`), adding:
- `request_id` (generated via `generate_request_id` in `util/request_util.rs` for tracing).
- Dynamic fields: `user_action_sequence` and `user_features` (populated later).

The pipeline executes via `phx_candidate_pipeline.execute(query).await`, logging latency and candidate counts.

### 2. Query Hydration Stage
Hydrators (in `query_hydrators` vector) enrich the query asynchronously:

- `UserActionSeqQueryHydrator`: Fetches `user_action_sequence` (protobuf with last ~128 engagements like likes/reposts) using `UserActionSequenceFetcher`.
- `UserFeaturesQueryHydrator`: Retrieves user demographics/interests via `StratoClient`.

These run in parallel, preparing data for ML inference (e.g., user history embeddings).

### 3. Candidate Sourcing Stage (PhoenixSource to Python Retrieval)
Sources (implementing `Source` trait) fetch raw candidates in parallel, merging into `Vec<PostCandidate>`.

- **PhoenixSource** (in `home-mixer/sources/phoenix_source.rs`): Enabled if `!in_network_only`. Calls `phoenix_retrieval_client.retrieve(user_id, sequence, max_results=1000)` (gRPC).
  - **Python Integration**: Hits `RecsysRetrievalInferenceRunner` (in `phoenix/runners.py`, see [Recsys Retrieval Model](../recsys-retrieval-model.md)).
    - Computes user embedding from history via transformer (user tower).
    - Performs ANN retrieval (dot-product) against pre-loaded post embeddings (`corpus_embeddings`).
    - Returns `top_k_indices` and `scores` (e.g., top 1000 out-of-network candidates).
  - Rust maps to `PostCandidate` structs: Sets `tweet_id`, `author_id`, `served_type=ForYouPhoenixRetrieval`. Filters invalids.
- Other sources: E.g., `ThunderSource` for in-network posts from follows.

Stats via `#[xai_stats_macro::receive_stats]`.

### 4. Candidate Hydration Stage
Hydrators (implementing `Hydrator` trait) enrich candidates in parallel:

- `CoreDataCandidateHydrator`: Fetches text/media via `TESClient`.
- `GizmoduckCandidateHydrator`: Adds author details (e.g., `screen_name`) via `GizmoduckClient`.
- Others: Video duration, subscriptions, in-network flags.

`seen_ids`/`served_ids` may skip redundant fetches.

### 5. Pre-Scoring Filtering Stage
Filters (implementing `Filter` trait) remove candidates sequentially:

- `DropDuplicatesFilter`: Dedups by `tweet_id`.
- **PreviouslySeenPostsFilter** / **PreviouslyServedPostsFilter**: Check against `seen_ids`/`served_ids`.
  - **Bloom Filters**: Use `bloom_filter_entries` for fast probabilistic "maybe seen" queries (false positives OK, avoids full scans on millions of IDs). Full lists used for confirmation.
- Others: Age, muted keywords, blocked authors, subscriptions.

See [Filters](../filters.md) for details.

### 6. Scoring Stage (PhoenixScorer to Python Ranking)
Scorers (implementing `Scorer` trait) compute scores sequentially (preserve order/count).

- **PhoenixScorer** (in `home-mixer/scorers/phoenix_scorer.rs`): Builds `tweet_infos` from candidates, calls `phoenix_client.predict(user_id, sequence, tweet_infos)` (gRPC).
  - **Python Integration**: Hits `RecsysInferenceRunner` (in `phoenix/runners.py`, see [Recsys Model](../recsys-model.md)).
    - Builds embeddings (user/history/candidates via `block_*_reduce`).
    - Runs transformer forward (Grok-1 inspired, with masking).
    - Outputs action probabilities (e.g., `p_favorite`, `p_reply`) via sigmoid; ranks by primary score.
    - Returns per-candidate `ActionPredictions`.
  - Rust: Parses to `PhoenixScores` (e.g., `favorite_score`), updates candidates with scores, `prediction_request_id`, `last_scored_at_ms`.
- Follow-ups: `WeightedScorer` (sum of scores), `AuthorDiversityScorer` (variety), `OONScorer` (out-of-network boost).

Failures return unscored candidates.

### 7. Selection Stage
`TopKScoreSelector` sorts by final `weighted_score`, selects top-K (e.g., 1500).

### 8. Post-Selection Processing
- Hydrators: E.g., `VFCandidateHydrator` (visibility metadata).
- Filters: `VFFilter` (remove spam/NSFW), `DedupConversationFilter`.

### 9. Side Effects Stage (CacheRequestInfoSideEffect)
Side effects (implementing `SideEffect` trait) run async post-selection:

- **CacheRequestInfoSideEffect**: Uses `StratoClient` to cache query metadata (e.g., `request_id`, selected IDs, scores, timestamps) in a distributed store. Updates `seen_ids`/`served_ids` for session continuity, analytics, and dedup in future requests. Integrates with bloom filters for next-query prep.

See [Auxiliary Components](../auxiliary-components.md) for caching details.

### 10. Response Generation
Maps `selected_candidates` to `ScoredPost` protos (with `tweet_id`, `score`, `served_type`, screen names). Returns `ScoredPostsResponse` with logging.

## Key Integrations and Optimizations
- **Rust-Python Bridge**: gRPC clients (`PhoenixRetrievalClient`, `PhoenixPredictionClient`) for low-latency inference. See [Runners](../runners.md).
- **Deduplication Handling**: `seen_ids`/`served_ids` + bloom filters prevent repetition; cached side effects persist across sessions.
- **Error Handling**: Fallbacks (e.g., no scores on ML failure), timeouts.
- **Scalability**: Parallel stages, probabilistic filters; Python serves batched inference.
- **Testing**: Unit/integration via [Testing Phoenix](../testing-phoenix.md); config in [Configuration](../configuration.md).

This flow ensures diverse, personalized feeds while minimizing compute (retrieval: millions → thousands; ranking: hundreds → top-K).