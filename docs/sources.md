---
title: Sources Module
slug: sources
description: Overview of the sources module in Rust's home-mixer crate, including PhoenixSource and ThunderSource. Details on fetching candidate posts, handling ScoredPostsQuery, and integration with the Thunder post store.
sidebar_label: Sources
sidebar_position: 5
---

# Sources Module

The `sources` module in the `home-mixer` crate serves as the entry point for candidate sourcing in the recommendation pipeline. It re-exports two primary source implementations: `PhoenixSource` and `ThunderSource`. These sources are responsible for fetching post candidates (`PostCandidate` instances) based on a `ScoredPostsQuery`, which encapsulates user context such as `user_id`, `user_action_sequence`, `followed_user_ids` (from `user_features`), `in_network_only` flag, and other metadata like `seen_ids` or `served_ids`.

This module integrates with the broader [candidate pipeline](/docs/candidate-pipeline.md) and relies on external services for retrieval. Below, we describe the key components, their implementations, and how they handle queries and fetching.

## Module Structure

- **Location**: `home-mixer/sources/mod.rs`
- **Key Declarations**:
  ```rust
  pub mod phoenix_source;
  pub mod thunder_source;
  ```
- **Pipeline Integration**: Sources are collected into a `Vec<Box<dyn Source<ScoredPostsQuery, PostCandidate>>>` and executed in parallel during the `fetch_candidates` stage of the [CandidatePipeline](/docs/candidate-pipeline.md). The pipeline's `sources()` method returns a slice of these boxed traits, enabling dynamic dispatch for candidate retrieval.

For more on the overall pipeline, see the [Candidate Pipeline](/docs/candidate-pipeline.md) documentation.

## PhoenixSource

### Purpose
`PhoenixSource` fetches out-of-network (OON) post candidates using a retrieval model. It is ideal for content discovery beyond the user's followed accounts, leveraging embedding-based similarity search.

### Struct Definition
Defined in `phoenix_source.rs`:
```rust
pub struct PhoenixSource {
    pub phoenix_retrieval_client: Arc<dyn PhoenixRetrievalClient + Send + Sync>,
}
```
- Depends on a `PhoenixRetrievalClient` trait (e.g., `ProdPhoenixRetrievalClient`), which communicates with the Phoenix retrieval service (a two-tower embedding model).

### Trait Implementation: `Source<ScoredPostsQuery, PostCandidate>`
- **`enable(&self, query: &ScoredPostsQuery) -> bool`**: Returns `!query.in_network_only`, so it only activates for mixed or OON feeds.
- **`get_candidates(&self, query: &ScoredPostsQuery) -> Result<Vec<PostCandidate>, String>`**:
  - Validates the presence of `query.user_action_sequence` (user engagement history); errors if absent.
  - Asynchronously calls `phoenix_retrieval_client.retrieve(user_id as u64, sequence.clone(), p::PHOENIX_MAX_RESULTS)`.
  - Processes the response:
    - Flattens `top_k_candidates` into scored candidates.
    - Filters and maps to `PostCandidate` structs, setting:
      - `tweet_id`: From `tweet_info.tweet_id as i64`.
      - `author_id`: From `tweet_info.author_id`.
      - `in_reply_to_tweet_id`: From `tweet_info.in_reply_to_tweet_id`.
      - `served_type`: `pb::ServedType::ForYouPhoenixRetrieval`.
      - Defaults for other fields (e.g., no ancestors or retweet info).
  - Uses metrics with `#[xai_stats_macro::receive_stats]` for observability.
- **`name(&self) -> &'static str`**: Returns `"PhoenixSource"` for logging.

### Integration Notes
- Relies on `user_action_sequence` hydrated earlier via [UserActionSeqQueryHydrator](/docs/candidate-pipeline.md#hydration).
- Candidates are enriched downstream (e.g., via `CoreDataCandidateHydrator` for text/media).
- Configured with `p::PHOENIX_MAX_RESULTS` from `params.rs` to cap results.

## ThunderSource

### Purpose
`ThunderSource` fetches in-network (IN) post candidates from recent posts by followed users. It uses a real-time in-memory store for low-latency retrieval, focusing on social graph-based content.

### Struct Definition
Defined in `thunder_source.rs`:
```rust
pub struct ThunderSource {
    pub thunder_client: Arc<ThunderClient>,
}
```
- Depends on `ThunderClient`, which handles gRPC connections to Thunder clusters (e.g., `ThunderCluster::Amp`).

### Trait Implementation: `Source<ScoredPostsQuery, PostCandidate>`
- **`enable(&self, _query: &ScoredPostsQuery) -> bool`**: Always `true` (no disable conditions).
- **`get_candidates(&self, query: &ScoredPostsQuery) -> Result<Vec<PostCandidate>, String>`**:
  - Selects a random channel via `thunder_client.get_random_channel(ThunderCluster::Amp)`.
  - Creates an `InNetworkPostsServiceClient` for the channel.
  - Builds `GetInNetworkPostsRequest`:
    - `user_id`: `query.user_id as u64`.
    - `following_user_ids`: Maps `query.user_features.followed_user_ids` to `u64`.
    - `max_results`: `p::THUNDER_MAX_RESULTS`.
    - Defaults: Empty `exclude_tweet_ids`, algorithm `"default"`, `debug: false`, `is_video_request: false`.
  - Calls `client.get_in_network_posts(request).await` and handles errors.
  - Processes the response:
    - For each `post` in `response.into_inner().posts`:
      - Extracts `in_reply_to_tweet_id` and `conversation_id` (as `Option<u64>`).
      - Builds `ancestors` vector: Includes reply-to ID; adds root conversation ID if different.
      - Maps to `PostCandidate`:
        - `tweet_id`: `post.post_id`.
        - `author_id`: `post.author_id as u64`.
        - `in_reply_to_tweet_id`: As extracted.
        - `ancestors`: As built.
        - `served_type`: `pb::ServedType::ForYouInNetwork`.
        - Defaults for other fields.
  - Uses metrics with `#[xai_stats_macro::receive_stats]`.
- **`name(&self) -> &'static str`**: Returns `"ThunderSource"`.

### Integration Notes
- Uses `followed_user_ids` from [UserFeaturesQueryHydrator](/docs/candidate-pipeline.md#hydration) (fetched via Strato/SocialGraph).
- No dependency on user sequences; prioritizes social connections.
- Configured with `p::THUNDER_MAX_RESULTS` for output limits.

## Fetching Candidates and Query Handling

### Pipeline Flow
- In [candidate_pipeline.rs](/docs/candidate-pipeline.md) and `phoenix_candidate_pipeline.rs`:
  - Sources run in parallel using `join_all` in `CandidatePipeline::fetch_candidates`.
  - Each receives a hydrated `ScoredPostsQuery` (post-query hydration adds `user_action_sequence` and `user_features`).
  - Results concatenate into a single `Vec<PostCandidate>` for hydration, filtering, and [scoring](/docs/scorers.md).
  - Errors: Logs individual failures without halting; empty results are handled gracefully.
  - Parallelism: Powered by Tokio async runtime.

### ScoredPostsQuery Handling
- Initialized in `server.rs` from gRPC proto (e.g., `viewer_id`, `in_network_only`, `seen_ids`).
- Pre-sourcing hydration: Adds required fields like `user_action_sequence` (for Phoenix) and `followed_user_ids` (for Thunder).
- Sources respect flags (e.g., Phoenix skips on `in_network_only=true`); supports bottom-paging via `is_bottom_request` (though not directly in sources).
- `request_id` enables per-request logging/metrics.

## Integration with Thunder Post Store

`ThunderSource` interacts with the Thunder backend (in the `thunder/` crate) for real-time IN content.

### Thunder Backend Overview
- **Key Files**: `thunder_service.rs`, `posts/post_store.rs` (core store; no explicit `mod.rs` in `posts/`, but `post_store.rs` is central).
- **ThunderClient**: Initialized in `phoenix_candidate_pipeline::prod()` with `ThunderClient::new().await`.
- **Service Flow** (`ThunderServiceImpl::get_in_network_posts`):
  - Fetches following list if needed (via `StratoClient`).
  - Queries `PostStore` for posts from followed users (top originals + replies/retweets/videos).
  - Applies excludes, limits (`max_results`), and stats.
- **PostStore** (in-memory, thread-safe with `DashMap` and `Arc`):
  - Stores full `LightPost` by `post_id`.
  - Per-user deques: `original_posts_by_user`, `secondary_posts_by_user` (replies/retweets), `video_posts_by_user`.
  - Operations:
    - Inserts (`insert_posts`) and deletes (`mark_as_deleted` via Kafka events).
    - Trimming: Retention-based (e.g., `start_auto_trim`).
    - Retrieval: Scans timelines (limited by `MAX_ORIGINAL_POSTS_PER_AUTHOR`, etc.), filters by recency/excludes, sorts by timestamp.
  - Metrics: Prometheus gauges (e.g., `GET_IN_NETWORK_POSTS_FOUND_POSTS_PER_AUTHOR` for post counts, freshness).
  - Concurrency: Semaphores (`max_concurrent_requests`) and timeouts (`request_timeout`).
  - Ingestion: Kafka consumers (`kafka_utils::start_kafka`) process create/delete events.

### Home-Mixer to Thunder Flow
- gRPC request from `ThunderSource` hits `InNetworkPostsService`.
- Ensures sub-ms lookups for real-time IN content (retention ~days).
- Balances with Phoenix for OON discovery.

This setup unifies IN (social) and OON (discovery) sourcing via `PostCandidate`, configurable for production scaling. For configuration details, see [Configuration](/docs/configuration.md). For testing, refer to [Testing Phoenix](/docs/testing-phoenix.md).