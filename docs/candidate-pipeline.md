---
title: Candidate Pipeline
description: Detailed structure of the candidate_pipeline module in Rust's home-mixer, including key structs, pipeline flow, hydration, feature extraction, and proto integrations.
slug: candidate-pipeline
sidebar_label: Candidate Pipeline
sidebar_position: 3
---

# Candidate Pipeline in Rust's Home-Mixer

The `candidate_pipeline` module in the home-mixer serves as a modular framework for constructing recommendation pipelines in Rust. It defines a series of traits that enable parallelizable and configurable stages for retrieving, enriching, filtering, scoring, and selecting post candidates. The core abstraction is the `CandidatePipeline` trait, which orchestrates the entire flow for a given query type (e.g., `ScoredPostsQuery`) and candidate type (e.g., `PostCandidate`). This trait requires implementations to provide accessors for components like query hydrators, sources, hydrators, filters, scorers, selectors, post-selection hydrators/filters, and side effects.

For an overview of the home-mixer, see the [Rust Home-Mixer Overview](../rust-home-mixer-overview.md). Related components like [Selectors](../selectors.md), [Scorers](../scorers.md), and [Sources](../sources.md) are detailed in their respective pages.

## Key Components and Traits

The module provides traits for each pipeline stage, allowing for composition and parallelism:

- **`Source<Q, C>`**: Retrieves raw candidates from external sources (e.g., in-network posts or ML-retrieved posts). See [Sources](../sources.md) for examples like ThunderSource.
- **`QueryHydrator<Q>`**: Enriches the query with user-specific data before sourcing.
- **`Hydrator<Q, C>`**: Adds features to candidates (e.g., text, author details).
- **`Filter<Q, C>`**: Partitions candidates into kept and removed sets based on rules (e.g., duplicates, muted keywords). See [Filters](../filters.md).
- **`Scorer<Q, C>`**: Computes or adjusts scores for ranking (e.g., ML predictions, diversity adjustments). See [Scorers](../scorers.md).
- **`Selector<Q, C>`**: Sorts and truncates candidates (e.g., top-K by score). See [Selectors](../selectors.md).
- **`SideEffect<Q, C>`**: Handles asynchronous post-processing (e.g., caching).

Additional utilities include:
- **`PipelineStage` Enum**: Defines stages like `QueryHydrator`, `Source`, `Hydrator`, `Filter`, `Scorer` for logging and error tracking.
- **`PipelineResult<Q, C>`**: Output struct containing `retrieved_candidates`, `filtered_candidates`, `selected_candidates`, and the hydrated query.
- **`HasRequestId` Trait**: Ensures queries have a stable ID for tracing, implemented as `fn request_id(&self) -> &str`.

The framework emphasizes parallelism (e.g., via `futures::join_all` for hydrators/sources) and fault tolerance (e.g., skipping failed components and logging errors). Filters and scorers run sequentially where order matters, while sources and hydrators can parallelize.

## Key Structs

### ScoredPostsQuery

This struct represents the input query for fetching and scoring posts, encapsulating user context, request metadata, and exclusion lists. It's designed for the home-mixer's "For You" timeline, supporting both in-network (followed users) and out-of-network content.

```rust
#[derive(Clone, Default, Debug)]
pub struct ScoredPostsQuery {
    pub user_id: i64,
    pub client_app_id: i32,
    pub country_code: String,
    pub language_code: String,
    pub seen_ids: Vec<i64>,
    pub served_ids: Vec<i64>,
    pub in_network_only: bool,
    pub is_bottom_request: bool,
    pub bloom_filter_entries: Vec<ImpressionBloomFilterEntry>,
    pub user_action_sequence: Option<xai_recsys_proto::UserActionSequence>,
    pub user_features: UserFeatures,
    pub request_id: String,
}
```

#### Fields Explanation
- `user_id`: The viewer's Twitter user ID.
- `seen_ids` and `served_ids`: Lists of post IDs to exclude (e.g., recently viewed or served to avoid repetition).
- `bloom_filter_entries`: Probabilistic filters for efficient duplicate detection of impressions.
- `user_action_sequence`: Optional engagement history (likes, retweets, etc.) from `xai_recsys_proto`, used for ML scoring.
- `user_features`: Nested struct for user-specific data (see [UserFeatures](#userfeatures-query-features) below).
- `in_network_only`: Flag to limit to followed users.
- `is_bottom_request`: Indicates if this is for loading more content at the feed's bottom, affecting filters like previously served posts.

The `new` constructor generates a unique `request_id` and sets defaults:

```rust
impl ScoredPostsQuery {
    pub fn new(
        user_id: i64,
        client_app_id: i32,
        country_code: String,
        language_code: String,
        seen_ids: Vec<i64>,
        served_ids: Vec<i64>,
        in_network_only: bool,
        is_bottom_request: bool,
        bloom_filter_entries: Vec<ImpressionBloomFilterEntry>,
    ) -> Self {
        let request_id = format!("{}-{}", generate_request_id(), user_id);
        Self {
            user_id,
            client_app_id,
            country_code,
            language_code,
            seen_ids,
            served_ids,
            in_network_only,
            is_bottom_request,
            bloom_filter_entries,
            user_action_sequence: None,
            user_features: UserFeatures::default(),
            request_id,
        }
    }
}
```

It implements `HasRequestId` for logging and `GetTwitterContextViewer` for proto compatibility, providing viewer details like `user_id` and locale.

### PostCandidate

This struct models a single post candidate, aggregating core attributes, scores, and metadata. It's enriched progressively through the pipeline and used for filtering/scoring.

```rust
#[derive(Clone, Debug, Default)]
pub struct PostCandidate {
    pub tweet_id: i64,
    pub author_id: u64,
    pub tweet_text: String,
    pub in_reply_to_tweet_id: Option<u64>,
    pub retweeted_tweet_id: Option<u64>,
    pub retweeted_user_id: Option<u64>,
    pub phoenix_scores: PhoenixScores,
    pub prediction_request_id: Option<u64>,
    pub last_scored_at_ms: Option<u64>,
    pub weighted_score: Option<f64>,
    pub score: Option<f64>,
    pub served_type: Option<pb::ServedType>,
    pub in_network: Option<bool>,
    pub ancestors: Vec<u64>,
    pub video_duration_ms: Option<i32>,
    pub author_followers_count: Option<i32>,
    pub author_screen_name: Option<String>,
    pub retweeted_screen_name: Option<String>,
    pub visibility_reason: Option<vf::FilteredReason>,
    pub subscription_author_id: Option<u64>,
}
```

#### Fields Explanation
- Core post data: `tweet_id`, `author_id`, `tweet_text`, reply/retweet links (`in_reply_to_tweet_id`, etc.).
- Scores: `phoenix_scores` (nested struct for ML predictions like `favorite_score`, `reply_score`), `weighted_score` (combined relevance), `score` (final adjusted score).
- Metadata: `served_type` (e.g., `ForYouInNetwork` from `xai_home_mixer_proto`), `in_network` (followed user?), `ancestors` (conversation chain), `visibility_reason` (filtering reasons from `xai_visibility_filtering`).
- Enriched features: `video_duration_ms`, author details, subscription info.

`PhoenixScores` is a nested struct for granular ML outputs:

```rust
#[derive(Clone, Debug, Default)]
pub struct PhoenixScores {
    pub favorite_score: Option<f64>,
    pub reply_score: Option<f64>,
    pub retweet_score: Option<f64>,
    // ... other action scores like click_score, dwell_score, etc.
    pub dwell_time: Option<f64>,  // Continuous metric
}
```

It implements `CandidateHelpers` for utility methods, like extracting screen names into a `HashMap<u64, String>` for authors and retweeters.

### UserFeatures (Query Features)

`UserFeatures` is a nested struct within `ScoredPostsQuery`, holding pre-fetched user data for personalization and filtering. It's hydrated early in the pipeline.

From usage in hydrators and filters:
- Includes `followed_user_ids: Vec<i64>` (list of followed accounts for in-network checks).
- `muted_keywords: Vec<String>` (user-muted terms for keyword filtering).
- Defaults to empty or zero values if not available.

It's fetched via a `QueryHydrator` using external services (e.g., Strato) and decoded from `StratoValue<UserFeatures>` protos.

## Overall Pipeline Flow for Processing Scored Posts

The pipeline processes scored posts via the `CandidatePipeline::execute` method, which chains stages for a query like `ScoredPostsQuery` to produce `PipelineResult<ScoredPostsQuery, PostCandidate>`. The flow is:

1. **Query Hydration**: Parallelize `QueryHydrator`s to enrich the query (e.g., fetch `UserFeatures` like followed IDs and action sequence).

   ```rust
   async fn hydrate_query(&self, query: Q) -> Q {
       let hydrators: Vec<_> = self.query_hydrators().iter().filter(|h| h.enable(&query)).collect();
       let results = join_all(hydrators.iter().map(|h| h.hydrate(&query))).await;
       let mut hydrated_query = query;
       for (hydrator, result) in hydrators.iter().zip(results) {
           if let Ok(hydrated) = result {
               hydrator.update(&mut hydrated_query, hydrated);
           }
       }
       hydrated_query
   }
   ```

   Example: `UserFeaturesQueryHydrator` calls a Strato client to get user features and updates the query.

2. **Candidate Sourcing**: Parallelize `Source`s to fetch raw `PostCandidate`s (e.g., from Thunder for in-network, Phoenix for out-of-network).
   - Sources filter via `enable` (e.g., Phoenix skips if `in_network_only`).
   - Example: `ThunderSource` queries followed users' recent posts via gRPC.

3. **Candidate Hydration**: Parallelize `Hydrator`s to enrich candidates (pre- and post-selection).
   - Pre-selection: Core data (text, replies), author info (via Gizmoduck), video duration, in-network status, subscriptions.
   - Post-selection: Additional features after top-K selection.

   ```rust
   let hydrated_candidates = self.hydrate(&hydrated_query, candidates).await;
   // Similar to hydrate_query, but for candidates
   ```

   Example: `CoreDataCandidateHydrator` fetches tweet text/author via TES client and updates fields like `tweet_text`.

4. **Filtering (Pre- and Post-Selection)**: Sequential `Filter`s partition candidates.
   - Pre-scoring: Dedup, core data checks, muted keywords, previously seen/served, visibility (e.g., `VFFilter` drops unsafe content).
   - Post-selection: Final checks like conversation dedup.

   ```rust
   async fn filter(&self, query: &Q, candidates: Vec<C>) -> (Vec<C>, Vec<C>) {
       let mut all_removed = Vec::new();
       for filter in self.filters().iter().filter(|f| f.enable(query)) {
           match filter.filter(query, candidates.clone()).await {
               Ok(result) => {
                   candidates = result.kept;
                   all_removed.extend(result.removed);
               }
               Err(_) => { /* Log and backup */ }
           }
       }
       (candidates, all_removed)
   }
   ```

   Returns `FilterResult { kept, removed }` for each.

5. **Scoring**: Sequential `Scorer`s adjust scores.
   - PhoenixScorer: Calls ML model with `user_action_sequence` for predictions (integrates `xai_recsys_proto::PredictNextActionsResponse`).
   - Others: Weighted combo, author diversity (decay repeats), OON adjustment (downweight out-of-network).

   ```rust
   async fn score(&self, query: &Q, mut candidates: Vec<C>) -> Vec<C> {
       for scorer in self.scorers().iter().filter(|s| s.enable(query)) {
           if let Ok(scored) = scorer.score(query, &candidates).await {
               scorer.update_all(&mut candidates, scored);
           }
       }
       candidates
   }
   ```

   Updates like `candidate.phoenix_scores.favorite_score = ...` from proto responses.

6. **Selection**: Sort by scorer-defined `score()` (e.g., `TopKScoreSelector` uses `candidate.score` or `NEG_INFINITY` fallback) and truncate to `result_size()` (e.g., top-K param).

   ```rust
   fn select(&self, query: &Q, candidates: Vec<C>) -> Vec<C> {
       if self.selector().enable(query) {
           self.selector().select(query, candidates)
       } else {
           candidates
       }
   }
   ```

7. **Side Effects**: Spawn async tasks for non-blocking ops (e.g., logging, caching) on final candidates.

   ```rust
   fn run_side_effects(&self, input: Arc<SideEffectInput<Q, C>>) {
       tokio::spawn(async move {
           let futures = self.side_effects().iter()
               .filter(|se| se.enable(input.query.clone()))
               .map(|se| se.run(input.clone()));
           let _ = join_all(futures).await;
       });
   }
   ```

In the server (`HomeMixerServer`), `get_scored_posts` creates a `ScoredPostsQuery` from a proto request, executes the pipeline (e.g., via `PhoenixCandidatePipeline`), and maps `selected_candidates` to `ScoredPostsResponse` protos, including scores and screen names. For more on runners, see [Runners](../runners.md).

## Hydration and Feature Extraction

- **Hydration**: Split into query-level (e.g., user features via Strato) and candidate-level (parallel fetches via clients like TES for text/media, Gizmoduck for authors). Updates use `Hydrator::update(&mut candidate, hydrated)` to merge fields without overwriting.
  - Example: `InNetworkCandidateHydrator` computes `in_network` from `followed_user_ids`.
  - Error handling: Failed hydrators log and skip, preserving defaults.

- **Feature Extraction**: Happens in hydrators:
  - Core: Tweet text, replies, retweets via `TESClient::get_tweet_core_datas`.
  - Author: Followers count, screen names via `GizmoduckClient::get_users`.
  - Media: Video duration via `TESClient::get_tweet_media_entities`.
  - Subscriptions: Author IDs via `TESClient::get_subscription_author_ids`.
  - ML Features: Extracted in scorers from Phoenix predictions (e.g., action probs from `top_log_probs` in `xai_recsys_proto`).

:::note
Hydration is parallelized where possible to minimize latency, but sequential for dependencies (e.g., scoring after hydration).
:::

## Integration with External Protos

The module integrates tightly with Twitter's internal protos for interoperability:

- **`xai_home_mixer_proto`**: For request/response (e.g., `ScoredPostsQuery` proto to Rust struct, `ScoredPost` with `tweet_id`, `score`, `screen_names`).
- **`xai_recsys_proto`**: Core for ML. `UserActionSequence` in query for engagement history; `PredictNextActionsResponse` in `PhoenixScorer` provides distributions (e.g., `candidate_distributions` with `top_log_probs` for actions like `ActionName::Favorite`, continuous values like `dwell_time`).
  - Example in scorer:
    ```rust
    let action_probs: HashMap<usize, f64> = distribution.top_log_probs.iter()
        .enumerate()
        .map(|(idx, log_prob)| (idx, (*log_prob as f64).exp()))
        .collect();
    // Maps to PhoenixScores, e.g., favorite_score from ActionName index
    ```
- **`xai_thunder_proto`**: For in-network sourcing (`GetInNetworkPostsRequest` with `following_user_ids`).
- **`xai_visibility_filtering`**: For `FilteredReason` in visibility checks.
- **`xai_strato`**: For decoding user features (`StratoValue<UserFeatures>`).
- **`xai_post_text`**: For tokenization in muted keyword matching.

This ensures seamless data flow from gRPC services (e.g., Thunder, Phoenix) into the Rust pipeline, with error mapping to strings for resilience. The result is a ranked list of up to `TOP_K_CANDIDATES_TO_SELECT` posts, logged with timings.

For broader context on data flow and models, refer to [Data Flow](../data-flow.md) and [RecSys Model](../recsys-model.md).