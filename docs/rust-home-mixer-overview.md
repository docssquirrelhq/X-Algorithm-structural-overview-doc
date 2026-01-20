---
title: Rust Home-Mixer Overview
description: A structural overview of the Rust-based home-mixer component, including its role in mixing and selecting candidate posts for recommendations, key modules, and pipeline interactions.
slug: rust-home-mixer-overview
sidebar_label: Rust Home-Mixer Overview
sidebar_position: 2
tags: [rust, home-mixer, pipeline, recommendations]
---

# Rust Home-Mixer Overview

The Rust-based home-mixer is a core orchestration component in the recommendation system, designed to assemble personalized "For You" feeds by retrieving, processing, and ranking candidate posts (tweets) from various sources. It serves as a central mixer that combines in-network (e.g., posts from followed users) and out-of-network (e.g., algorithmic recommendations) content, applies filters for safety and relevance, scores for quality and diversity, and selects the top candidates to create an engaging, diverse timeline. This component exposes a gRPC service (`ScoredPostsService`) that handles queries for scored posts based on user context—such as viewer ID, seen/served post IDs, location, and user features—and returns a ranked list of `ScoredPost` objects containing tweet details, scores, and metadata.

Built for high throughput and scalability, home-mixer uses asynchronous Rust (via Tokio), parallel execution for stages like source fetching and hydration, and configurable parameters (e.g., max results per source, top-K selection size) to tune performance. It processes a `ScoredPostsQuery` input and produces a `ScoredPostsResponse` output, leveraging a modular pipeline framework to ensure maintainability and extensibility. Key data structures include `PostCandidate` (enriched post data with scores, network status, and metadata) and `PipelineResult` (tracking outputs across stages).

For more details on related components, see:
- [Candidate Pipeline](candidate-pipeline)
- [Sources](sources)
- [Filters](filters)
- [Scorers](scorers)
- [Selectors](selectors)

## Role in Recommendations

Home-mixer's primary role is to balance recency, relevance, diversity, and safety in feed generation. It:
- Retrieves candidates from multiple upstream services (e.g., in-network timelines and recommendation models).
- Hydrates queries with user context (e.g., followed users, muted keywords).
- Enriches and filters candidates to remove low-quality or ineligible content.
- Scores and ranks them using heuristics and model predictions.
- Selects a fixed-size set of top posts, applying post-selection checks.
- Logs metrics and side effects (e.g., caching impressions) for monitoring.

This process ensures feeds are personalized while adhering to platform policies, such as visibility rules and deduplication. The component runs as a standalone gRPC server with features like request compression (Gzip/Zstd), reflection for debugging, and readiness probes.

## Main Modules

Home-mixer is structured into modular components under the `home-mixer/` directory, each implementing trait-based interfaces from the `candidate_pipeline` framework. This allows pluggable swaps (e.g., adding new sources) without disrupting the core flow. Below is an overview of the key modules and their interactions.

### Candidate Pipeline (`candidate_pipeline`)

The foundational orchestrator that defines the end-to-end workflow. It implements the `CandidatePipeline` trait, which provides accessors for pipeline stages (e.g., `sources()`, `filters()`, `scorers()`) and enforces a sequential execution order with parallelizable steps.

- **Key Components**:
  - `PipelineStage` enum: Tracks stages like `QueryHydrator`, `Source`, `Hydrator`, `Filter`, `Scorer`, `PostSelectionHydrator`, and `PostSelectionFilter` for logging and error handling.
  - `PipelineResult<Q, C>`: Accumulates outputs, including `retrieved_candidates`, `filtered_candidates`, `selected_candidates`, and the hydrated query (shared via `Arc<Q>`).
  - `HasRequestId` trait: Generates a unique request ID for tracing across stages.

- **Interactions**: The `execute` method runs the full pipeline, handling errors (e.g., logging hydrator failures), truncating results to configured sizes, and applying side effects. It uses `Arc` for safe sharing and parallel ops like `join_all` for efficiency.

### Sources (`sources`)

Responsible for fetching raw candidate posts from upstream services in parallel, providing diversity (e.g., in-network vs. out-of-network).

- **Key Implementations** (implement `Source<Q, C>` trait with `get_candidates(&query) -> Result<Vec<C>, String>`):
  - `ThunderSource`: Retrieves recent in-network posts via `ThunderClient` (gRPC to `InNetworkPostsService`). Builds `GetInNetworkPostsRequest` with user/followed IDs and limits (e.g., `THUNDER_MAX_RESULTS`). Outputs `PostCandidate`s with metadata like `tweet_id`, `author_id`, and `served_type: ForYouInNetwork`.
  - `PhoenixSource`: Fetches out-of-network recommendations via `PhoenixRetrievalClient`, using `user_action_sequence` for personalization (up to `PHOENIX_MAX_RESULTS`). Maps to `PostCandidate`s with `served_type: ForYouPhoenixRetrieval`. Skipped for in-network-only queries.

- **Interactions**: Enabled via `enable(&query)`. Outputs concatenate into the pipeline's retrieved candidates, feeding into hydrators. Errors (e.g., service unavailability) are logged as strings.

### Filters (`filters`)

Remove ineligible candidates at pre-scoring and post-selection stages to optimize compute and ensure quality. They partition inputs into `kept` and `removed` via `FilterResult<C>`.

- **Key Implementations** (implement `Filter<Q, C>` with `filter(&query, candidates) -> Result<FilterResult<C>, String>`):
  - `MutedKeywordFilter`: Checks tweet text against user-muted keywords using `TweetTokenizer` and `MatchTweetGroup`.
  - `VFFilter` (Visibility Filter): Drops posts violating safety rules (e.g., spam) based on `visibility_reason`.
  - `DedupConversationFilter`: Keeps the highest-scored post per conversation (via `get_conversation_id`).
  - `PreviouslyServedPostsFilter`: Removes served or related posts for bottom-requests.
  - `CoreDataHydrationFilter`: Validates basic data (e.g., non-empty text).
  - `DropDuplicatesFilter`: Deduplicates by `tweet_id`.
  - `IneligibleSubscriptionFilter`: Drops subscription-only posts unless user subscribes.

- **Interactions**: Run sequentially after hydration; accumulate removals. Enabled via `enable(&query)`. Outputs proceed to scorers or final assembly.

### Scorers (`scorers`)

Assign or adjust scores for ranking, processing in parallel and updating `PostCandidate.score` (often from `weighted_score` or model predictions).

- **Key Implementations** (implement `Scorer<Q, C>` with `score(&query, candidates) -> Result<Vec<C>, String>`):
  - `AuthorDiversityScorer`: Applies decay to repeated authors (e.g., `multiplier = (1 - floor) * decay^position + floor`).
  - `OONScorer`: Weights out-of-network scores lower (via `OON_WEIGHT_FACTOR`) to favor in-network.

- **Interactions**: Run after pre-scoring filters, using hydrated data (e.g., Phoenix scores like `favorite_score`). Update candidates in-place; outputs go to selectors.

### Selectors (`selectors`)

Rank and truncate candidates to a target size (e.g., top-K by score).

- **Key Implementation** (implements `Selector<Q, C>`):
  - `TopKScoreSelector`: Sorts descending by `candidate.score` (or fallback) and selects up to `TOP_K_CANDIDATES_TO_SELECT`.

- **Interactions**: Single selector post-scoring; enabled via `enable(&query)`. Outputs feed post-selection stages.

### Supporting Modules

- **Hydrators** (`candidate_hydrators` and `query_hydrators`): Enrich data in parallel (implement `Hydrator<Q, C>`). Examples: `CoreDataCandidateHydrator` (fetches tweet text/author via `TESClient`), `InNetworkCandidateHydrator` (sets network flags), `SubscriptionHydrator` (checks subscriptions). Query hydrators prepare user features (e.g., followed IDs).
- **Side Effects** (`side_effects`): Post-pipeline actions like caching (via `SideEffect` trait).
- **Server** (`server`): Wraps the pipeline in `HomeMixerServer`, handling gRPC `get_scored_posts`: validates input, executes pipeline, maps to `ScoredPost` (e.g., adds `screen_names`), and logs timings.

## Pipeline Interactions: From Query Hydration to Final Selection

The pipeline (`CandidatePipeline::execute`) processes stages sequentially with parallelism where possible, starting from a `ScoredPostsQuery` and yielding a `PipelineResult`. High-level flow:

1. **Query Hydration**: Parallel `QueryHydrator`s fetch/enrich user context (e.g., `followed_user_ids`, `muted_keywords`) into `UserFeatures`. Merge non-fatally.

2. **Candidate Fetching**: Parallel `Source::get_candidates` (e.g., Thunder + Phoenix) to retrieve raw `PostCandidate`s (e.g., 100-500 total).

3. **Candidate Hydration**: Parallel `Hydrator::hydrate` enriches with metadata (e.g., text, in-network flags via TESClient). Update originals.

4. **Pre-Scoring Filtering**: Sequential `Filter::filter` removes invalids (e.g., duplicates, muted content). Track `filtered_candidates`.

5. **Scoring**: Parallel/sequential `Scorer::score` adjusts scores (e.g., diversity, OON weighting). Update via `update`.

6. **Selection**: `Selector::select` sorts and truncates (e.g., top 50 by score).

7. **Post-Selection Hydration**: Optional parallel hydrators for final enrichment.

8. **Post-Selection Filtering**: Sequential filters (e.g., visibility, conversation dedup). Append to filtered list.

9. **Truncation and Side Effects**: Limit to `result_size()` (e.g., 20-30). Run `SideEffect`s (e.g., logging).

10. **Response Assembly**: Map selected candidates to `ScoredPost` (e.g., extract visibility, screen names). Return via gRPC.

This modular design minimizes bottlenecks: early filters cut load, parallelism speeds fetches/hydrations, and traits enable testing/swaps. For implementation details, refer to [Candidate Pipeline](candidate-pipeline).

:::note
The pipeline is configurable (e.g., via `params` module) for A/B testing or scaling. Metrics (e.g., latency, candidate counts) are emitted via `xai_stats_macro`.
:::
