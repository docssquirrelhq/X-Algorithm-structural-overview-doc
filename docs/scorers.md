---
title: Scorers Module
slug: scorers
description: Detailed overview of the scorers module in Rust's home-mixer, including PhoenixScorer, WeightedScorer, AuthorDiversityScorer, and OONScorer. Covers score computation for PostCandidates, dependencies, and pipeline integration.
sidebar_label: Scorers
sidebar_position: 4
---

# Scorers Module

The scorers module in the home-mixer component of the Rust codebase is a critical part of the recommendation pipeline. It is responsible for computing and refining relevance scores for `PostCandidate` instances, which represent potential posts to recommend to users. These scores are derived from machine learning predictions, weighted combinations, diversity adjustments, and network-based penalties. The module ensures that candidates are ranked effectively before final selection, promoting engaging, diverse, and relevant content.

This documentation covers the module's structure, key implementations (PhoenixScorer, WeightedScorer, AuthorDiversityScorer, and OONScorer), score computation logic, dependencies (such as Phoenix ML models), and how scorers are combined in the pipeline. For an overview of the broader home-mixer system, see [Rust Home-Mixer Overview](../rust-home-mixer-overview.md). For details on the candidate pipeline, refer to [Candidate Pipeline](../candidate-pipeline.md).

## Module Structure

The scorers module is defined in `home-mixer/scorers/mod.rs` and organizes its components into sub-modules:

```rust
pub mod author_diversity_scorer;
pub mod oon_scorer;
pub mod phoenix_scorer;
pub mod weighted_scorer;
```

Each scorer implements the `Scorer<ScoredPostsQuery, PostCandidate>` trait from the `xai_candidate_pipeline` crate. This trait standardizes the scoring interface:

- `async fn score(&self, query: &Q, candidates: &[C]) -> Result<Vec<C>, String>`: Processes a batch of candidates asynchronously, returning updated versions in the same order. Scorers do not drop candidates (filtering occurs elsewhere); they only update scores.
- `fn update(&self, candidate: &mut C, scored: C)`: Merges scored fields (e.g., `phoenix_scores`, `weighted_score`, `score`) into an existing candidate.
- `fn enable(&self, query: &Q) -> bool`: Optionally skips the scorer based on query conditions (e.g., if no user data is available).
- `fn name(&self) -> &'static str`: Provides a name for logging and monitoring.

Scorers are integrated into the `PhoenixCandidatePipeline` (detailed in [Candidate Pipeline](../candidate-pipeline.md)) as a vector of boxed trait objects. In `home-mixer/candidate_pipeline/phoenix_candidate_pipeline.rs`, they are instantiated and ordered as follows:

```rust
let phoenix_scorer = Box::new(PhoenixScorer { phoenix_client });
let weighted_scorer = Box::new(WeightedScorer);
let author_diversity_scorer = Box::new(AuthorDiversityScorer::default());
let oon_scorer = Box::new(OONScorer);
let scorers: Vec<Box<dyn Scorer<ScoredPostsQuery, PostCandidate>>> = vec![
    phoenix_scorer,
    weighted_scorer,
    author_diversity_scorer,
    oon_scorer,
];
```

In the pipeline's `execute` method (from `xai_candidate_pipeline::candidate_pipeline::CandidatePipeline`), scorers run sequentially after filtering and hydration but before selection. The pipeline invokes `self.score(&hydrated_query, kept_candidates).await`, iterating over each scorer, applying `score`, and using `update_all` to merge results progressively. This accumulates scores in `PostCandidate` fields:

- `phoenix_scores`: Raw ML predictions.
- `weighted_score`: Linear combination of predictions.
- `score`: Final adjusted score for selection.

The final `score` drives the `TopKScoreSelector` (see [Selectors](../selectors.md)) to sort and select top-K candidates. Errors in scoring propagate as strings but allow partial processing.

Dependencies common to all scorers include:
- `ScoredPostsQuery`: Contains user context (e.g., `user_action_sequence` for engagement history, `user_features`).
- `PostCandidate` fields: `author_id`, `in_network`, `video_duration_ms`, `phoenix_scores` (a struct with per-action probabilities), `weighted_score`, `score`, `prediction_request_id`, `last_scored_at_ms`.
- Parameters from `crate::params` (e.g., weights like `FAVORITE_WEIGHT`, `OON_WEIGHT_FACTOR`).
- External services: Phoenix ML client for predictions.

:::note
Scorers are designed for modularity—new ones can be added to the vector without altering the pipeline. They operate on batches for efficiency and use async for I/O-bound operations like ML inference.
:::

## PhoenixScorer

The `PhoenixScorer` fetches raw machine learning predictions from the Phoenix model, a Grok-based transformer that estimates engagement probabilities for each candidate post.

### Structure and Dependencies
- Defined in `home-mixer/scorers/phoenix_scorer.rs`.
- Holds an `Arc<dyn PhoenixPredictionClient + Send + Sync>` for async calls to the Phoenix service.
- Depends on protobuf types from `xai_recsys_proto` (e.g., `PredictNextActionsResponse`, `TweetInfo`, `ActionName`, `ContinuousActionName`).
- Requires `ScoredPostsQuery::user_action_sequence` (user's recent engagements) and candidate details (tweet/author IDs, retweet IDs).
- Phoenix predicts isolated per-candidate scores (no inter-candidate attention), enabling caching.

### Implementation
Implements `Scorer<ScoredPostsQuery, PostCandidate>`. The `score` method is monitored with `#[xai_stats_macro::receive_stats]`.

Key signature:
```rust
#[async_trait]
impl Scorer<ScoredPostsQuery, PostCandidate> for PhoenixScorer {
    async fn score(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<Vec<PostCandidate>, String> {
        // Constructs TweetInfo protos, calls phoenix_client.predict, builds predictions map,
        // extracts scores, and returns updated candidates.
    }
    fn update(&self, candidate: &mut PostCandidate, scored: PostCandidate) {
        candidate.phoenix_scores = scored.phoenix_scores;
        candidate.prediction_request_id = scored.prediction_request_id;
        candidate.last_scored_at_ms = scored.last_scored_at_ms;
    }
}
```

### Score Computation
- If `user_action_sequence` is available, builds `TweetInfo` protos from candidates (using retweet IDs if present).
- Invokes `self.phoenix_client.predict(user_id, sequence.clone(), tweet_infos).await` to obtain `PredictNextActionsResponse`.
- Processes the response in `build_predictions_map`:
  - Extracts `top_log_probs` and converts to probabilities using `exp`.
  - Handles `continuous_actions_values` (e.g., predicted dwell time).
  - Creates a `HashMap<u64, ActionPredictions>` mapping tweet IDs to predictions.
- For each candidate, retrieves predictions by tweet (or retweet) ID and calls `extract_phoenix_scores` to populate `PhoenixScores`:
  ```rust
  fn extract_phoenix_scores(&self, p: &ActionPredictions) -> PhoenixScores {
      PhoenixScores {
          favorite_score: p.get(ActionName::ServerTweetFav),
          reply_score: p.get(ActionName::ServerTweetReply),
          retweet_score: p.get(ActionName::ServerTweetRetweet),
          photo_expand_score: p.get(ActionName::ClientTweetPhotoExpand),
          click_score: p.get(ActionName::ClientTweetClick),
          profile_click_score: p.get(ActionName::ClientTweetClickProfile),
          vqv_score: p.get(ActionName::ClientTweetVideoQualityView),
          share_score: p.get(ActionName::ClientTweetShare),
          share_via_dm_score: p.get(ActionName::ClientTweetClickSendViaDirectMessage),
          share_via_copy_link_score: p.get(ActionName::ClientTweetShareViaCopyLink),
          dwell_score: p.get(ActionName::ClientTweetRecapDwelled),
          quote_score: p.get(ActionName::ServerTweetQuote),
          quoted_click_score: p.get(ActionName::ClientQuotedTweetClick),
          follow_author_score: p.get(ActionName::ClientTweetFollowAuthor),
          not_interested_score: p.get(ActionName::ClientTweetNotInterestedIn),
          block_author_score: p.get(ActionName::ClientTweetBlockAuthor),
          mute_author_score: p.get(ActionName::ClientTweetMuteAuthor),
          report_score: p.get(ActionName::ClientTweetReport),
          dwell_time: p.get_continuous(ContinuousActionName::DwellTime),
      }
  }
  ```
  - `ActionPredictions::get` and `get_continuous` use enum indices (e.g., `ActionName::ServerTweetFav as usize`) to fetch from internal maps.
- Sets `prediction_request_id` and `last_scored_at_ms` for traceability.
- If no sequence or prediction fails, candidates remain unchanged.
- Only updates `phoenix_scores`, `prediction_request_id`, and `last_scored_at_ms`.

This provides foundational ML outputs for downstream scorers. For Phoenix model details, see [Python Phoenix Overview](../python-phoenix-overview.md) and [RecSys Model](../recsys-model.md).

## WeightedScorer

The `WeightedScorer` combines Phoenix predictions into a single `weighted_score` using predefined weights, emphasizing positive engagements and penalizing negatives.

### Structure and Dependencies
- Defined in `home-mixer/scorers/weighted_scorer.rs`.
- Stateless empty struct `WeightedScorer`.
- Depends on `PostCandidate::phoenix_scores` (from PhoenixScorer).
- Uses weights from `crate::params` (e.g., `FAVORITE_WEIGHT = 1.0`, `BLOCK_AUTHOR_WEIGHT = -2.0`).
- Normalizes via `crate::util::score_normalizer::normalize_score`.

### Implementation
Implements `Scorer<ScoredPostsQuery, PostCandidate>`. Query-independent.

Key signature:
```rust
#[async_trait]
impl Scorer<ScoredPostsQuery, PostCandidate> for WeightedScorer {
    async fn score(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<Vec<PostCandidate>, String> {
        // Computes weighted_score for each, normalizes, and returns.
    }
    fn update(&self, candidate: &mut PostCandidate, scored: PostCandidate) {
        candidate.weighted_score = scored.weighted_score;
    }
}
```

### Score Computation
- For each candidate, invokes `compute_weighted_score`:
  ```rust
  fn compute_weighted_score(candidate: &PostCandidate) -> f64 {
      let s: &PhoenixScores = &candidate.phoenix_scores;
      let vqv_weight = Self::vqv_weight_eligibility(candidate);  // 0 if no video or duration < MIN_VIDEO_DURATION_MS
      let combined_score = Self::apply(s.favorite_score, p::FAVORITE_WEIGHT)
          + Self::apply(s.reply_score, p::REPLY_WEIGHT)
          + Self::apply(s.retweet_score, p::RETWEET_WEIGHT)
          + Self::apply(s.photo_expand_score, p::PHOTO_EXPAND_WEIGHT)
          + Self::apply(s.click_score, p::CLICK_WEIGHT)
          + Self::apply(s.profile_click_score, p::PROFILE_CLICK_WEIGHT)
          + Self::apply(s.vqv_score, vqv_weight)
          + Self::apply(s.share_score, p::SHARE_WEIGHT)
          + Self::apply(s.share_via_dm_score, p::SHARE_VIA_DM_WEIGHT)
          + Self::apply(s.share_via_copy_link_score, p::SHARE_VIA_COPY_LINK_WEIGHT)
          + Self::apply(s.dwell_score, p::DWELL_WEIGHT)
          + Self::apply(s.quote_score, p::QUOTE_WEIGHT)
          + Self::apply(s.quoted_click_score, p::QUOTED_CLICK_WEIGHT)
          + Self::apply(s.dwell_time, p::CONT_DWELL_TIME_WEIGHT)
          + Self::apply(s.follow_author_score, p::FOLLOW_AUTHOR_WEIGHT)
          + Self::apply(s.not_interested_score, p::NOT_INTERESTED_WEIGHT)
          + Self::apply(s.block_author_score, p::BLOCK_AUTHOR_WEIGHT)
          + Self::apply(s.mute_author_score, p::MUTE_AUTHOR_WEIGHT)
          + Self::apply(s.report_score, p::REPORT_WEIGHT);
      Self::offset_score(combined_score)
  }
  ```
  - `apply(score: Option<f64>, weight: f64) -> f64`: Returns `score * weight` if present, else 0.
  - `vqv_weight_eligibility`: Applies `VQV_WEIGHT` only for videos longer than `MIN_VIDEO_DURATION_MS`.
  - `offset_score`: Adds `NEGATIVE_SCORES_OFFSET` for negatives, scales by `WEIGHTS_SUM` and `NEGATIVE_WEIGHTS_SUM` to downrank disliked content.
- Normalizes the result and sets `weighted_score`.
- Only updates `weighted_score`.

This linear combination balances various engagement signals.

## AuthorDiversityScorer

The `AuthorDiversityScorer` promotes feed variety by attenuating scores for repeated authors.

### Structure and Dependencies
- Defined in `home-mixer/scorers/author_diversity_scorer.rs`.
- Struct with `decay_factor: f64` (e.g., 0.9) and `floor: f64` (e.g., 0.1) from params (`AUTHOR_DIVERSITY_DECAY`, `AUTHOR_DIVERSITY_FLOOR`).
- Depends on `PostCandidate::weighted_score` and `author_id`.
- Temporarily sorts candidates for computation but preserves original order.

### Implementation
Implements `Scorer<ScoredPostsQuery, PostCandidate>`. Monitored with stats.

Key signature:
```rust
#[async_trait]
impl Scorer<ScoredPostsQuery, PostCandidate> for AuthorDiversityScorer {
    async fn score(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<Vec<PostCandidate>, String> {
        // Sorts by weighted_score, applies multipliers per author, sets score.
    }
    fn update(&self, candidate: &mut PostCandidate, scored: PostCandidate) {
        candidate.score = scored.score;
    }
}
```

### Score Computation
- Sorts candidates descending by `weighted_score` (using `f64::NEG_INFINITY` for missing).
- Builds `HashMap<u64, usize>` for author occurrence counts.
- For each in sorted order, computes position among same-author posts and applies `multiplier`:
  ```rust
  fn multiplier(&self, position: usize) -> f64 {
      (1.0 - self.floor) * self.decay_factor.powf(position as f64) + self.floor
  }
  ```
  - First post per author: ~1.0 multiplier.
  - Subsequent: Exponential decay (e.g., 0.9^position adjusted by floor).
- Sets `score = weighted_score * multiplier` (None if missing).
- Outputs in original order.
- Only updates `score`.

This sets the initial `score` without dropping candidates, ensuring diversity.

## OONScorer

The `OONScorer` (Out-of-Network) downranks posts from unfollowed authors to prioritize in-network content.

### Structure and Dependencies
- Defined in `home-mixer/scorers/oon_scorer.rs`.
- Stateless empty struct `OONScorer`.
- Depends on `PostCandidate::score` (from AuthorDiversityScorer) and `in_network` (hydrated from sources like Thunder; see [Sources](../sources.md)).
- Uses `p::OON_WEIGHT_FACTOR` (e.g., 0.5 to penalize out-of-network).

### Implementation
Implements `Scorer<ScoredPostsQuery, PostCandidate>`.

Key signature:
```rust
#[async_trait]
impl Scorer<ScoredPostsQuery, PostCandidate> for OONScorer {
    async fn score(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<Vec<PostCandidate>, String> {
        // Applies factor if !in_network.
    }
    fn update(&self, candidate: &mut PostCandidate, scored: PostCandidate) {
        candidate.score = scored.score;
    }
}
```

### Score Computation
- For each candidate: If `in_network == Some(false)`, sets `score = score * p::OON_WEIGHT_FACTOR`.
- Otherwise, unchanged.
- Simple multiplicative adjustment.
- Only updates `score`.

This finalizes the score, favoring followed authors.

## Pipeline Combination Strategies

Scorers execute in fixed order in `PhoenixCandidatePipeline::scorers`:

1. **PhoenixScorer**: Async ML inference populates `phoenix_scores`.
2. **WeightedScorer**: Synchronous linear combo yields `weighted_score` (video-aware).
3. **AuthorDiversityScorer**: Sorts temporarily to compute per-author decay, sets initial `score`.
4. **OONScorer**: Applies network penalty to finalize `score`.

This sequence respects dependencies (e.g., Weighted requires Phoenix). The pipeline handles partial failures gracefully. Final scores feed into [Selectors](../selectors.md) for top-K selection in `HomeMixerServer::get_scored_posts` (see [Runners](../runners.md)).

For configuration of weights and params, see [Configuration](../configuration.md). For testing scorers, refer to [Testing Phoenix](../testing-phoenix.md).

:::tip Developer Notes
- Extend by implementing the trait and adding to the vector.
- Monitor with stats macros for performance.
- Ensure async safety for ML calls.
:::
