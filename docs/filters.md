---
title: Filters Module in Rust Home-Mixer
slug: filters
description: Overview of the filters module in Rust's home-mixer, with a focus on DedupConversationFilter for deduplicating conversation trees, the Filter trait, FilterResult, and pipeline integration.
sidebar_label: Filters
sidebar_position: 6
---

# Filters Module in Rust Home-Mixer

The filters module in the home-mixer component is a collection of Rust modules that implement candidate filtering logic for post recommendations. It is defined in `home-mixer/filters/mod.rs`, which re-exports various filter implementations for use in the candidate pipeline. The module includes filters such as:

- `age_filter`: Removes posts older than a configurable duration based on tweet creation time.
- `author_socialgraph_filter`: Excludes posts from authors blocked or muted by the viewer.
- `core_data_hydration_filter`: Drops candidates lacking essential data like author ID or non-empty tweet text.
- `dedup_conversation_filter`: Deduplicates posts within conversation branches (detailed below).
- `drop_duplicates_filter`: Removes exact duplicates based on tweet IDs.
- `ineligible_subscription_filter`: Filters out subscription-only posts from unsubscribed authors.
- `muted_keyword_filter`: Excludes posts matching user-muted keywords using tokenization and matching.
- `previously_seen_posts_filter`: Removes posts likely seen before using seen IDs and Bloom filters.
- `previously_served_posts_filter`: Drops posts already served in prior requests (enabled only for bottom requests).
- `retweet_deduplication_filter`: Deduplicates retweets by keeping the first occurrence of a tweet (original or retweet).
- `self_tweet_filter`: Removes posts authored by the viewer themselves.
- `vf_filter`: Applies visibility filtering based on safety or other reasons, dropping posts that should be hidden.

These filters operate on `PostCandidate` structs within the `ScoredPostsQuery` context, processing vectors of candidates sequentially to partition them into kept and removed sets. For more on the overall candidate pipeline, see the [Candidate Pipeline](../candidate-pipeline.md) documentation.

## DedupConversationFilter: Logic for Deduplicating Using Ancestors and Tweet IDs

The `DedupConversationFilter` is designed to eliminate redundancy in conversation threads by retaining only the highest-scored candidate per conversation branch. It treats conversations as trees, where each branch is identified by a unique conversation ID derived from the post's ancestors and tweet ID. This prevents showing multiple similar posts from the same thread, prioritizing quality (via score) over quantity.

### Key Logic

1. **Conversation ID Calculation**:
   - For a given `PostCandidate`, the conversation ID is computed as the minimum ID among its `ancestors` (a vector of `u64` representing parent/reply chain IDs) or falls back to the candidate's own `tweet_id` (cast to `u64`) if no ancestors exist.
   - This uses `get_conversation_id(&candidate)`, which finds `ancestors.iter().copied().min().unwrap_or(candidate.tweet_id as u64)`. The minimum ensures all posts in the same root-level branch share the same ID, effectively grouping thread variants.

2. **Deduplication Process**:
   - Initializes a `HashMap<u64, (usize, f64)>` (`best_per_convo`) to track the index in the `kept` vector and the best score for each conversation ID.
   - Iterates over input candidates:
     - Extracts the conversation ID and the candidate's score (defaults to 0.0 if unset).
     - If the conversation ID is new (not in the map):
       - Adds the candidate to `kept` and records its index and score in the map.
     - If already seen:
       - Compares the new score to the stored best score.
       - If the new score is higher, replaces the existing candidate in `kept` (using `std::mem::replace`) with the new one, adds the old one to `removed`, and updates the score.
       - Otherwise, adds the new candidate to `removed`.
   - This ensures only one representative per conversation branch survives, favoring the highest score. Scores are from `candidate.score.unwrap_or(0.0)`.

3. **Handling Edge Cases**:
   - Original posts (no ancestors) use their own tweet ID as the conversation ID.
   - Replies or quotes inherit the min ancestor ID, grouping them under the thread root.
   - No explicit handling for empty ancestors; falls back to tweet ID.
   - The filter assumes candidates are pre-scored; it relies on existing scores for comparison.

This logic promotes diverse, high-quality thread representation without altering scores or adding metadata.

Here's a simplified Rust code snippet illustrating the core deduplication logic:

```rust
use std::collections::HashMap;

fn dedup_conversation(candidates: Vec<PostCandidate>) -> (Vec<PostCandidate>, Vec<PostCandidate>) {
    let mut kept = Vec::new();
    let mut removed = Vec::new();
    let mut best_per_convo: HashMap<u64, (usize, f64)> = HashMap::new();

    for (idx, candidate) in candidates.into_iter().enumerate() {
        let convo_id = get_conversation_id(&candidate);
        let score = candidate.score.unwrap_or(0.0);

        match best_per_convo.get(&convo_id) {
            None => {
                let kept_idx = kept.len();
                kept.push(candidate);
                best_per_convo.insert(convo_id, (kept_idx, score));
            }
            Some(&(kept_idx, best_score)) => {
                if score > best_score {
                    let old_candidate = std::mem::replace(&mut kept[kept_idx], candidate);
                    removed.push(old_candidate);
                    best_per_convo.insert(convo_id, (kept_idx, score));
                } else {
                    removed.push(candidate);
                }
            }
        }
    }
    (kept, removed)
}

fn get_conversation_id(candidate: &PostCandidate) -> u64 {
    candidate
        .ancestors
        .iter()
        .copied()
        .min()
        .unwrap_or(candidate.tweet_id as u64)
}
```

:::note
The actual implementation is async and integrates with the `Filter` trait, but this captures the essence of the deduplication algorithm.
:::

## Filter Trait Implementation

The `DedupConversationFilter` implements the `Filter<ScoredPostsQuery, PostCandidate>` trait from `xai_candidate_pipeline::filter`. This trait defines a standardized interface for all filters in the pipeline:

### Core Methods

- `async fn filter(&self, query: &ScoredPostsQuery, candidates: Vec<PostCandidate>) -> Result<FilterResult<PostCandidate>, String>`: The main entry point. Processes candidates and returns a result. For `DedupConversationFilter`, it ignores the query (no query-specific logic) and applies the deduplication as described, producing `FilterResult { kept, removed }`. Errors are strings (e.g., for unexpected failures, though none are explicitly thrown here).
- `fn enable(&self, query: &ScoredPostsQuery) -> bool`: Defaults to `true`, so it always runs unless overridden.
- `fn name(&self) -> &'static str`: Returns a stable name like "DedupConversationFilter" for logging/metrics.

### Trait Bounds

- Requires `Send + Sync + Any` for thread-safety and dynamic dispatch.
- Uses `tonic::async_trait` for async compatibility.
- Generic over query `Q` (`ScoredPostsQuery`) and candidate `C` (`PostCandidate`), both requiring `Clone + Send + Sync + 'static`.

The implementation is boxed (`Box<dyn Filter<Q, C>>`) for use in pipeline collections, allowing heterogeneous filter lists. For details on the broader pipeline, refer to the [Candidate Pipeline](../candidate-pipeline.md) page.

## FilterResult Struct

`FilterResult<C>` is a simple container from `xai_candidate_pipeline::filter`:

```rust
pub struct FilterResult<C> {
    pub kept: Vec<C>,    // Candidates that pass the filter and proceed.
    pub removed: Vec<C>, // Candidates excluded by this filter.
}
```

- Each filter produces one, partitioning the input `Vec<C>`.
- `kept` continues to the next stage; `removed` is collected across filters for logging but discarded from further processing.
- No mutations to candidates; only selection.

This struct enables transparent tracking of filtering decisions for debugging and metrics.

## Placement in the Pipeline

Filters, including `DedupConversationFilter`, integrate into the `CandidatePipeline` (from `xai_candidate_pipeline::candidate_pipeline`) via the `CandidatePipeline` trait. The pipeline executes stages sequentially:

1. **Query Hydration**: Enrich the `ScoredPostsQuery` (e.g., fetch user features). See [Sources](../sources.md) for related data fetching.
2. **Candidate Fetching**: Retrieve raw `PostCandidate`s from sources.
3. **Hydration**: Add features (e.g., visibility, media) to candidates.
4. **Filtering (Pre-Scoring)**: Run main filters like `DedupConversationFilter` here via `self.filter(&hydrated_query, hydrated_candidates.clone()).await`.
   - Filters execute sequentially: For each enabled filter, call `filter` on the current `kept` set, accumulate `removed` across all, and log counts (e.g., "kept X, removed Y").
   - `DedupConversationFilter` fits here as it relies on scores (pre-scoring) and ancestors (hydrated), reducing redundancy before expensive scoring.
   - If a filter fails, it logs an error and reverts to the backup (previous candidates).
5. **Scoring**: Apply scorers to remaining kept candidates. See [Scorers](../scorers.md).
6. **Selection**: Sort/truncate (e.g., by score). See [Selectors](../selectors.md).
7. **Post-Selection Hydration**: Optional re-hydration.
8. **Post-Selection Filtering**: A second filter pass (e.g., for final checks), accumulating more removals.
9. **Side Effects**: Async logging/metrics.
10. **Truncation**: Limit to `result_size()`.

`DedupConversationFilter` is in the main `filters()` list (pre-scoring), not post-selection, to efficiently prune early. The pipeline returns a `PipelineResult` with `filtered_candidates` (all removed across filters) for debugging. In `HomeMixerServer`, filtered results feed into the gRPC response. For an overview of the Rust home-mixer, see [Rust Home-Mixer Overview](../rust-home-mixer-overview.md).