---
title: Selectors Module
slug: selectors
description: Documentation on the selectors module in Rust's home-mixer, focusing on TopKScoreSelector and its role in the candidate pipeline.
sidebar_label: Selectors
sidebar_position: 4
---

# Selectors Module

The selectors module in the home-mixer component of the Rust codebase handles the final ranking and truncation of candidates in the recommendation pipeline. It defines a mechanism to sort candidates based on their scores and select the top-k highest-scoring ones, ensuring the output feed is limited to a configurable number of items. This module is part of the broader candidate pipeline framework, which orchestrates stages like sourcing, hydration, filtering, scoring, and selection for generating scored posts.

The module is located in `home-mixer/selectors/` and primarily exposes the `TopKScoreSelector` as the default implementation for selecting candidates. It integrates with the `ScoredPostsQuery` (query type) and `PostCandidate` (candidate type) used throughout the pipeline. For more on the overall pipeline, see the [Candidate Pipeline](../candidate-pipeline.md) documentation.

## The Selector Trait

The core abstraction is the `Selector` trait, defined in the candidate pipeline framework (`candidate-pipeline/selector.rs`). This trait outlines how to score individual candidates, sort them, and optionally limit the output size. It provides a default implementation for the `select` method, which sorts candidates by descending score and truncates to the specified size if provided.

Key methods in the trait:

- `score(&self, candidate: &C) -> f64`: Extracts or computes a score for a single candidate. This is crucial for sorting, as higher scores indicate higher relevance.
- `sort(&self, candidates: Vec<C>) -> Vec<C>`: Sorts candidates in descending order of their scores using `partial_cmp`. If scores are equal, the order is preserved as `Ordering::Equal`.
- `size(&self) -> Option<usize>`: Returns an optional limit for the number of candidates to keep after sorting. If `None`, no truncation occurs.
- `select(&self, query: &Q, candidates: Vec<C>) -> Vec<C>`: Default behavior sorts the candidates and truncates if a size is specified. It can be overridden for custom logic.
- `enable(&self, query: &Q) -> bool`: Determines if the selector should run for a given query (defaults to `true`).
- `name(&self) -> &'static str`: Provides a stable name for logging and metrics, derived from the type name.

Here's the trait definition:

```rust
pub trait Selector<Q, C>: Send + Sync
where
    Q: Clone + Send + Sync + 'static,
    C: Clone + Send + Sync + 'static,
{
    /// Default selection: sort and truncate based on provided configs
    fn select(&self, _query: &Q, candidates: Vec<C>) -> Vec<C> {
        let mut sorted = self.sort(candidates);
        if let Some(limit) = self.size() {
            sorted.truncate(limit);
        }
        sorted
    }

    /// Decide if this selector should run for the given query
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    /// Extract the score from a candidate to use for sorting.
    fn score(&self, candidate: &C) -> f64;

    /// Sort candidates by their scores in descending order.
    fn sort(&self, candidates: Vec<C>) -> Vec<C> {
        let mut sorted = candidates;
        sorted.sort_by(|a, b| {
            self.score(b)
                .partial_cmp(&self.score(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    /// Optionally provide a size to select. Defaults to no truncation if not overridden.
    fn size(&self) -> Option<usize> {
        None
    }

    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }
}
```

This trait ensures selectors are thread-safe (`Send + Sync`) and generic over query (`Q`) and candidate (`C`) types, making it reusable across pipelines.

## TopKScoreSelector Implementation

The `TopKScoreSelector` is the concrete implementation in `home-mixer/selectors/top_k_score_selector.rs`. It uses the candidate's pre-computed score (from earlier scoring stages like `PhoenixScorer` or `WeightedScorer`) for sorting. If no score is available, it defaults to negative infinity to deprioritize the candidate.

- `score(&self, candidate: &PostCandidate) -> f64`: Directly returns `candidate.score.unwrap_or(f64::NEG_INFINITY)`. This relies on scores populated by prior scorers in the pipeline (e.g., weighted engagement predictions).
- `size(&self) -> Option<usize>`: Returns `Some(params::TOP_K_CANDIDATES_TO_SELECT)`, enforcing a fixed top-k limit configured via module parameters.

Code snippet for `TopKScoreSelector`:

```rust
use crate::candidate_pipeline::candidate::PostCandidate;
use crate::candidate_pipeline::query::ScoredPostsQuery;
use crate::params;
use xai_candidate_pipeline::selector::Selector;

pub struct TopKScoreSelector;

impl Selector<ScoredPostsQuery, PostCandidate> for TopKScoreSelector {
    fn score(&self, candidate: &PostCandidate) -> f64 {
        candidate.score.unwrap_or(f64::NEG_INFINITY)
    }
    fn size(&self) -> Option<usize> {
        Some(params::TOP_K_CANDIDATES_TO_SELECT)
    }
}
```

The module re-exports this via `home-mixer/selectors/mod.rs`:

```rust
mod top_k_score_selector;

pub use top_k_score_selector::TopKScoreSelector;
```

## Scoring and Top-K Selection

Scoring in the selector context is lightweight: it doesn't compute new scores but extracts them from candidates hydrated and scored in prior pipeline stages. For example:

- Candidates enter the selection stage after scorers like `WeightedScorer` (which combines ML predictions into a `weighted_score`) or `AuthorDiversityScorer` (which adjusts for author repetition and sets a final `score`). See the [Scorers](../scorers.md) documentation for details.
- The `TopKScoreSelector` uses the final `score` field on `PostCandidate` for comparison. This score typically incorporates factors like predicted likes, replies, retweets, and diversity adjustments.
- If a candidate lacks a score (e.g., due to filtering or errors), it's assigned `f64::NEG_INFINITY`, ensuring it ranks last.

The top-k selection is a two-step process in the default `select` implementation:

1. **Sorting**: Candidates are sorted in descending score order using the `sort` method. Ties are handled gracefully with `partial_cmp` and `Ordering::Equal`.
2. **Truncation**: If `size()` returns `Some(k)`, the list is truncated to the first `k` items via `Vec::truncate`. This ensures the pipeline outputs at most `k` candidates, preventing overload in downstream stages like post-selection filtering or response serialization.

In the broader pipeline (`candidate-pipeline/candidate_pipeline.rs`), selection happens after scoring:

```rust
let scored_candidates = self.score(&hydrated_query, kept_candidates).await;

let selected_candidates = self.select(&hydrated_query, scored_candidates);
```

Here, `self.select` invokes the trait's default `select` method on the configured selector (e.g., `TopKScoreSelector`), sorting by the extracted scores.

## Role in the Candidate Pipeline

The selector is the penultimate stage in the candidate pipeline:

- **Pre-Selection**: Sources fetch raw candidates (e.g., from `ThunderSource`; see [Sources](../sources.md)), hydrators enrich them, filters remove ineligible ones (see [Filters](../filters.md)), and scorers assign scores (e.g., `OONScorer` for in-network weighting, `AuthorDiversityScorer` for diversity).
- **Selection**: `TopKScoreSelector` sorts by score and truncates to top-k, producing `selected_candidates`.
- **Post-Selection**: Additional hydrators/filters (e.g., visibility filtering) refine the output, and side effects (e.g., caching) run last.
- **Output**: The pipeline returns a `PipelineResult` with `selected_candidates`, which the server (`server.rs`) maps to gRPC responses (`ScoredPostsResponse`).

In `phoenix_candidate_pipeline.rs` (the main pipeline builder), the selector is instantiated as:

```rust
let selector = TopKScoreSelector;
```

This is then used in the `CandidatePipeline` trait implementation:

```rust
fn selector(&self) -> &dyn Selector<ScoredPostsQuery, PostCandidate> {
    &self.selector
}
```

During execution (`execute` method), after scoring, it calls `self.select(&hydrated_query, scored_candidates)`, applying the top-k logic. The final selected candidates are further processed before being returned in `PipelineResult`.

This ensures efficient, score-driven ranking while respecting configurable limits. For an overview of the Rust home-mixer, see [Rust Home-Mixer Overview](../rust-home-mixer-overview.md).

## Configuration Parameters

The top-k size is controlled by `params::TOP_K_CANDIDATES_TO_SELECT`, a constant in the `params` module (imported as `crate::params`). This parameter is hardcoded or configurable at build-time (e.g., via environment or config files). It directly influences the `size()` return value in `TopKScoreSelector`, limiting the feed to high-relevance items.

In the pipeline's `result_size(&self) -> usize`, it uses `params::RESULT_SIZE` for final truncation after post-selection, but selection itself uses `TOP_K_CANDIDATES_TO_SELECT` to pre-limit before expensive post-selection steps. For more on configuration, see [Configuration](../configuration.md).

:::tip
This design balances performance (early truncation) with quality (score-based prioritization), making the system scalable for real-time feeds.
:::
