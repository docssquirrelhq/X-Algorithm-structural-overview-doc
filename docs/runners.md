---
title: Runners Module
sidebar_label: Runners
description: Overview of the runners.py module in Phoenix, including model runners, inference runners, and utility functions for recommendation systems.
---

# Runners Module

The `runners.py` module in Phoenix provides classes and utility functions for running recommendation models, particularly for ranking and retrieval tasks in a recommender system. It leverages JAX and Haiku for efficient, differentiable computations, enabling model initialization, forward passes, and inference workflows. This module is central to handling the encoding of users and candidates, as well as retrieving top-k recommendations.

Key components include:
- **Base classes** for shared logic in model and inference runners.
- **Specific runners** for ranking (`ModelRunner` and `RecsysInferenceRunner`) and retrieval (`RetrievalModelRunner` and `RecsysRetrievalInferenceRunner`).
- **Utility functions** for creating dummy data and example corpora.
- **Output structures** like `RankingOutput` and `RetrievalOutput` for structured results.

The module supports JAX transformations (via Haiku) for transforming functions into pure, stateless computations, which is crucial for parallelization and just-in-time (JIT) compilation on accelerators like GPUs/TPUs.

## Utility Functions

These functions help in creating synthetic data for model initialization and testing.

### `create_dummy_batch_from_config`

Creates a dummy `RecsysBatch` filled with zeros, used for model initialization without real data.

```python
def create_dummy_batch_from_config(
    hash_config: Any,
    history_len: int,
    num_candidates: int,
    num_actions: int,
    batch_size: int = 1,
) -> RecsysBatch:
```

**Arguments:**
- `hash_config`: Configuration with hash dimensions (e.g., `num_user_hashes`, `num_item_hashes`, `num_author_hashes`).
- `history_len`: Length of user history sequence.
- `num_candidates`: Number of candidate items.
- `num_actions`: Number of engagement action types (e.g., 19 for favorites, replies, etc.).
- `batch_size`: Size of the batch (default: 1).

**Returns:** A `RecsysBatch` with zero-initialized tensors for user hashes, history, candidates, etc.

This is essential for initializing models without loading real datasets.

### `create_dummy_embeddings_from_config`

Similar to the batch creator, but for `RecsysEmbeddings` with zero-initialized embedding tensors.

```python
def create_dummy_embeddings_from_config(
    hash_config: Any,
    emb_size: int,
    history_len: int,
    num_candidates: int,
    batch_size: int = 1,
) -> RecsysEmbeddings:
```

**Arguments:**
- Same as above, plus `emb_size`: Embedding dimension size.

**Returns:** A `RecsysEmbeddings` object with zero tensors for user, history, and candidate embeddings.

### `create_example_corpus`

Generates a synthetic corpus of embeddings for testing retrieval, normalized to unit length.

```python
def create_example_corpus(
    corpus_size: int,
    emb_size: int,
    seed: int = 123,
) -> Tuple[jax.Array, jax.Array]:
```

**Arguments:**
- `corpus_size`: Number of items in the corpus.
- `emb_size`: Embedding dimension.
- `seed`: Random seed for reproducibility.

**Returns:** Tuple of `(corpus_embeddings: [N, D], corpus_post_ids: [N])`, where embeddings are L2-normalized.

This utility is useful for end-to-end testing of retrieval without a real corpus.

### ACTIONS List

A predefined list of engagement action types used in ranking:

```python
ACTIONS: List[str] = [
    "favorite_score", "reply_score", "repost_score", "photo_expand_score",
    "click_score", "profile_click_score", "vqv_score", "share_score",
    # ... (full list of 19 actions)
]
```

This defines the output probabilities in `RankingOutput`.

## Base Classes

### `BaseModelRunner`

Abstract base class for model runners, handling initialization and forward function creation.

```python
@dataclass
class BaseModelRunner(ABC):
    bs_per_device: float = 2.0
    rng_seed: int = 42
```

**Key Methods:**
- `model` (property): Returns the model configuration.
- `make_forward_fn` (abstractmethod): Creates the Haiku-transformed forward function.
- `initialize`: Sets up the model, computes batch size based on local devices, and prepares the forward function.

Uses JAX for device-aware batch sizing and logging.

### `BaseInferenceRunner`

Abstract base for inference runners, providing dummy data creation.

```python
@dataclass
class BaseInferenceRunner(ABC):
    name: str
```

**Key Methods:**
- `runner` (property): Returns the underlying `BaseModelRunner`.
- `create_dummy_batch` / `create_dummy_embeddings`: Convenience wrappers for utilities.
- `initialize` (abstractmethod): Subclass-specific setup.

## Ranking Runners

These handle scoring and ranking candidates based on user history and engagements.

### `ModelRunner` (inherits from `BaseModelRunner`)

Runner for the Phoenix ranking model (`PhoenixModelConfig`).

```python
@dataclass
class ModelRunner(BaseModelRunner):
    _model: PhoenixModelConfig = None
```

**Key Methods:**
- `make_forward_fn`: Transforms the model's forward pass using `hk.transform`.
- `init`: Initializes parameters with a given RNG key and dummy data.
- `load_or_init`: Loads or initializes the training state.

The forward pass computes `RecsysModelOutput` (logits for actions).

### `RecsysInferenceRunner` (inherits from `BaseInferenceRunner`)

Inference interface for ranking.

```python
@dataclass
class RecsysInferenceRunner(BaseInferenceRunner):
    _runner: ModelRunner
```

**Key Methods:**
- `initialize`: Sets up parameters, caches the model, and creates a Haiku-transformed ranking function (`hk_rank_candidates`).
- `rank`: Applies the ranking function to get `RankingOutput`.

**Inference Flow for Ranking:**
1. Prepare `RecsysBatch` (user history, candidates) and `RecsysEmbeddings` (pre-looked-up).
2. Forward pass: Compute logits via the model.
3. Apply sigmoid to get probabilities for each action (e.g., `p_favorite_score`).
4. Use primary score (favorite) for sorting to get `ranked_indices`.
5. Return `RankingOutput` with scores and indices.

JAX transformations ensure the function is pure and compilable. Example:

```python
output = runner.rank(batch, embeddings)
top_candidates = output.ranked_indices[:, :10]  # Top-10 per user
```

### `RankingOutput`

Structured output for ranking results.

```python
class RankingOutput(NamedTuple):
    scores: jax.Array  # [B, C, num_actions]
    ranked_indices: jax.Array  # [B, C]
    p_favorite_score: jax.Array  # And one for each action...
```

## Retrieval Runners

These focus on encoding users/candidates and retrieving top-k from a corpus using similarity.

### `RetrievalModelRunner` (inherits from `BaseModelRunner`)

Runner for the Phoenix retrieval model (`PhoenixRetrievalModelConfig`).

```python
@dataclass
class RetrievalModelRunner(BaseModelRunner):
    _model: PhoenixRetrievalModelConfig = None
```

**Key Methods:**
- `make_forward_fn`: Transforms the retrieval forward pass, including user/candidate representation building.
- `init` / `load_or_init`: Similar to ranking, but includes corpus embeddings and `top_k`.

### `RecsysRetrievalInferenceRunner` (inherits from `BaseInferenceRunner`)

Inference interface for retrieval, supporting encoding and top-k retrieval.

```python
@dataclass
class RecsysRetrievalInferenceRunner(BaseInferenceRunner):
    _runner: RetrievalModelRunner = None
    corpus_embeddings: jax.Array | None = None
    corpus_post_ids: jax.Array | None = None
```

**Key Methods:**
- `initialize`: Sets up parameters and separate Haiku-transformed functions for encoding user, encoding candidates, and full retrieval.
- `encode_user`: Computes user representation `[B, D]` from batch and embeddings.
- `encode_candidates`: Computes candidate representations `[B, C, D]`.
- `set_corpus`: Sets pre-computed corpus embeddings and IDs for reuse.
- `retrieve`: Performs top-k retrieval using dot-product similarity (assumed in model).

**Inference Flows:**

1. **Encoding Users:**
   - Input: `RecsysBatch` (user + history) and `RecsysEmbeddings`.
   - Process: Model builds pooled/aggregated representation via `build_user_representation`.
   - Output: User vectors `[B, D]`.
   - JAX Flow: `hk.transform` on `hk_encode_user` ensures stateless computation.

2. **Encoding Candidates:**
   - Input: `RecsysBatch` (candidates only) and `RecsysEmbeddings`.
   - Process: Model builds representations via `build_candidate_representation`.
   - Output: Candidate vectors `[B, C, D]`.
   - Useful for offline candidate generation.

3. **Retrieving Top-K:**
   - Setup: Call `set_corpus` with `[N, D]` embeddings (e.g., from `create_example_corpus`).
   - Input: User batch/embeddings, `top_k` (default 100).
   - Process:
     - Encode users to get `[B, D]`.
     - Compute similarities: Dot product with corpus `[B, N]`.
     - Select top-k indices and scores per user.
   - Output: `RetrievalOutput` with `user_representation`, `top_k_indices` `[B, K]`, `top_k_scores` `[B, K]`.
   - JAX Flow: Full `hk_retrieve` transformation handles the end-to-end retrieval, compilable for batch efficiency.

Example usage:

```python
# Initialize
runner = RecsysRetrievalInferenceRunner(model_runner, "phoenix_retrieval")
runner.initialize()

# Set corpus
corpus_emb, post_ids = create_example_corpus(10000, 128)
runner.set_corpus(corpus_emb, post_ids)

# Retrieve
output = runner.retrieve(user_batch, user_embeddings, top_k=50)
top_posts = post_ids[output.top_k_indices]  # Map back to IDs
```

JAX transformations (`hk.without_apply_rng(hk.transform(...))`) make these functions parameter-applicable without RNG, ideal for inference.

## Integration Notes

- Runners assume pre-computed embeddings (e.g., from embedding tables).
- Use with JAX for sharding across devices; batch sizes auto-scale with local GPUs.
- For full pipeline, combine with [Recsys Model](../recsys-model.md) and [Retrieval Model](../recsys-retrieval-model.md).
- Testing: Use utilities for smoke tests; see [Testing Phoenix](../testing-phoenix.md) for more.

This module enables scalable, JAX-accelerated inference for Phoenix's two-tower retrieval and ranking architecture.