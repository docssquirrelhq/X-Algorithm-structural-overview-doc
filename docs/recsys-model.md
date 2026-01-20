---
title: Recommendation System Model (recsys_model.py)
slug: recsys-model
description: Detailed overview of the Phoenix recommendation system's ranking model, including data structures, embedding reductions, input building, transformer forward pass, and output logits.
sidebar_label: Recsys Model
sidebar_position: 9
---

# Recommendation System Model (recsys_model.py)

The `recsys_model.py` file in the Phoenix module implements a transformer-based ranking model for recommendation systems. This model processes user history, candidate posts, and contextual features to generate logits for potential user actions (e.g., likes, reposts) on candidate items. It emphasizes modularity by separating raw input features (hashes, actions, product surfaces) from pre-computed embeddings, using hash-based lookups for efficiency in large-scale systems.

The model is built using Haiku (for neural network modules) and JAX (for array operations), integrating a custom `Transformer` from the `grok` library. Key design principles include:
- **Hash-based Embeddings**: Multiple hashes per entity (user, post, author) to handle collisions and improve representation robustness.
- **Sequence Modeling**: Treats the input as a sequence: [user] + [history posts] + [candidates], with causal attention for history and cross-attention for ranking.
- **Efficiency**: Supports bfloat16 precision, padding masks for variable lengths, and learned projections for fusing multi-hash embeddings.

This documentation focuses on the core components: data structures, embedding reduction functions, input building, the transformer forward pass, and output logits. For an overview of the broader Phoenix system, see [Python Phoenix Overview](/docs/python-phoenix-overview.md). Related models like retrieval are covered in [Recsys Retrieval Model](/docs/recsys-retrieval-model.md).

## Data Structures

### RecsysBatch
`RecsysBatch` is a `NamedTuple` representing the input batch of raw features (excluding embeddings, which are passed separately). It contains hashes for entity lookups, multi-hot action vectors, and categorical product surface indices.

- **Fields**:
  - `user_hashes`: [B, num_user_hashes] – Integer hashes for the user (0 for padding).
  - `history_post_hashes`: [B, S, num_item_hashes] – Hashes for historical posts (S = history sequence length, e.g., 128).
  - `history_author_hashes`: [B, S, num_author_hashes] – Hashes for authors of historical posts.
  - `history_actions`: [B, S, num_actions] – Multi-hot vectors indicating user engagements (e.g., [1, 0, 1] for like and repost).
  - `history_product_surface`: [B, S] – Categorical indices for UI contexts (e.g., home feed vs. search; vocab size = 16).
  - `candidate_post_hashes`: [B, C, num_item_hashes] – Hashes for candidate posts (C = candidate sequence length, e.g., 32).
  - `candidate_author_hashes`: [B, C, num_author_hashes] – Hashes for candidate authors.
  - `candidate_product_surface`: [B, C] – Categorical indices for candidate contexts.

**Purpose**: Encapsulates sparse, hash-based inputs for efficient processing. Embeddings are pre-looked up externally to decouple retrieval from computation.

### RecsysEmbeddings
`RecsysEmbeddings` is a dataclass container for pre-computed embeddings from hash tables. These are looked up before the model forward pass.

- **Fields**:
  - `user_embeddings`: [B, num_user_hashes, D] – User embeddings (D = embedding dimension).
  - `history_post_embeddings`: [B, S, num_item_hashes, D] – Embeddings for historical posts.
  - `candidate_post_embeddings`: [B, C, num_item_hashes, D] – Embeddings for candidate posts.
  - `history_author_embeddings`: [B, S, num_author_hashes, D] – Author embeddings for history.
  - `candidate_author_embeddings`: [B, C, num_author_hashes, D] – Author embeddings for candidates.

**Purpose**: Holds dense representations fetched via hashes, allowing the model to focus on fusion and transformation rather than lookup.

### RecsysModelOutput
A simple `NamedTuple` for model outputs.

- **Fields**:
  - `logits`: [B, C, num_actions] – Unnormalized scores for actions on each candidate.

**Purpose**: Provides action-specific logits for downstream ranking (e.g., apply sigmoid for probabilities).

## Configuration: PhoenixModelConfig

`PhoenixModelConfig` is a dataclass defining hyperparameters for the model.

- **Key Fields**:
  - `model`: A `TransformerConfig` for the core transformer architecture.
  - `emb_size`: Embedding dimension (D).
  - `num_actions`: Number of possible user actions (e.g., like, repost, reply).
  - `history_seq_len`: Maximum history length (default: 128).
  - `candidate_seq_len`: Maximum candidates (default: 32).
  - `hash_config`: `HashConfig` for hash counts (e.g., `num_user_hashes=2`).
  - `product_surface_vocab_size`: Vocab size for UI contexts (default: 16).
  - `fprop_dtype`: Computation dtype (default: `jnp.bfloat16` for efficiency).

- **Methods**:
  - `__post_init__`: Ensures `hash_config` is initialized.
  - `initialize()`: Marks config as ready.
  - `make()`: Creates a `PhoenixModel` instance, with logging if uninitialized.

**Purpose**: Centralizes configuration, enabling easy experimentation. The `make` method integrates with the transformer factory.

## Embedding Reduction Functions

These functions fuse multi-hash embeddings into single representations using learned linear projections. They handle padding (hash 0 = invalid) and are invoked during input building. All use `hk.initializers.VarianceScaling` for stable initialization.

### block_user_reduce
Reduces multiple user hash embeddings into a single user vector.

- **Inputs**:
  - `user_hashes`: [B, num_user_hashes]
  - `user_embeddings`: [B, num_user_hashes, D]
  - `num_user_hashes`, `emb_size` (D), `embed_init_scale` (default: 1.0)

- **Process**:
  1. Reshape embeddings to [B, 1, num_user_hashes * D].
  2. Project via learned matrix `proj_mat_1` [num_user_hashes * D, D] to [B, 1, D].
  3. Create padding mask [B, 1] based on non-zero first hash.

- **Outputs**: User embedding [B, 1, D] and mask [B, 1].

**Example Code Snippet**:
```python
user_embedding, user_padding_mask = block_user_reduce(
    batch.user_hashes,
    recsys_embeddings.user_embeddings,
    hash_config.num_user_hashes,
    config.emb_size
)
```

**Explanation**: Concatenates and projects to learn a robust user representation from multiple hashes, masking invalid users.

### block_history_reduce
Fuses history post, author, action, and product surface embeddings into a sequence.

- **Inputs**:
  - `history_post_hashes`: [B, S, num_item_hashes]
  - `history_post_embeddings`: [B, S, num_item_hashes, D]
  - `history_author_embeddings`: [B, S, num_author_hashes, D]
  - `history_product_surface_embeddings`: [B, S, D] (pre-computed)
  - `history_actions_embeddings`: [B, S, D] (pre-computed)
  - `num_item_hashes`, `num_author_hashes`, `embed_init_scale`

- **Process**:
  1. Reshape post/author embeddings to [B, S, num_hashes * D].
  2. Concatenate with actions and surfaces to [B, S, total_dim].
  3. Project via `proj_mat_3` [total_dim, D] to [B, S, D].
  4. Padding mask [B, S] from non-zero post hashes.

- **Outputs**: History embeddings [B, S, D] and mask [B, S].

**Explanation**: Integrates contextual history (posts + engagements + UI) into a unified sequence embedding, enabling the transformer to model temporal user behavior.

### block_candidate_reduce
Similar to `block_history_reduce`, but for candidates (no actions).

- **Inputs**: Analogous to history, but for candidates [B, C, ...] and without actions.
- **Process**: Concatenate post/author/surface, project via `proj_mat_2`, mask from post hashes.
- **Outputs**: Candidate embeddings [B, C, D] and mask [B, C].

**Explanation**: Prepares candidate representations for ranking, focusing on item features and context without historical actions.

## Model Class: PhoenixModel

`PhoenixModel` is the main `hk.Module` implementing the ranking logic.

### Helper Methods for Embeddings and Lookup

- **`_get_action_embeddings(actions: [B, S, num_actions]) -> [B, S, D]`**:
  - Converts multi-hot actions to signed embeddings (2 * actions - 1) via projection matrix [num_actions, D].
  - Masks zero-action entries.
  - **Explanation**: Handles sparse, multi-label actions by learning dense representations; signing captures positive/negative signals.

- **`_single_hot_to_embeddings(input: [B, S], vocab_size, emb_size, name)` -> [B, S, D]`**:
  - One-hot encodes input and looks up from embedding table [vocab_size, D].
  - **Explanation**: Standard categorical embedding for product surfaces (e.g., 16 UI types).

- **`_get_unembedding() -> [D, num_actions]`**:
  - Returns learned matrix for projecting to action logits.
  - **Explanation**: Decodes transformer outputs back to per-action scores.

### Input Building: build_inputs

Orchestrates the full input preparation.

- **Inputs**: `batch: RecsysBatch`, `recsys_embeddings: RecsysEmbeddings`
- **Process**:
  1. Compute product surface embeddings via `_single_hot_to_embeddings`.
  2. Compute action embeddings via `_get_action_embeddings` (for history only).
  3. Apply reduction functions: `block_user_reduce`, `block_history_reduce`, `block_candidate_reduce`.
  4. Concatenate: embeddings [B, 1 + S + C, D], padding_mask [B, 1 + S + C].
  5. Compute `candidate_start_offset` = 1 + S (position of candidates in sequence).

- **Outputs**: Embeddings, mask, offset (all cast to `fprop_dtype`).

**Explanation**: Constructs the transformer sequence by fusing all features. Embeddings are normalized implicitly via projections; explicit normalization occurs post-transformer.

### Transformer Forward Pass: __call__

The core forward method for ranking.

- **Inputs**: `batch`, `recsys_embeddings`
- **Process**:
  1. Call `build_inputs` to get embeddings, mask, offset.
  2. Pass to `self.model` (Transformer): Encodes the sequence with padding mask and `candidate_start_offset` (likely for causal attention on history + bidirectional on candidates).
  3. Apply `layer_norm` to output embeddings [B, 1 + S + C, D].
  4. Extract candidate embeddings [B, C, D].
  5. Project via `_get_unembedding` to logits [B, C, num_actions].

- **Output**: `RecsysModelOutput(logits=logits)` (in `fprop_dtype`).

**Explanation**: The transformer models user-history context and candidate interactions. Layer normalization stabilizes representations before unembedding. Logits enable action-specific ranking (e.g., sort candidates by favorite logit).

## Overall Model Flow

1. **Embedding Lookup (External)**: Use hashes from `RecsysBatch` to fetch `RecsysEmbeddings`.
2. **Input Fusion**: `build_inputs` reduces and concatenates into sequence.
3. **Transformer Encoding**: Processes sequence with masks; candidates attend to user + history.
4. **Output Generation**: Normalize, extract candidates, unembed to action logits.
5. **Ranking**: Use logits for candidate re-ranking (e.g., in [Runners](/docs/runners.md)).

:::tip
For training, optimize via cross-entropy on action labels. See [Data Flow](/docs/data-flow.md) for integration with pipelines.
:::

This model powers fine-grained ranking in Phoenix. For configuration details, refer to [Configuration](/docs/configuration.md).