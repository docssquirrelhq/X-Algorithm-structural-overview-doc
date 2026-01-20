---
title: Python Phoenix Overview
description: An overview of the Python Phoenix component, including its role in recommendation models for retrieval and ranking, key files like recsys_model.py and recsys_retrieval_model.py, and configurations such as PhoenixRetrievalModelConfig and HashConfig.
sidebar_label: Python Phoenix Overview
sidebar_position: 1
tags: [python, phoenix, recommendation, retrieval, ranking]
---

# Python Phoenix Overview

The Phoenix Python component is a core implementation of a recommendation system designed to power content ranking and retrieval, particularly for predicting user engagement with items like posts (e.g., likes, reposts, replies). It leverages transformer-based architectures adapted from large-scale language models to handle both efficient candidate retrieval from vast corpora and precise ranking of shortlisted candidates. This system operates in a two-stage pipeline, enabling scalable processing: first retrieving relevant candidates from millions of items, then ranking a smaller subset (e.g., hundreds) based on predicted engagement probabilities across multiple action types.

Phoenix is built for high-performance inference and training, emphasizing efficiency in recommendation tasks. It uses hash-based embeddings for sparse feature representation, custom attention masking to isolate candidates during ranking, and L2-normalized embeddings for similarity-based retrieval. The codebase is structured around modular components for user encoding, candidate processing, and model configuration, making it adaptable for production-scale deployment in content feeds.

For more details on specific components, see the [Recsys Model](docs/recsys-model.md) and [Recsys Retrieval Model](docs/recsys-retrieval-model.md) pages.

## Role in Recommendation Models: Retrieval and Ranking

Phoenix plays a pivotal role in modern recommendation systems by addressing the challenges of scale and personalization. The system predicts engagement for content items based on user features, interaction history, and item attributes (e.g., post and author details).

- **Retrieval Stage**: This stage efficiently identifies top candidates from a large corpus (millions of items) using a two-tower architecture. The "user tower" encodes user profiles and engagement history into a dense embedding via a transformer. The "candidate tower" projects item features (e.g., post and author embeddings) into a shared space. Similarity is computed via dot products on L2-normalized embeddings, enabling approximate nearest neighbor (ANN) search to retrieve top-K items (e.g., 1000s). This narrows the search space quickly without exhaustive computation.

- **Ranking Stage**: For the retrieved candidates, a more expressive transformer model scores each one independently for multiple engagement actions (e.g., like, repost, reply). A key innovation is "candidate isolation" via attention masking: candidates can attend to user/history context but not to each other, ensuring scores are unbiased by batch composition. Outputs are multi-action logits, converted to probabilities (e.g., via sigmoid) for ranking by primary metrics like favorite probability.

This pipeline supports real-time personalization, such as in social media feeds, by balancing recall (retrieval) and precision (ranking). The models predict across 19+ action types, including positive engagements (e.g., click, dwell) and negative signals (e.g., block, report), enabling diverse ranking strategies.

The architecture draws from transformer designs but customizes them for recsys: input embeddings combine hashed features (users, posts, authors) with actions and product surfaces; outputs focus on per-candidate predictions rather than sequence generation.

For broader context on how Phoenix fits into the overall system, refer to the [Data Flow](docs/data-flow.md) and [Configuration](docs/configuration.md) pages.

## Key Files: recsys_model.py and recsys_retrieval_model.py

The Python codebase uses JAX for array computations and automatic differentiation, paired with Haiku (hk) for modular neural network definitions. Transformers are ported from a base implementation (e.g., Grok-1 style), with custom layers for embeddings, attention, and normalization. Haiku's `hk.Module` and `hk.transform` enable stateless, functional-style models suitable for JAX's just-in-time compilation and parallelism.

### recsys_model.py

This file defines the ranking model (`PhoenixModel`), focusing on transformer-based scoring of candidates. It handles input construction from hashed features and pre-looked-up embeddings, applies a transformer with special masking, and projects outputs to action logits.

Core structure:
- **PhoenixModelConfig**: A dataclass configuring the model, including transformer params (`model: TransformerConfig`), embedding size (`emb_size`), action count (`num_actions`), sequence lengths (`history_seq_len`, `candidate_seq_len`), and hashing (`hash_config: HashConfig`). It supports initialization and creation of the model instance.

  ```python
  @dataclass
  class PhoenixModelConfig:
      model: TransformerConfig
      emb_size: int
      num_actions: int
      history_seq_len: int = 128
      candidate_seq_len: int = 32
      # ... other fields like fprop_dtype, hash_config
      def make(self):
          return PhoenixModel(model=self.model.make(), config=self, fprop_dtype=self.fprop_dtype)
  ```

- **PhoenixModel (hk.Module)**: The main ranking module. It embeds actions via projection (`_get_action_embeddings`), looks up categorical features (`_single_hot_to_embeddings`), and builds inputs (`build_inputs`) by concatenating user, history, and candidate embeddings with padding masks. The transformer processes this sequence, extracting candidate outputs for unembedding to logits `[B, num_candidates, num_actions]`.
  - Key method: `__call__` runs the forward pass, applying layer norm and unembedding.
  - Embeddings use variance-scaled initializers for stability.
  - Example snippet for action embedding:
    ```python
    def _get_action_embeddings(self, actions: jax.Array) -> jax.Array:
        # Projects multi-hot actions to [B, S, D]
        action_projection = hk.get_parameter("action_projection", [num_actions, D], init=embed_init)
        actions_signed = (2 * actions - 1).astype(jnp.float32)
        action_emb = jnp.dot(actions_signed, action_projection)
        # Mask invalid actions
        return action_emb.astype(self.fprop_dtype)
    ```
  - The model ensures candidate isolation by passing `candidate_start_offset` to the transformer, which applies a custom attention mask (e.g., via `make_recsys_attn_mask` in supporting files).

This file emphasizes ranking's need for expressive modeling, using the full transformer for context-aware predictions. For in-depth details, see the [Recsys Model](docs/recsys-model.md) page.

### recsys_retrieval_model.py

This implements the two-tower retrieval model (`PhoenixRetrievalModel`), sharing the transformer for user encoding but using a simpler MLP-based candidate tower for efficiency.

Core structure:
- **PhoenixRetrievalModelConfig**: Similar to the ranking config but tailored for retrieval, with fields like `history_seq_len` and `hash_config`. It initializes and creates the model.

  ```python
  @dataclass
  class PhoenixRetrievalModelConfig:
      model: TransformerConfig
      emb_size: int
      history_seq_len: int = 128
      candidate_seq_len: int = 32
      hash_config: HashConfig = None
      # ... initialization logic
      def make(self):
          return PhoenixRetrievalModel(model=self.model.make(), config=self)
  ```

- **PhoenixRetrievalModel (hk.Module)**: Encodes users via transformer (`build_user_representation`) and candidates via `CandidateTower` (`build_candidate_representation`). Retrieval uses matrix multiplication for top-K selection.
  - **CandidateTower (hk.Module)**: A lightweight MLP projecting concatenated post+author embeddings to normalized vectors `[B, C, D]`. Uses SiLU activation and L2 normalization.
    ```python
    @dataclass
    class CandidateTower(hk.Module):
        emb_size: int
        def __call__(self, post_author_embedding: jax.Array) -> jax.Array:
            # Reshape and project: concat -> hidden (SiLU) -> output
            proj_1 = hk.get_parameter("candidate_tower_projection_1", [input_dim, emb_size * 2])
            hidden = jax.nn.silu(jnp.dot(post_author_embedding, proj_1))
            candidate_embeddings = jnp.dot(hidden, proj_2)
            # L2 normalize
            return candidate_embeddings / jnp.sqrt(jnp.maximum(jnp.sum(candidate_embeddings**2, axis=-1, keepdims=True), EPS))
    ```
  - User representation pools transformer outputs from user+history, normalized for similarity.
  - `__call__` computes user embeddings, then retrieves top-K via `_retrieve_top_k` (dot product + `jax.lax.top_k`).
  - Shares embedding logic (e.g., `_get_action_embeddings`) with the ranking model for consistency.

Haiku's transformation wraps methods like `build_user_representation` for parameter management, enabling efficient JAX compilation. JAX arrays (e.g., `jnp.bfloat16` for dtype) optimize for GPU/TPU inference. For more on this model, see the [Recsys Retrieval Model](docs/recsys-retrieval-model.md) page.

## Highlighted Configurations: PhoenixRetrievalModelConfig and HashConfig

- **PhoenixRetrievalModelConfig**: Central to retrieval, it defines the transformer's scale (e.g., `emb_size=128`, `num_layers=2`, `num_q_heads=2`) and sequence handling. It integrates `HashConfig` for embedding lookups and ensures L2 normalization for ANN compatibility. Used in runners (e.g., `RetrievalModelRunner`) for init and inference. See the [Configuration](docs/configuration.md) page for full details.

- **HashConfig**: Handles sparse, multi-hash embeddings to represent entities like users/posts/authors without full vocabularies. Specifies hash counts (e.g., `num_user_hashes=2`, `num_item_hashes=2`), enabling ensemble-like representations. Integrated via functions like `block_user_reduce` and `block_history_reduce`, which project concatenated hash embeddings (e.g., `[B, num_hashes * D] -> [B, D]`) using learned matrices.

  ```python
  @dataclass
  class HashConfig:
      num_user_hashes: int = 2
      num_item_hashes: int = 2
      num_author_hashes: int = 2
  ```

  This reduces memory for large-scale features, with padding masks for invalid hashes (e.g., hash=0).

## Codebase Structure and Usage

Supporting files like `runners.py` provide inference wrappers (e.g., `RecsysRetrievalInferenceRunner` for encoding/retrieval, `RecsysInferenceRunner` for ranking), using Haiku transforms for stateless application. Demo scripts (`run_retrieval.py`, `run_ranker.py`) showcase end-to-end usage: create batches/embeddings, initialize models, and output ranked results with visualizations (e.g., progress bars for scores).

Tests (e.g., `test_recsys_retrieval_model.py`) validate shapes, normalization, and top-K logic using JAX's `hk.without_apply_rng` for pure forward passes. The overall structure promotes modularity: configs drive model creation, Haiku modules handle layers, and JAX enables vectorized ops for batch efficiency.

This Python component integrates with broader systems (e.g., Rust pipelines for production), forming a robust recsys backbone. For testing and running, check the [Runners](docs/runners.md) and [Testing Phoenix](docs/testing-phoenix.md) pages.

:::note
Phoenix's design prioritizes efficiency and scalability, making it suitable for real-time recommendation in large-scale environments like social media platforms.
:::
