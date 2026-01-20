---
title: Testing Phoenix
slug: testing-phoenix
description: An overview of the testing structure in the Phoenix recommendation system, focusing on unit tests for key components like CandidateTower, PhoenixRetrievalModel, and inference runners. Includes details on setup fixtures and example data generation.
sidebar_label: Testing Phoenix
sidebar_position: 12
---

# Testing Phoenix

The Phoenix recommendation system employs a comprehensive unit testing strategy to ensure the reliability and correctness of its core components, particularly in the retrieval stage. Tests are written using Python's `unittest` framework (with some use of `pytest` in related files) and leverage libraries like JAX, Haiku, and NumPy for model validation. The testing structure emphasizes functional correctness, such as output shapes, L2 normalization (essential for embedding similarity computations), and retrieval logic.

Testing is modular, with dedicated files for different aspects of the pipeline. This page focuses on the primary test file `test_recsys_retrieval_model.py`, which covers the retrieval model and runners. A complementary file, `test_recsys_model.py`, tests shared utilities like attention masking used in both retrieval and ranking stages. Tests use small-scale configurations for efficiency, ensuring fast execution while validating key behaviors.

For more on the overall Phoenix architecture, see the [Phoenix Overview](/docs/python-phoenix-overview.md) and [RecSys Retrieval Model](/docs/recsys-retrieval-model.md) pages.

## Key Test Files

- **`test_recsys_retrieval_model.py`**: The main file for retrieval-specific tests, covering the `CandidateTower`, full `PhoenixRetrievalModel`, and inference runners. It verifies forward passes, normalization, shape consistency, and top-k retrieval logic.
- **`test_recsys_model.py`**: Tests core model utilities, such as attention masking for the transformer, which isolates candidates to prevent cross-attention. This supports both retrieval (user tower) and ranking pipelines.

These files are located in the codebase's test directory and can be run via `python test_recsys_retrieval_model.py` or `pytest test_recsys_model.py -v` for verbose output.

## Setup Fixtures and Example Data Generation

Tests rely on reusable setup fixtures to initialize configurations and generate synthetic data, promoting consistency and reducing boilerplate. Common patterns include:

### Configuration Fixtures
- **PhoenixRetrievalModelConfig**: A central fixture defining model hyperparameters, such as embedding size (`emb_size=64`), sequence lengths (history: `16`, candidates: `8`), hash configurations (`HashConfig` with 2 hashes each for users, items, and authors), and a simple `TransformerConfig` (1 layer, `emb_size=64`, `num_heads=2`).
- **HashConfig**: Specifies the number of hashes for user/item/author embeddings, used in data preparation.
- These are set in `setUp` methods of test classes to create a standardized environment.

### Example Data Generation
Tests use helper functions from the `runners` module to create reproducible, random inputs:
- **`create_example_batch(batch_size, emb_size, history_len, num_candidates, num_actions, num_user_hashes, num_item_hashes, num_author_hashes, product_surface_vocab_size)`**: Generates a batch of user history data, including action sequences, product surfaces, and hashed embeddings. Outputs a tuple of `(batch, embeddings)`, where:
  - `batch` is a structured array with fields like user history tokens, candidate offsets, and metadata.
  - `embeddings` are pre-computed embedding tables for hashes and vocab.
  - Example usage: Creates a batch of size 2 with 19 actions and vocab size 16.
- **`create_example_corpus(corpus_size, emb_size)`**: Produces a corpus of item embeddings (L2-normalized) and post IDs for retrieval testing. Default corpus size is 100.
  - Outputs: `(corpus_embeddings, corpus_post_ids)`, where embeddings are random normals normalized to unit length.

These functions ensure inputs are JAX-compatible (e.g., using `jax.random.PRNGKey(0)` for seeding) and mimic real-world data distributions without requiring external datasets.

Haiku's `hk.transform` and `hk.without_apply_rng` are used to create stateless forward functions for testing, initializing parameters with a fixed RNG key.

## Unit Tests for CandidateTower

The `CandidateTower` is a key component in the retrieval model, responsible for projecting concatenated post and author embeddings into a shared, normalized space via mean pooling (optionally with linear projection). Tests in `TestCandidateTower` validate its core functionality:

- **`test_candidate_tower_output_shape`**: 
  - Input: Random tensor of shape `[batch_size=4, num_candidates=8, num_hashes=4, emb_size=64]`.
  - Verifies output shape is `[batch_size, num_candidates, emb_size]` after processing.
  - Ensures dimensional reduction via pooling works correctly.

- **`test_candidate_tower_normalized`**:
  - Computes L2 norms of the output embeddings.
  - Asserts norms are approximately 1.0 (within 5 decimal places) using `np.testing.assert_array_almost_equal`.
  - Critical for dot-product similarity in retrieval.

- **`test_candidate_tower_mean_pooling`**:
  - Similar to the normalization test but focuses on mean pooling behavior (no linear layer in this variant).
  - Confirms shape preservation and normalization post-pooling.

These tests use a simple forward function wrapped in Haiku for parameter initialization and application.

## Unit Tests for PhoenixRetrievalModel

The `PhoenixRetrievalModel` implements a two-tower architecture: one for user representation (from history) and one for candidates (from corpus items). The `TestPhoenixRetrievalModel` class tests the full model, including `build_user_representation`, `build_candidate_representation`, and the `__call__` method for end-to-end retrieval.

- **Setup**: Uses `_create_test_batch()` and `_create_test_corpus()` for inputs (batch_size=2, corpus_size=100, top_k=10).

- **`test_model_forward`**:
  - Tests the model's `__call__(batch, embeddings, corpus_embeddings, top_k)`.
  - Verifies output structure: `user_representation` shape `[batch_size, emb_size]`, `top_k_indices` and `top_k_scores` shapes `[batch_size, top_k]`.

- **`test_user_representation_normalized`**:
  - Extracts user representations from the forward pass.
  - Asserts L2 norms ≈ 1.0 for the batch.

- **`test_candidate_representation_normalized`**:
  - Tests `build_candidate_representation(batch, embeddings)`.
  - Verifies candidate reps shape `[batch_size, candidate_seq_len, emb_size]` (implicit) and norms ≈ 1.0 across `[batch_size, candidate_seq_len]`.

- **`test_retrieve_top_k`**:
  - Validates top-k retrieval in `__call__`.
  - Checks shapes, ensures indices are valid (0 ≤ indices < corpus_size), and scores are in descending order per batch item (using NumPy assertions).

These tests confirm the model's ability to compute normalized embeddings and perform efficient ANN-style retrieval via dot products.

## Unit Tests for Runners

Runners handle inference workflows, such as encoding users and retrieving candidates. The `TestRetrievalInferenceRunner` class tests `RecsysRetrievalInferenceRunner`, which wraps `RetrievalModelRunner`.

- **Setup**: Similar to model tests, with `PhoenixRetrievalModelConfig` and batch/corpus generation.

- **`test_runner_initialization`**:
  - Creates a runner with `bs_per_device=0.125` (batch size per device).
  - Calls `initialize()` and asserts parameters (`runner.params`) are set.

- **`test_runner_encode_user`**:
  - Uses `encode_user(batch, embeddings)`.
  - Verifies output shape `[batch_size, emb_size]` for user representations.

- **`test_runner_retrieve`**:
  - Sets corpus via `set_corpus(corpus_embeddings, corpus_post_ids)`.
  - Calls `retrieve(batch, embeddings, top_k=10)`.
  - Checks output shapes for user reps and top-k results, ensuring integration with the model.

These tests validate the runner's role in production-like inference, including corpus management.

## Additional Testing Notes

- **Attention Masking Tests** (from `test_recsys_model.py`): Complement retrieval tests by verifying `make_recsys_attn_mask`, which ensures causal attention in user history and isolates candidates (no cross-candidate attention). Includes edge cases like single/multiple candidates and dtype preservation.
- **Best Practices**: Tests prioritize numerical stability (e.g., normalization checks) and use fixed seeds for reproducibility. For extension, add integration tests for the full retrieval-to-ranking pipeline (see [Runners](/docs/runners.md)).
- **Running Tests**: Execute with `unittest` or `pytest` for coverage. Expand with property-based testing (e.g., via Hypothesis) for robustness.

This testing suite ensures Phoenix's retrieval components are shape-correct, normalized, and logically sound, supporting scalable recommendation serving.
