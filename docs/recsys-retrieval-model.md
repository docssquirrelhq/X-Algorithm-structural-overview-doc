---
title: Phoenix Retrieval Model
slug: recsys-retrieval-model
description: Documentation for the Phoenix retrieval model in recsys_retrieval_model.py, covering the two-tower architecture, CandidateTower, user representation, top-k retrieval, and corpus handling.
sidebar_label: Retrieval Model
sidebar_position: 10
---

# Phoenix Retrieval Model

The Phoenix Retrieval Model, implemented in `recsys_retrieval_model.py`, is a two-tower neural network designed for efficient candidate retrieval in recommendation systems. It leverages the Phoenix Transformer architecture for encoding user representations and a lightweight projection tower for candidate (post + author) embeddings. The model enables fast approximate nearest neighbor (ANN) search by projecting both user and candidate representations into a shared, L2-normalized embedding space where similarity is computed via dot product (equivalent to cosine similarity).

This retrieval stage focuses on selecting a small set of relevant candidates (top-k) from a large corpus for each user, which can then be ranked in subsequent stages. The model handles user features, interaction history, and corpus embeddings efficiently using JAX and Haiku.

Key components include:
- **User Tower**: Encodes user profile and history using the Phoenix Transformer.
- **Candidate Tower**: Projects concatenated post and author embeddings.
- **Retrieval Mechanism**: Matrix multiplication for similarity scores followed by top-k selection.

For an overview of the broader Phoenix system, see [Python Phoenix Overview](../python-phoenix-overview.md) and [Recsys Model](../recsys-model.md).

## RetrievalOutput

The `RetrievalOutput` is a `NamedTuple` that encapsulates the results of the retrieval process:

```python
class RetrievalOutput(NamedTuple):
    """Output of the retrieval model."""
    user_representation: jax.Array  # [B, D] L2-normalized user embedding
    top_k_indices: jax.Array       # [B, K] Indices of top-k candidates from the corpus
    top_k_scores: jax.Array        # [B, K] Similarity scores for the top-k candidates
```

- `user_representation`: The final L2-normalized embedding for each user in the batch (shape `[B, D]`, where `B` is batch size and `D` is embedding dimension).
- `top_k_indices`: Integer indices into the corpus embeddings, selecting the top-k most similar candidates per user.
- `top_k_scores`: Raw dot-product similarity scores for the selected candidates (higher is better).

This output is returned by the main `__call__` method of `PhoenixRetrievalModel` and can be used for downstream ranking or evaluation.

## CandidateTower

The `CandidateTower` is a simple Haiku module (`hk.Module`) that projects concatenated post and author embeddings into the shared embedding space. It uses a two-layer MLP with SiLU activation and L2 normalization to produce candidate representations suitable for similarity search.

### Configuration
```python
@dataclass
class CandidateTower(hk.Module):
    emb_size: int  # Embedding dimension (e.g., 512)
    name: Optional[str] = None
```

### Forward Pass
The `__call__` method processes input embeddings:

```python
def __call__(self, post_author_embedding: jax.Array) -> jax.Array:
    # Input shape: [B, C, num_hashes, D] or [B, num_hashes, D]
    # Reshape to [B, C, concat_dim] or [B, concat_dim]
    
    # Two-layer projection: concat_dim -> 2*emb_size -> emb_size
    # With SiLU activation after first layer
    # L2-normalize output: [B, C, D] or [B, D]
```

- **Input**: Concatenated post and author embeddings (`post_author_embedding`), typically from hashed lookups in `RecsysEmbeddings`. Supports batched candidates (`C` for number of candidates per user).
- **Projection Layers**: 
  - First layer: Linear projection to `2 * emb_size` using variance-scaled initialization.
  - Activation: SiLU (Swish).
  - Second layer: Linear projection to `emb_size`.
- **Normalization**: L2-normalization ensures unit vectors, enabling efficient dot-product similarity.
- **Output**: Normalized candidate representations, ready for matrix multiplication with user embeddings.

This tower is lightweight compared to the user tower, allowing pre-computation of corpus embeddings for fast retrieval.

:::tip
The `CandidateTower` is instantiated within `PhoenixRetrievalModel` and shared across candidates for efficiency.
:::

## User Representation Building

User representations are built in the `build_user_representation` method of `PhoenixRetrievalModel`. This process encodes static user features and dynamic interaction history using the Phoenix Transformer, followed by mean pooling to produce a single vector per user.

### Configuration
The model is configured via `PhoenixRetrievalModelConfig`:

```python
@dataclass
class PhoenixRetrievalModelConfig:
    model: TransformerConfig  # Phoenix Transformer config
    emb_size: int            # Embedding size (D)
    history_seq_len: int = 128  # Max history length
    candidate_seq_len: int = 32 # Max candidates per user (not used in retrieval)
    hash_config: HashConfig = None
    product_surface_vocab_size: int = 16  # Vocab for product surfaces
    # ... other fields like fprop_dtype=jnp.bfloat16
```

### Building Process
1. **Embed User Features**:
   - User hashes (`batch.user_hashes`) are reduced using `block_user_reduce` to aggregate embeddings from `RecsysEmbeddings.user_embeddings`.
   - Produces `user_embeddings` [B, 1, D] and padding mask.

2. **Embed History**:
   - **Product Surfaces**: Single-hot encoded via `_single_hot_to_embeddings` lookup table [B, history_len, D].
   - **Actions**: Multi-hot vectors (`batch.history_actions`) projected via `_get_action_embeddings` (signed and masked).
   - **Posts and Authors**: History post/author hashes reduced using `block_history_reduce` with embeddings from `RecsysEmbeddings`.
   - Concatenates to `history_embeddings` [B, history_len, D] and padding mask.

3. **Transformer Encoding**:
   - Concatenate user + history: `embeddings` [B, seq_len, D] where `seq_len = 1 + history_seq_len`.
   - Feed into `self.model` (Phoenix Transformer) with padding mask.
   - Output: Transformer-hidden states [B, seq_len, D].

4. **Pooling and Normalization**:
   - Mask and sum embeddings along sequence dimension (mean pooling over valid tokens).
   - L2-normalize to get `user_representation` [B, D].

```python
def build_user_representation(self, batch: RecsysBatch, recsys_embeddings: RecsysEmbeddings) -> Tuple[jax.Array, jax.Array]:
    # ... embedding steps ...
    model_output = self.model(embeddings.astype(self.fprop_dtype), padding_mask, candidate_start_offset=None)
    # Mean pool with mask
    # L2 normalize
    return user_representation, user_norm  # user_norm is pre-normalization L2 norm
```

This produces a compact, context-aware user embedding that captures both profile and recent interactions.

## Top-k Retrieval via Matrix Multiplication

Retrieval is performed in the `_retrieve_top_k` method, using efficient matrix operations for similarity search.

### Process
1. **Similarity Computation**:
   - Compute dot products: `scores = jnp.matmul(user_representation, corpus_embeddings.T)` → [B, N] where N is corpus size.
   - If provided, apply `corpus_mask` to set invalid scores to `-INF`.

2. **Top-k Selection**:
   - Use `jax.lax.top_k(scores, top_k)` to get scores and indices.
   - Output: `top_k_indices` [B, K], `top_k_scores` [B, K].

```python
def _retrieve_top_k(self, user_representation: jax.Array, corpus_embeddings: jax.Array, top_k: int, corpus_mask: Optional[jax.Array] = None) -> Tuple[jax.Array, jax.Array]:
    scores = jnp.matmul(user_representation, corpus_embeddings.T)
    if corpus_mask is not None:
        scores = jnp.where(corpus_mask[None, :], scores, -INF)
    top_k_scores, top_k_indices = jax.lax.top_k(scores, top_k)
    return top_k_indices, top_k_scores
```

This is called within the main `__call__` method after building user representations:

```python
def __call__(self, batch: RecsysBatch, recsys_embeddings: RecsysEmbeddings, corpus_embeddings: jax.Array, top_k: int, corpus_mask: Optional[jax.Array] = None) -> RetrievalOutput:
    user_representation, _ = self.build_user_representation(batch, recsys_embeddings)
    top_k_indices, top_k_scores = self._retrieve_top_k(user_representation, corpus_embeddings, top_k, corpus_mask)
    return RetrievalOutput(user_representation=user_representation, top_k_indices=top_k_indices, top_k_scores=top_k_scores)
```

The matrix multiplication is highly parallelizable on accelerators, making it suitable for large-scale retrieval.

## Corpus Handling and Similarity Search

### Corpus Embeddings
- **Input**: `corpus_embeddings` [N, D], pre-computed L2-normalized representations of all candidates (posts + authors) in the corpus.
- These are typically generated offline using the `CandidateTower` on the full item catalog and stored for fast access.
- Supports dynamic corpora via optional `corpus_mask` [N], a boolean array indicating valid entries (e.g., to exclude expired or filtered items). Invalid scores are masked to `-INF` during top-k.

### Similarity Search
- **Metric**: Dot product in the normalized space, equivalent to cosine similarity: \(\cos(\theta) = \mathbf{u} \cdot \mathbf{c}\) where \(\mathbf{u}\) and \(\mathbf{c}\) are unit vectors.
- **Efficiency**: 
  - Exact search via matrix multiplication for moderate corpus sizes.
  - For very large corpora (e.g., millions of items), this can be approximated with ANN libraries like FAISS (not implemented here, but compatible via pre-computed embeddings).
- **Handling**: The model assumes the corpus is fixed during inference. Updates to the corpus require re-projecting new candidates through the `CandidateTower`.

:::caution
Ensure `corpus_embeddings` are L2-normalized to match user representations. Mismatches can lead to incorrect similarity scores.
:::

### Integration
The retrieval model integrates with the broader pipeline via `RecsysBatch` (user/history/candidate hashes) and `RecsysEmbeddings` (pre-looked-up embeddings from a shared embedding table). For details on embeddings and hashing, see [Recsys Model](../recsys-model.md).

This design balances expressiveness (via Transformer) with retrieval speed, making it ideal for real-time recommendation systems.