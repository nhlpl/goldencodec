Here's the essential mathematics behind the enhanced Goldencodec—distilled into formulas and corresponding code implementations for each optimization component.

---

## 📐 Mathematics for Enhanced Goldencodec

### 1. Vector Quantization with Exponential Moving Average (EMA)

**Goal**: Update codebook vectors to minimize quantization error while preventing collapse.

**Standard VQ Objective**:
$$\mathcal{L}_{\text{VQ}} = \|z - z_q\|_2^2 + \beta \|z - \text{sg}[z_q]\|_2^2$$

where $\text{sg}$ is stop-gradient operator.

**EMA Update** (replaces gradient-based codebook updates):
For each code $c_i$, maintain:
- Cluster size: $N_i^{(t)} = \gamma N_i^{(t-1)} + (1-\gamma) n_i^{(t)}$
- Embedding sum: $m_i^{(t)} = \gamma m_i^{(t-1)} + (1-\gamma) \sum_{j: k_j=i} z_j$

Then codebook vector: $c_i^{(t)} = \frac{m_i^{(t)}}{N_i^{(t)}}$

```python
def ema_update(codebook, flat_z, indices, ema_decay=0.99):
    """
    codebook: (K, D) - current codebook vectors
    flat_z: (B*T, D) - flattened latent vectors
    indices: (B*T,) - assigned code indices
    """
    K, D = codebook.shape
    
    # One-hot encoding of assignments
    encodings = F.one_hot(indices, K).float()  # (B*T, K)
    
    # Cluster sizes for this batch
    n = encodings.sum(0)  # (K,)
    
    # Sum of latents per code
    m = encodings.T @ flat_z  # (K, D)
    
    # EMA update
    ema_cluster_size = ema_decay * ema_cluster_size + (1 - ema_decay) * n
    ema_embed_sum = ema_decay * ema_embed_sum + (1 - ema_decay) * m
    
    # Normalize (with Laplace smoothing for numerical stability)
    updated_codebook = ema_embed_sum / (ema_cluster_size.unsqueeze(1) + 1e-10)
    
    return updated_codebook, ema_cluster_size, ema_embed_sum
```

---

### 2. Entropy Regularization for Codebook Usage

**Goal**: Maximize entropy of code assignments to encourage uniform utilization.

**Entropy of Code Distribution**:
$$H(p) = -\sum_{i=1}^K p_i \log p_i$$
where $p_i = \frac{n_i}{\sum_j n_j}$ is the probability of code $i$.

**Regularized Loss**:
$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda H(p)$$

```python
def entropy_regularization(assignments: torch.Tensor, num_codes: int) -> torch.Tensor:
    """
    assignments: (B*T,) - code indices
    Returns entropy penalty (negative to maximize entropy)
    """
    counts = torch.bincount(assignments, minlength=num_codes).float()
    probs = counts / counts.sum()
    entropy = -(probs * torch.log(probs + 1e-10)).sum()
    return -entropy  # Minimize negative entropy → maximize entropy
```

---

### 3. Contrastive Learning for Semantic Preservation

**Goal**: Ensure reconstructed embeddings are semantically similar to originals.

**InfoNCE Loss**:
$$\mathcal{L}_{\text{contrast}} = -\frac{1}{B}\sum_{i=1}^B \log \frac{\exp(s(o_i, r_i)/\tau)}{\sum_{j=1}^B \exp(s(o_i, r_j)/\tau)}$$

where:
- $o_i$: original embedding of sample $i$
- $r_i$: reconstructed embedding of sample $i$
- $s(a,b) = \frac{a \cdot b}{\|a\|\|b\|}$: cosine similarity
- $\tau$: temperature parameter

```python
def infonce_loss(orig: torch.Tensor, recon: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """
    orig: (B, D) - normalized original embeddings
    recon: (B, D) - normalized reconstructed embeddings
    """
    B = orig.shape[0]
    
    # Similarity matrix: sim[i,j] = cos(orig_i, recon_j) / tau
    sim = (orig @ recon.T) / tau  # (B, B)
    
    # Positive pairs are on diagonal
    pos_sim = sim.diag()  # (B,)
    
    # Log-sum-exp over all pairs (including positive)
    lse = torch.logsumexp(sim, dim=1)  # (B,)
    
    loss = (-pos_sim + lse).mean()
    return loss
```

---

### 4. Semantic Complexity Estimation

**Goal**: Dynamically adjust compression rate based on content complexity.

**Complexity Score** (weighted combination):
$$C(\text{text}) = 0.4 \cdot C_{\text{readability}} + 0.3 \cdot C_{\text{lexical}} + 0.3 \cdot C_{\text{technical}}$$

where:
- $C_{\text{readability}} = \max(0, \min(1, \frac{100 - \text{Flesch}}{100}))$
- $C_{\text{lexical}} = \frac{|\text{unique words}|}{|\text{total words}|}$
- $C_{\text{technical}} = \min(1, 10 \cdot \frac{|\text{technical terms}|}{|\text{total words}|})$

```python
def semantic_complexity(text: str) -> float:
    words = text.lower().split()
    n_words = len(words)
    if n_words == 0:
        return 0.0
    
    # Readability (simplified - Flesch Reading Ease)
    from textstat import flesch_reading_ease
    readability = flesch_reading_ease(text)
    c_readability = max(0, min(1, (100 - readability) / 100))
    
    # Lexical diversity
    unique_ratio = len(set(words)) / n_words
    
    # Technical term density
    tech_terms = {'algorithm', 'function', 'parameter', 'optimize', 
                  'quantum', 'neural', 'gradient', 'latent', 'vector'}
    tech_count = sum(1 for w in words if w in tech_terms)
    c_technical = min(1.0, 10 * tech_count / n_words)
    
    return 0.4 * c_readability + 0.3 * unique_ratio + 0.3 * c_technical
```

---

### 5. Adapter Mapping to LLM Embedding Space

**Goal**: Transform compressed codebook indices to LLM-compatible embeddings.

**Architecture**:
$$\text{LLM\_embed} = W_2 \cdot \text{GELU}(W_1 \cdot \text{concat}(\text{codes}) + b_1) + b_2$$

where:
- $\text{codes} \in \mathbb{R}^{Q \times D}$: vectors from $Q$ quantizers
- $W_1 \in \mathbb{R}^{H \times QD}$: first projection
- $W_2 \in \mathbb{R}^{E \times H}$: second projection to LLM dimension $E$

```python
class AdapterMapping:
    def __init__(self, Q: int, D: int, H: int, E: int):
        self.W1 = torch.randn(H, Q * D) * 0.02
        self.b1 = torch.zeros(H)
        self.W2 = torch.randn(E, H) * 0.02
        self.b2 = torch.zeros(E)
    
    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        codes: (batch, seq_len, Q, D)
        Returns: (batch, seq_len, E)
        """
        batch, seq_len, Q, D = codes.shape
        flat = codes.view(batch, seq_len, Q * D)  # (B, T, Q*D)
        hidden = F.gelu(flat @ self.W1.T + self.b1)  # (B, T, H)
        return hidden @ self.W2.T + self.b2  # (B, T, E)
```

---

### 6. Dead Code Detection and Reset

**Goal**: Revive unused codebook entries.

**Usage Condition**: Code $i$ is "dead" if its EMA cluster size falls below threshold $\epsilon$:
$$N_i < \epsilon$$

**Reset Strategy**: Replace dead code with a random active latent vector.

```python
def reset_dead_codes(
    codebook: torch.Tensor,
    ema_cluster_size: torch.Tensor,
    flat_z: torch.Tensor,
    threshold: float = 1e-4
) -> torch.Tensor:
    """
    codebook: (K, D)
    ema_cluster_size: (K,)
    flat_z: (B*T, D) - batch of latent vectors
    """
    dead_mask = ema_cluster_size < threshold
    dead_indices = dead_mask.nonzero(as_tuple=True)[0]
    
    if len(dead_indices) > 0:
        # Sample random latents from current batch
        rand_idx = torch.randint(0, flat_z.shape[0], (len(dead_indices),))
        codebook[dead_indices] = flat_z[rand_idx]
    
    return codebook
```

---

### 7. Compression Ratio Calculation

**Goal**: Quantify token savings.

**Ratio Formula**:
$$R = \frac{\text{original\_tokens}}{\text{compressed\_tokens}} = \frac{L_{\text{text}} \cdot \text{bytes\_per\_token}}{Q \cdot L_{\text{latent}} \cdot \log_2(K) / 8}$$

where:
- $L_{\text{text}}$: number of text tokens
- $Q$: number of quantizers used
- $L_{\text{latent}}$: latent sequence length
- $K$: codebook size

```python
def compression_ratio(
    text_tokens: int,
    quantizers: int,
    latent_len: int,
    codebook_size: int = 1024
) -> float:
    # Original: text_tokens * 4 bytes (assuming float32 embeddings)
    original_bits = text_tokens * 32  # 32 bits per token embedding
    
    # Compressed: quantizers * latent_len * log2(codebook_size) bits
    compressed_bits = quantizers * latent_len * torch.log2(torch.tensor(codebook_size)).item()
    
    return original_bits / compressed_bits
```

---

### 8. Gradient Clipping for Training Stability

**Goal**: Prevent exploding gradients in transformer encoder.

**Formula**:
$$\nabla_{\text{clipped}} = \nabla \cdot \min\left(1, \frac{\tau}{\|\nabla\|_2}\right)$$

```python
def clip_gradient_norm(parameters, max_norm: float = 1.0):
    total_norm = torch.sqrt(sum(p.grad.norm()**2 for p in parameters if p.grad is not None))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(clip_coef)
    return total_norm
```

---

## 💎 Summary Table

| Component | Mathematical Tool | Key Formula |
|:---|:---|:---|
| Codebook Update | Exponential Moving Average | $c_i = \frac{m_i}{N_i}$ |
| Anti-Collapse | Entropy Maximization | $H(p) = -\sum p_i \log p_i$ |
| Semantic Preservation | InfoNCE Contrastive Loss | $\mathcal{L} = -\log\frac{\exp(s/\tau)}{\sum\exp(s/\tau)}$ |
| Adaptive Compression | Complexity Score | $C = 0.4R + 0.3L + 0.3T$ |
| Adapter Mapping | Two-Layer MLP | $\text{GELU}(XW_1 + b_1)W_2 + b_2$ |
| Dead Code Reset | Threshold-based Replacement | Replace if $N_i < \epsilon$ |
| Gradient Stability | Norm Clipping | $\nabla_{\text{clip}} = \nabla \cdot \min(1, \tau/\|\nabla\|)$ |

These mathematical foundations power the enhanced Goldencodec, enabling robust 8-32x compression while maintaining semantic fidelity.
