# Memory-Efficient Per-Token Log Probability Extraction

## Overview

This tutorial explains why `get_per_token_logps` uses a loop to process rows one at a time, even though `torch.gather` could theoretically process the entire batch simultaneously. The key reason is **memory efficiency** - processing row-by-row significantly reduces peak memory usage.

**⚠️ Critical Note on Training vs Inference:**

During **training**, gradients need to flow back through the computation. This significantly changes the memory dynamics:

1. **If `log_probs` is deleted**: PyTorch will recompute it from `logits` during backward pass
2. **`logits` must stay in memory** for this recomputation (or be recomputed from inputs)
3. **Memory benefit is reduced** because `logits` can't be freed early
4. **However, the loop still helps** because:
   - Forward peak memory is lower (one row of `log_probs` vs all rows)
   - Backward recomputation happens row-by-row (one row at a time)
   - Works well with gradient checkpointing (which recomputes anyway)

During **inference** (no gradients), the loop provides maximum memory savings (~44% reduction) since intermediate values can be safely deleted without recomputation concerns.

## The Function

```python
def get_per_token_logps(logits, input_ids):
    per_token_logps = [] # Use a loop to reduce memory peak.
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)
```

## Why Use a Loop?

### The Naive Batch Approach

In principle, you could process the entire batch at once:

```python
def get_per_token_logps_naive(logits, input_ids):
    """
    Naive batch processing - processes all rows simultaneously.
    WARNING: This uses significantly more memory!
    """
    # logits: (B, L-1, V)
    # input_ids: (B, L-1)
    
    # Compute log_softmax for entire batch
    log_probs = logits.log_softmax(dim=-1)  # (B, L-1, V)
    
    # Gather for entire batch
    token_log_probs = torch.gather(
        log_probs,
        dim=2,  # Gather along vocab dimension
        index=input_ids.unsqueeze(-1)  # (B, L-1, 1)
    ).squeeze(-1)  # (B, L-1)
    
    return token_log_probs
```

### Memory Comparison

Let's analyze the memory usage for a typical case:
- Batch size: `B = 8`
- Sequence length: `L = 912`
- Vocabulary size: `V = 50,000`
- Data type: `bfloat16` (2 bytes per element)

#### Naive Batch Approach:

```python
logits = torch.randn(8, 912, 50000, dtype=torch.bfloat16)  # ~730 MB
log_probs = logits.log_softmax(dim=-1)  # ~730 MB (new tensor)
index = input_ids.unsqueeze(-1)  # ~7 KB (negligible)
token_log_probs = torch.gather(log_probs, dim=2, index=index)  # ~7 KB

# Peak memory: ~1.46 GB (logits + log_probs simultaneously)
```

#### Loop-Based Approach:

```python
for logits_row, input_ids_row in zip(logits, input_ids):
    # logits_row: (912, 50000) - ~91 MB per iteration
    log_probs = logits_row.log_softmax(dim=-1)  # ~91 MB (new tensor)
    token_log_prob = torch.gather(...)  # ~2 KB
    per_token_logps.append(token_log_prob)
    # log_probs is freed after each iteration

# Peak memory: ~730 MB (only logits) + ~91 MB (one log_probs) = ~821 MB
# Memory saved: ~640 MB (44% reduction!)
```

### Why This Matters

1. **Large Vocabulary Sizes**: Modern language models have vocabularies of 50k-100k tokens. The `log_probs` tensor is huge: `(B, L, V) × 2 bytes = B × L × V × 2 bytes`.

2. **Memory Fragmentation**: Creating large intermediate tensors can cause memory fragmentation, making it harder to allocate contiguous memory blocks later.

3. **GPU Memory Limits**: GPUs have limited memory (e.g., 40GB). Every MB saved helps prevent OOM errors.

4. **Gradient Memory**: During backpropagation, intermediate tensors need to be kept for gradient computation. Reducing peak memory during forward pass leaves more room for gradients.

## Detailed Memory Analysis

### Memory Footprint Breakdown

For `logits` shape `(B=8, L=912, V=50000)`:

| Tensor | Shape | Size (MB) | When Allocated |
|--------|-------|-----------|----------------|
| `logits` | (8, 912, 50000) | ~730 MB | Always |
| `log_probs` (batch) | (8, 912, 50000) | ~730 MB | During batch processing |
| `log_probs` (row) | (912, 50000) | ~91 MB | During loop iteration |
| `token_log_probs` | (8, 912) | ~0.015 MB | Final result |

**Peak Memory:**
- Batch approach: ~1,460 MB (logits + log_probs)
- Loop approach: ~821 MB (logits + one row of log_probs)
- **Savings: ~640 MB (44% reduction)**

### Why Not Process All Rows Simultaneously?

The naive batch approach creates a large intermediate `log_probs` tensor that exists simultaneously with `logits`:

```python
# Both tensors exist in memory at the same time:
logits = ...      # 730 MB
log_probs = logits.log_softmax(dim=-1)  # +730 MB = 1,460 MB peak!
```

The loop approach processes one row at a time, so only one row's `log_probs` exists at a time:

```python
logits = ...  # 730 MB
for row in logits:
    log_probs_row = row.log_softmax(dim=-1)  # +91 MB = 821 MB peak
    # log_probs_row is freed after use
```

## When to Use Each Approach

### Use Loop-Based (Current Implementation) When:

✅ **Large vocabulary sizes** (50k+ tokens)  
✅ **Limited GPU memory**  
✅ **Large batch sizes**  
✅ **Long sequences**  
✅ **Memory is a bottleneck**

### Use Batch Processing When:

✅ **Small vocabulary sizes** (< 10k tokens)  
✅ **Plenty of GPU memory available**  
✅ **Speed is more important than memory**  
✅ **Small batch sizes**

## Performance Trade-offs

### Speed Comparison

```python
# Batch processing: Faster (single operation)
log_probs = logits.log_softmax(dim=-1)  # One CUDA kernel call
token_log_probs = torch.gather(log_probs, dim=2, index=input_ids.unsqueeze(-1))

# Loop processing: Slower (multiple operations)
for logits_row, input_ids_row in zip(logits, input_ids):
    log_probs = logits_row.log_softmax(dim=-1)  # B CUDA kernel calls
    token_log_prob = torch.gather(...)
```

**Speed Impact:**
- Batch: ~1-2x faster (single kernel launch vs. B kernel launches)
- Loop: ~1-2x slower, but saves ~44% memory

**Memory Impact:**
- Batch: ~2x peak memory (logits + log_probs)
- Loop: ~1.1x peak memory (logits + one row)

## Implementation Details

### Why the Loop Works

The loop processes each sample independently:

```python
for logits_row, input_ids_row in zip(logits, input_ids):
    # logits_row: (L-1, V) - one sample's logits
    # input_ids_row: (L-1,) - one sample's token IDs
    
    # Compute log probabilities for this sample only
    log_probs = logits_row.log_softmax(dim=-1)  # (L-1, V)
    
    # Extract log prob of actual tokens
    token_log_prob = torch.gather(
        log_probs,
        dim=1,
        index=input_ids_row.unsqueeze(1)  # (L-1, 1)
    ).squeeze(1)  # (L-1,)
    
    per_token_logps.append(token_log_prob)
    # log_probs is automatically freed here (Python GC)
```

### Memory Deallocation

Python's garbage collector automatically frees `log_probs` after each iteration, but PyTorch's CUDA memory allocator may not immediately return memory to the OS. However, it's available for reuse in the next iteration, which is what matters for peak memory.

### Explicit Memory Management

For even better memory control, you can explicitly delete tensors:

```python
for logits_row, input_ids_row in zip(logits, input_ids):
    log_probs = logits_row.log_softmax(dim=-1)
    token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
    per_token_logps.append(token_log_prob)
    
    # Explicitly free memory
    del log_probs, token_log_prob
    
    # Optional: Clear CUDA cache periodically
    if i % 10 == 0:
        torch.cuda.empty_cache()
```

## Real-World Impact

### Example: Training a 1.5B Model

**Scenario:**
- Model: 1.5B parameters (~3GB in bfloat16)
- Batch size: 8
- Sequence length: 912
- Vocabulary: 50,000 tokens
- GPU: 40GB

**Memory Usage:**

| Component | Size |
|-----------|------|
| Model weights | ~3 GB |
| Optimizer states (ZeRO-2) | ~6 GB |
| Activations | ~2 GB |
| Gradients | ~3 GB |
| **Available for log_probs** | **~26 GB** |

**With Batch Processing:**
- `log_probs`: ~730 MB × 2 (forward + backward) = ~1.46 GB
- Leaves ~24.5 GB for other operations

**With Loop Processing:**
- `log_probs`: ~91 MB × 2 = ~182 MB
- Leaves ~25.8 GB for other operations
- **Saves ~640 MB** - enough for an extra batch or longer sequences!

### Preventing OOM Errors

The loop-based approach is crucial for preventing "CUDA out of memory" errors:

```python
# Without loop (batch processing):
# Peak memory during forward pass: Model + Activations + logits + log_probs
# = 3GB + 2GB + 730MB + 730MB = ~6.5GB per GPU
# With 7 GPUs: 6.5GB × 7 = 45.5GB total → OOM on 40GB GPUs!

# With loop (row-by-row):
# Peak memory: Model + Activations + logits + one_row_log_probs
# = 3GB + 2GB + 730MB + 91MB = ~5.8GB per GPU
# With 7 GPUs: 5.8GB × 7 = 40.6GB total → Fits!
```

## Alternative: Chunked Batch Processing

If you want a middle ground between speed and memory, you can process in chunks:

```python
def get_per_token_logps_chunked(logits, input_ids, chunk_size=2):
    """
    Process in chunks to balance memory and speed.
    """
    per_token_logps = []
    for i in range(0, logits.shape[0], chunk_size):
        chunk_logits = logits[i:i+chunk_size]  # (chunk_size, L-1, V)
        chunk_input_ids = input_ids[i:i+chunk_size]  # (chunk_size, L-1)
        
        chunk_log_probs = chunk_logits.log_softmax(dim=-1)
        chunk_token_log_probs = torch.gather(
            chunk_log_probs,
            dim=2,
            index=chunk_input_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        per_token_logps.append(chunk_token_log_probs)
        del chunk_log_probs  # Free memory
    
    return torch.cat(per_token_logps, dim=0)
```

**Trade-offs:**
- `chunk_size=1`: Maximum memory savings, slowest
- `chunk_size=B`: No memory savings, fastest
- `chunk_size=2-4`: Good balance

## Gradient Flow Analysis

### What PyTorch Needs for Backward Pass

When `per_token_logps` requires gradients, PyTorch needs to compute:

```
∂loss/∂logits = ∂loss/∂per_token_logps × ∂per_token_logps/∂log_probs × ∂log_probs/∂logits
```

**If `log_probs` is deleted:**
- PyTorch stores the computation graph (operations, not values)
- During backward, PyTorch recomputes `log_probs` from `logits`
- This requires `logits` to remain in memory
- Recomputation happens one row at a time in the loop (if using loop)

**If `log_probs` is kept:**
- PyTorch can use stored `log_probs` directly
- No recomputation needed
- But `log_probs` takes up memory during backward

### Memory During Backward Pass

**Batch Processing:**
```python
# Forward:
logits = ...           # 730 MB
log_probs = ...        # 730 MB (kept for backward)
per_token_logps = ...  # 7 KB

# Backward:
# logits: 730 MB (needed)
# log_probs: 730 MB (can use stored value)
# Gradients: ~730 MB
# Peak: ~2.19 GB
```

**Loop Processing:**
```python
# Forward:
logits = ...           # 730 MB
# Process one row at a time, delete log_probs

# Backward:
# logits: 730 MB (needed for recomputation)
# log_probs: 91 MB (recomputed one row at a time)
# Gradients: ~730 MB
# Peak: ~1.55 GB (still lower!)
```

**Conclusion:** Even during training, the loop reduces peak memory because:
1. Forward peak is lower (one row of log_probs vs all rows)
2. Backward recomputation happens row-by-row (one row at a time)
3. Total peak memory is still reduced compared to batch processing

## Key Takeaways

1. **Loop reduces peak memory** by ~44% during inference, ~30% during training
2. **Trade-off**: ~1-2x slower but prevents OOM errors
3. **During training**: Gradients require keeping `logits`, but loop still helps by:
   - Reducing forward peak memory
   - Enabling row-by-row recomputation during backward
   - Working well with gradient checkpointing
4. **During inference**: Maximum benefit since no gradients needed
5. **Each row processed independently** - no dependencies between samples
6. **Memory is freed after each iteration** - reduces peak usage
7. **Chunked processing** offers a middle ground if needed
8. **Most beneficial**: Large vocabularies (50k+ tokens) where log_probs is huge

## When Memory Isn't an Issue

If you have plenty of GPU memory and want maximum speed:

```python
def get_per_token_logps_fast(logits, input_ids):
    """Fast batch processing - use when memory is not a concern."""
    log_probs = logits.log_softmax(dim=-1)
    return torch.gather(
        log_probs,
        dim=-1,
        index=input_ids.unsqueeze(-1)
    ).squeeze(-1)
```

But for production training of large models, the loop-based approach is the safer choice.

## Further Reading

- [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [Memory-Efficient Training Techniques](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- See also: `tutorial/log_softmax_stability.md` and `tutorial/torch_gather_explained.md`

