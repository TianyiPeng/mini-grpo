# torch.gather Tutorial: Extracting Values by Index

## Overview

`torch.gather` is a powerful PyTorch function that allows you to extract values from a tensor using indices along a specified dimension. It's particularly useful for extracting per-token log probabilities in language model training, where you need to select the probability of the actual token that was generated at each position.

## Basic Syntax

```python
torch.gather(input, dim, index, *, sparse_grad=False, out=None)
```

**Parameters:**
- `input`: Source tensor to gather values from
- `dim`: Dimension along which to gather (0, 1, 2, ...)
- `index`: Tensor of indices specifying which values to extract
- `sparse_grad`: Whether to use sparse gradients (advanced)

**Returns:** A tensor with the same shape as `index`, containing values gathered from `input`

## How It Works

`torch.gather` extracts values from `input` based on indices in `index` along dimension `dim`. The operation can be thought of as:

```python
# For each position in the output tensor:
output[i][j][k] = input[i][j][index[i][j][k]]  # if dim=2
```

**Key Rules:**
1. `index` must have the same number of dimensions as `input`
2. All dimensions except `dim` must match between `input` and `index`
3. Values in `index` must be valid indices for dimension `dim` in `input`
4. Output shape matches `index` shape

## Simple Example

```python
import torch

# Create a 2D tensor (3 rows, 4 columns)
input = torch.tensor([
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [90, 100, 110, 120]
])
# Shape: (3, 4)

# Gather along dimension 1 (columns)
index = torch.tensor([
    [0, 2],  # Row 0: get columns 0 and 2
    [1, 3],  # Row 1: get columns 1 and 3
    [2, 0]   # Row 2: get columns 2 and 0
])
# Shape: (3, 2)

result = torch.gather(input, dim=1, index=index)
# Result:
# [[10, 30],   # Row 0: input[0][0]=10, input[0][2]=30
#  [60, 80],   # Row 1: input[1][1]=60, input[1][3]=80
#  [110, 90]]  # Row 2: input[2][2]=110, input[2][0]=90
# Shape: (3, 2) - matches index shape
```

## Use Case: Extracting Per-Token Log Probabilities

In language model training, we need to extract the log probability of the **actual token** that was generated at each position. This is where `torch.gather` shines.

### The Problem

After computing log probabilities for all tokens in the vocabulary at each position:
- `log_probs`: shape `(seq_len, vocab_size)` - log probability of each vocabulary token at each position
- `input_ids`: shape `(seq_len,)` - the actual token ID at each position

We want: log probability of the actual token at each position → shape `(seq_len,)`

### The Solution with torch.gather

```python
import torch

# Example: sequence length 5, vocabulary size 10
seq_len = 5
vocab_size = 10

# Log probabilities: for each position, probability of each vocab token
log_probs = torch.randn(seq_len, vocab_size)
# Shape: (5, 10)
# log_probs[i, j] = log probability of token j at position i

# Actual token IDs in the sequence
input_ids = torch.tensor([3, 7, 1, 9, 2])
# Shape: (5,)
# Position 0 has token 3, position 1 has token 7, etc.

# Extract log probability of actual token at each position
# We need to gather along dimension 1 (vocab dimension)
# But input_ids needs to be reshaped to match log_probs dimensions

# Step 1: Add dimension to match log_probs
index = input_ids.unsqueeze(1)  # Shape: (5, 1)
# [[3],
#  [7],
#  [1],
#  [9],
#  [2]]

# Step 2: Gather along dimension 1
token_log_probs = torch.gather(log_probs, dim=1, index=index)
# Shape: (5, 1)
# [[log_probs[0, 3]],  # Log prob of token 3 at position 0
#  [log_probs[1, 7]],  # Log prob of token 7 at position 1
#  [log_probs[2, 1]],  # Log prob of token 1 at position 2
#  [log_probs[3, 9]],  # Log prob of token 9 at position 3
#  [log_probs[4, 2]]]  # Log prob of token 2 at position 4

# Step 3: Remove the extra dimension
token_log_probs = token_log_probs.squeeze(1)
# Shape: (5,)
# [log_probs[0, 3], log_probs[1, 7], log_probs[2, 1], log_probs[3, 9], log_probs[4, 2]]
```

### Complete Example from Codebase

In `grpo_ref_split.py` and `ref_server.py`, this is implemented as:

```python
def get_per_token_logps(logits, input_ids):
    """
    Extract log probability of actual tokens from logits.
    
    Args:
        logits: (batch_size, seq_len, vocab_size) - raw model outputs
        input_ids: (batch_size, seq_len) - actual token IDs
    
    Returns:
        per_token_logps: (batch_size, seq_len) - log prob of actual token at each position
    """
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        # logits_row: (seq_len, vocab_size)
        # input_ids_row: (seq_len,)
        
        # Compute log probabilities for all tokens
        log_probs = logits_row.log_softmax(dim=-1)  # (seq_len, vocab_size)
        
        # Extract log prob of actual token at each position
        # input_ids_row.unsqueeze(1): (seq_len, 1) - indices for gathering
        token_log_prob = torch.gather(
            log_probs, 
            dim=1,                    # Gather along vocab dimension
            index=input_ids_row.unsqueeze(1)  # (seq_len, 1)
        ).squeeze(1)                  # Remove extra dim: (seq_len,)
        
        per_token_logps.append(token_log_prob)
    
    return torch.stack(per_token_logps)  # (batch_size, seq_len)
```

## Visual Explanation

Let's trace through a concrete example:

```python
# Position 0: log_probs[0] = [-2.1, -1.5, -3.2, -0.8, -2.9, ...]
#            actual token = 3
#            → extract log_probs[0, 3] = -0.8

# Position 1: log_probs[1] = [-1.2, -2.8, -1.9, -0.5, -3.1, ...]
#            actual token = 1
#            → extract log_probs[1, 1] = -2.8

# Position 2: log_probs[2] = [-2.5, -1.1, -2.3, -0.9, -1.8, ...]
#            actual token = 4
#            → extract log_probs[2, 4] = -1.8
```

**Before gather:**
```
log_probs:                    input_ids:
[[-2.1, -1.5, -3.2, -0.8, ...],  [3,
 [-1.2, -2.8, -1.9, -0.5, ...],   1,
 [-2.5, -1.1, -2.3, -0.9, ...]]   4]
```

**After gather:**
```
token_log_probs:
[-0.8,  # log_probs[0, 3]
 -2.8,  # log_probs[1, 1]
 -1.8]  # log_probs[2, 4]
```

## Why Not Use Indexing?

You might wonder: why not just use advanced indexing like `log_probs[range(seq_len), input_ids]`?

**Answer:** You can! But `torch.gather` has advantages:

1. **More explicit**: Makes the dimension being gathered along clear
2. **Gradient-friendly**: Better gradient flow in some cases
3. **Consistent API**: Works well with batched operations
4. **Broadcasting**: Handles broadcasting automatically

**Equivalent using advanced indexing:**
```python
# This also works:
token_log_probs = log_probs[torch.arange(seq_len), input_ids]
# But requires matching dimensions and can be less clear
```

## Common Patterns

### Pattern 1: Gathering along last dimension (most common)

```python
# Input: (batch, seq_len, vocab_size)
# Index: (batch, seq_len)
log_probs = model(input_ids).logits.log_softmax(dim=-1)  # (B, L, V)
token_log_probs = torch.gather(
    log_probs, 
    dim=2,  # or dim=-1
    index=input_ids.unsqueeze(-1)  # (B, L, 1)
).squeeze(-1)  # (B, L)
```

### Pattern 2: Gathering along first dimension

**⚠️ Common Bug Warning:** This pattern is often implemented incorrectly! When gathering along `dim=0`, you **cannot** just use `unsqueeze(1)` like with the last dimension.

```python
# Input: (num_classes, features)
# Index: (batch_size,)
features = torch.randn(10, 128)  # 10 classes, 128 features each
class_indices = torch.tensor([3, 7, 1, 9, 2])  # Which class for each sample

# ❌ WRONG - This is a common bug! Shape mismatch!
# index = class_indices.unsqueeze(1)  # (5, 1)
# result = torch.gather(features, dim=0, index=index)  # ERROR or unexpected behavior!

# ✅ CORRECT - When gathering along dim=0, index must match ALL other dimensions
# features: (10, 128), so index must be (batch_size, 128), not (batch_size, 1)!
selected_features = torch.gather(
    features,
    dim=0,  # Gather along class dimension
    index=class_indices.unsqueeze(1).expand(-1, 128)  # (5, 128) - MUST match features.shape[1:]
)
# Result: (5, 128) - one feature vector per sample
```

**Why the difference?**
- When `dim=1` (last dim): `index` can be `(batch, 1)` - only needs to match non-dim dimensions
- When `dim=0` (first dim): `index` must be `(batch, features)` - **all** non-dim dimensions must match!

**The rule:** `index.shape[i]` must equal `input.shape[i]` for all `i != dim`. When `dim=0`, this means `index.shape[1:]` must match `input.shape[1:]` exactly.

### Pattern 3: Multi-dimensional gathering

```python
# Input: (batch, height, width, channels)
# Index: (batch, height, width) - which channel to select at each pixel
image = torch.randn(2, 32, 32, 3)  # Batch of RGB images
channel_indices = torch.randint(0, 3, (2, 32, 32))  # Random channel per pixel

selected_channels = torch.gather(
    image,
    dim=3,  # Gather along channel dimension
    index=channel_indices.unsqueeze(-1)  # (2, 32, 32, 1)
).squeeze(-1)  # (2, 32, 32)
```

## Important Notes

### 1. Index Bounds Checking

`torch.gather` does **not** check bounds by default. Invalid indices can cause undefined behavior:

```python
log_probs = torch.randn(5, 10)  # vocab_size = 10
input_ids = torch.tensor([3, 7, 15, 9, 2])  # 15 is invalid!

# This will work but may give unexpected results
result = torch.gather(log_probs, dim=1, index=input_ids.unsqueeze(1))
```

**Solution:** Clamp indices before gathering:
```python
vocab_size = log_probs.shape[-1]
input_ids = torch.clamp(input_ids, min=0, max=vocab_size - 1)
```

### 3. Dimension Matching Examples

The `index` tensor must have the same number of dimensions as `input`, and **all dimensions except `dim` must match exactly**:

```python
# ✅ Correct - Gathering along last dimension (dim=1)
log_probs = torch.randn(5, 10)      # (seq_len, vocab_size)
index = torch.tensor([[3], [7], [1], [9], [2]])  # (seq_len, 1)
result = torch.gather(log_probs, dim=1, index=index)
# index.shape[0] matches log_probs.shape[0] ✓
# index.shape[1] can be anything (determines output size) ✓

# ❌ Wrong - dimensions don't match
log_probs = torch.randn(5, 10)      # (seq_len, vocab_size)
index = torch.tensor([3, 7, 1, 9, 2])  # (seq_len,) - missing dimension!
# result = torch.gather(log_probs, dim=1, index=index)  # Error!

# ✅ Correct - Gathering along first dimension (dim=0)
features = torch.randn(10, 128)     # (num_classes, features)
class_indices = torch.tensor([3, 7, 1, 9, 2])  # (batch_size,)
index = class_indices.unsqueeze(1).expand(-1, 128)  # (batch_size, 128)
result = torch.gather(features, dim=0, index=index)
# index.shape[1] matches features.shape[1] ✓ (must match!)
# index.shape[0] can be anything (determines output size) ✓

# ❌ Wrong - This is the bug! Index doesn't match non-dim dimensions
features = torch.randn(10, 128)     # (num_classes, features)
index = torch.tensor([[3], [7], [1], [9], [2]])  # (batch_size, 1)
# result = torch.gather(features, dim=0, index=index)  # Shape mismatch!
# Error: index.shape[1]=1 doesn't match features.shape[1]=128
```

**Key Insight:** When gathering along `dim=0`, you need to expand the index to match **all other dimensions**. This is different from gathering along the last dimension where you can use `unsqueeze` to add a dimension.

### 4. Output Shape

The output shape always matches the `index` shape:

```python
log_probs = torch.randn(5, 10)           # (seq_len, vocab_size)
index = torch.tensor([[3], [7], [1], [9], [2]])  # (5, 1)
result = torch.gather(log_probs, dim=1, index=index)
# result.shape = (5, 1) - matches index shape!

# After squeeze:
result = result.squeeze(1)  # (5,) - removes dimension 1
```

## Performance Considerations

1. **Vectorization**: `torch.gather` is highly optimized and vectorized
2. **Memory**: Creates a new tensor (doesn't modify input)
3. **Gradients**: Supports automatic differentiation
4. **GPU**: Efficiently runs on GPU

## Comparison with Alternatives

### Alternative 1: Advanced Indexing
```python
# Using advanced indexing
token_log_probs = log_probs[torch.arange(seq_len), input_ids]
# Pros: More concise
# Cons: Less explicit about which dimension, can be confusing
```

### Alternative 2: Loop
```python
# Using a loop (slow!)
token_log_probs = []
for i in range(seq_len):
    token_log_probs.append(log_probs[i, input_ids[i]])
token_log_probs = torch.stack(token_log_probs)
# Pros: Very explicit
# Cons: Slow, not vectorized
```

### Alternative 3: torch.gather (Recommended)
```python
# Using torch.gather
token_log_probs = torch.gather(
    log_probs, 
    dim=1, 
    index=input_ids.unsqueeze(1)
).squeeze(1)
# Pros: Fast, vectorized, explicit, gradient-friendly
# Cons: Requires understanding gather semantics
```

## Key Takeaways

1. **`torch.gather` extracts values** from a tensor using indices along a specified dimension
2. **Output shape matches index shape** - this is crucial for understanding
3. **Common use case**: Extracting per-token log probabilities in language models
4. **Always ensure index dimensions match** input dimensions (except the gather dimension)
5. **Clamp indices** to valid ranges to avoid undefined behavior
6. **More efficient than loops** and more explicit than advanced indexing for this use case

## Further Reading

- [PyTorch Documentation: torch.gather](https://pytorch.org/docs/stable/generated/torch.gather.html)
- [PyTorch Documentation: torch.index_select](https://pytorch.org/docs/stable/generated/torch.index_select.html) (related function)
- [Advanced Indexing in PyTorch](https://pytorch.org/docs/stable/tensor_attributes.html#torch-dtype)

