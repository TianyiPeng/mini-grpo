# torch.stack Tutorial: Combining Tensors Along a New Dimension

## Overview

`torch.stack` is a PyTorch function that combines a sequence of tensors along a **new dimension**. Unlike `torch.cat` (concatenate), which joins tensors along an existing dimension, `torch.stack` creates a new dimension and stacks tensors along it. This is particularly useful when you have a list of tensors with the same shape and want to create a batched tensor.

## Basic Syntax

```python
torch.stack(tensors, dim=0, *, out=None)
```

**Parameters:**
- `tensors`: Sequence of tensors to stack (list, tuple, etc.)
- `dim`: Dimension along which to stack (default: 0)
- `out`: Optional output tensor

**Returns:** A new tensor with one more dimension than the input tensors

## Key Difference: stack vs cat

### torch.cat (Concatenate)

`torch.cat` joins tensors along an **existing dimension**:

```python
import torch

a = torch.tensor([1, 2, 3])  # Shape: (3,)
b = torch.tensor([4, 5, 6])  # Shape: (3,)

# Concatenate along existing dimension 0
result = torch.cat([a, b], dim=0)  # Shape: (6,)
# [1, 2, 3, 4, 5, 6]
```

### torch.stack (Stack)

`torch.stack` creates a **new dimension** and stacks along it:

```python
a = torch.tensor([1, 2, 3])  # Shape: (3,)
b = torch.tensor([4, 5, 6])  # Shape: (3,)

# Stack along new dimension 0
result = torch.stack([a, b], dim=0)  # Shape: (2, 3)
# [[1, 2, 3],
#  [4, 5, 6]]

# Stack along new dimension -1 (last dimension)
result = torch.stack([a, b], dim=-1)  # Shape: (3, 2)
# [[1, 4],
#  [2, 5],
#  [3, 6]]
```

## How It Works

`torch.stack` requires:
1. **All input tensors must have the same shape**
2. **All input tensors must be on the same device**
3. **All input tensors must have the same dtype**

The output shape is determined by:
- Input shape: `(S1, S2, ..., Sn)`
- Number of tensors: `N`
- Stack dimension: `dim`
- Output shape: `(S1, ..., Sdim, N, Sdim+1, ..., Sn)` (if `dim=0`, it's `(N, S1, S2, ..., Sn)`)

## Simple Examples

### Example 1: Stacking 1D Tensors

```python
import torch

# Create three 1D tensors
a = torch.tensor([1, 2, 3])      # Shape: (3,)
b = torch.tensor([4, 5, 6])      # Shape: (3,)
c = torch.tensor([7, 8, 9])      # Shape: (3,)

# Stack along dimension 0 (creates batch dimension)
result = torch.stack([a, b, c], dim=0)  # Shape: (3, 3)
# [[1, 2, 3],
#  [4, 5, 6],
#  [7, 8, 9]]

# Stack along dimension -1 (creates feature dimension)
result = torch.stack([a, b, c], dim=-1)  # Shape: (3, 3)
# [[1, 4, 7],
#  [2, 5, 8],
#  [3, 6, 9]]
```

### Example 2: Stacking 2D Tensors

```python
# Create two 2D tensors (matrices)
a = torch.tensor([[1, 2], [3, 4]])  # Shape: (2, 2)
b = torch.tensor([[5, 6], [7, 8]])  # Shape: (2, 2)

# Stack along dimension 0
result = torch.stack([a, b], dim=0)  # Shape: (2, 2, 2)
# [[[1, 2],
#   [3, 4]],
#  [[5, 6],
#   [7, 8]]]

# Stack along dimension 1
result = torch.stack([a, b], dim=1)  # Shape: (2, 2, 2)
# [[[1, 2],
#   [5, 6]],
#  [[3, 4],
#   [7, 8]]]

# Stack along dimension 2
result = torch.stack([a, b], dim=2)  # Shape: (2, 2, 2)
# [[[1, 5],
#   [2, 6]],
#  [[3, 7],
#   [4, 8]]]
```

## Use Case: Combining Per-Token Log Probabilities

In `get_per_token_logps`, `torch.stack` is used to combine log probabilities from multiple samples in a batch:

```python
def get_per_token_logps(logits, input_ids):
    """
    Compute per-token log probabilities for a batch.
    
    Args:
        logits: (batch_size, seq_len, vocab_size)
        input_ids: (batch_size, seq_len)
    
    Returns:
        per_token_logps: (batch_size, seq_len)
    """
    per_token_logps = []  # List to collect results
    
    # Process each sample in the batch
    for logits_row, input_ids_row in zip(logits, input_ids):
        # logits_row: (seq_len, vocab_size)
        # input_ids_row: (seq_len,)
        
        log_probs = logits_row.log_softmax(dim=-1)  # (seq_len, vocab_size)
        token_log_prob = torch.gather(
            log_probs, 
            dim=1, 
            index=input_ids_row.unsqueeze(1)
        ).squeeze(1)  # (seq_len,)
        
        per_token_logps.append(token_log_prob)  # Add to list
    
    # Stack all samples into a batch tensor
    result = torch.stack(per_token_logps, dim=0)  # (batch_size, seq_len)
    return result
```

### Step-by-Step Visualization

```python
# Input:
logits = torch.randn(3, 5, 10)  # 3 samples, 5 tokens, 10 vocab size
input_ids = torch.randint(0, 10, (3, 5))  # 3 samples, 5 tokens

# After processing each sample:
per_token_logps = [
    torch.tensor([-0.5, -1.2, -0.8, -1.5, -0.9]),  # Sample 0: (5,)
    torch.tensor([-1.1, -0.7, -1.3, -0.6, -1.0]),  # Sample 1: (5,)
    torch.tensor([-0.9, -1.4, -0.5, -1.1, -0.7])   # Sample 2: (5,)
]

# After torch.stack:
result = torch.stack(per_token_logps, dim=0)  # (3, 5)
# [[-0.5, -1.2, -0.8, -1.5, -0.9],
#  [-1.1, -0.7, -1.3, -0.6, -1.0],
#  [-0.9, -1.4, -0.5, -1.1, -0.7]]
```

## Why Use torch.stack Instead of torch.cat?

### In get_per_token_logps

We use `torch.stack` because:
1. **Creates batch dimension**: We want shape `(batch_size, seq_len)`, not `(batch_size * seq_len,)`
2. **Preserves structure**: Each sample's log probabilities remain separate
3. **Enables batch operations**: Can easily index by sample: `result[0]` gets first sample

**If we used `torch.cat` instead:**

```python
# Wrong approach:
result = torch.cat(per_token_logps, dim=0)  # Shape: (batch_size * seq_len,)
# Flattens everything into one long vector!

# Correct approach:
result = torch.stack(per_token_logps, dim=0)  # Shape: (batch_size, seq_len)
# Keeps batch structure intact
```

## Common Patterns

### Pattern 1: Creating Batch from List

```python
# Common pattern: collect results in a list, then stack
results = []
for i in range(batch_size):
    result = process_sample(samples[i])
    results.append(result)

# Stack into batch tensor
batch_result = torch.stack(results, dim=0)  # (batch_size, ...)
```

### Pattern 2: Stacking Along Different Dimensions

```python
# Stack along first dimension (batch dimension)
batch = torch.stack(tensors, dim=0)  # (N, ...)

# Stack along last dimension (feature dimension)
features = torch.stack(tensors, dim=-1)  # (..., N)

# Stack along middle dimension
stacked = torch.stack(tensors, dim=1)  # (..., N, ...)
```

### Pattern 3: Stacking Multiple Views

```python
# Stack different views of the same data
x = torch.randn(10, 20)  # (height, width)

# Stack along channel dimension
channels = [
    x,           # Red channel
    x.roll(1, 0), # Green channel (shifted)
    x.roll(1, 1)  # Blue channel (shifted)
]
rgb = torch.stack(channels, dim=0)  # (3, 10, 20)
```

## Important Requirements

### 1. All Tensors Must Have Same Shape

```python
# ✅ Correct - all same shape
a = torch.randn(3, 4)
b = torch.randn(3, 4)
c = torch.randn(3, 4)
result = torch.stack([a, b, c], dim=0)  # (3, 3, 4)

# ❌ Wrong - different shapes
a = torch.randn(3, 4)
b = torch.randn(3, 5)  # Different width!
# result = torch.stack([a, b], dim=0)  # Error!
```

### 2. All Tensors Must Be on Same Device

```python
# ✅ Correct - same device
a = torch.randn(3, 4).cuda()
b = torch.randn(3, 4).cuda()
result = torch.stack([a, b], dim=0)

# ❌ Wrong - different devices
a = torch.randn(3, 4).cuda()
b = torch.randn(3, 4).cpu()
# result = torch.stack([a, b], dim=0)  # Error!
```

### 3. All Tensors Must Have Same Dtype

```python
# ✅ Correct - same dtype
a = torch.randn(3, 4, dtype=torch.float32)
b = torch.randn(3, 4, dtype=torch.float32)
result = torch.stack([a, b], dim=0)

# ❌ Wrong - different dtypes
a = torch.randn(3, 4, dtype=torch.float32)
b = torch.randn(3, 4, dtype=torch.float64)
# result = torch.stack([a, b], dim=0)  # Error!
```

## Memory Considerations

### Memory Usage

`torch.stack` creates a **new tensor** that contains copies of all input tensors:

```python
a = torch.randn(100, 100)  # ~40 KB
b = torch.randn(100, 100)  # ~40 KB
c = torch.randn(100, 100)  # ~40 KB

# Stack creates new tensor with all data
result = torch.stack([a, b, c], dim=0)  # ~120 KB
# Original tensors still exist in memory!
```

**Memory impact:**
- Input: `N` tensors of shape `(S1, S2, ...)`
- Output: One tensor of shape `(N, S1, S2, ...)`
- Memory: `N × size_of_one_tensor` (all data is copied)

### When to Delete Input Tensors

After stacking, you can delete the original list to free memory:

```python
per_token_logps = []
for ...:
    token_log_prob = ...
    per_token_logps.append(token_log_prob)

result = torch.stack(per_token_logps, dim=0)
del per_token_logps  # Free the list (tensors are copied to result)
```

## Comparison with Alternatives

### Alternative 1: Pre-allocate and Fill

```python
# Instead of stack, pre-allocate tensor
batch_size, seq_len = 8, 100
result = torch.zeros(batch_size, seq_len)

for i in range(batch_size):
    result[i] = process_sample(samples[i])

# Pros: No intermediate list, potentially more memory efficient
# Cons: Requires knowing output shape in advance
```

### Alternative 2: List Comprehension + Stack

```python
# More Pythonic
result = torch.stack([
    process_sample(sample) 
    for sample in samples
], dim=0)

# Pros: Concise
# Cons: Less control over memory management
```

### Alternative 3: Using torch.cat (Wrong for This Use Case)

```python
# This would flatten the batch dimension!
result = torch.cat(per_token_logps, dim=0)  # (batch_size * seq_len,)
# Wrong shape for batch processing!
```

## Performance Considerations

1. **Stacking is fast**: Single operation, highly optimized
2. **Memory overhead**: Creates copies of all tensors
3. **Better than loops**: More efficient than manual indexing
4. **GPU-friendly**: Efficiently runs on GPU

## Common Mistakes

### Mistake 1: Forgetting dim Parameter

```python
# Default dim=0 might not be what you want
result = torch.stack(tensors)  # Stacks along dim=0
# Better to be explicit:
result = torch.stack(tensors, dim=0)
```

### Mistake 2: Mixing Shapes

```python
# This will fail!
a = torch.randn(3, 4)
b = torch.randn(3, 5)  # Different shape
# result = torch.stack([a, b], dim=0)  # RuntimeError!
```

### Mistake 3: Empty List

```python
# This will fail!
result = torch.stack([], dim=0)  # RuntimeError: empty list
# Check list is not empty first:
if per_token_logps:
    result = torch.stack(per_token_logps, dim=0)
```

## Key Takeaways

1. **`torch.stack` creates a new dimension** and stacks tensors along it
2. **All input tensors must have the same shape, device, and dtype**
3. **Output shape**: `(N, S1, S2, ...)` where `N` is number of tensors
4. **Useful for**: Combining results from loops into batched tensors
5. **Memory**: Creates copies of all input tensors
6. **Common use case**: Converting list of per-sample results into batch tensor

## Further Reading

- [PyTorch Documentation: torch.stack](https://pytorch.org/docs/stable/generated/torch.stack.html)
- [PyTorch Documentation: torch.cat](https://pytorch.org/docs/stable/generated/torch.cat.html) (related function)
- See also: `tutorial/memory_efficient_logprobs.md` (uses `torch.stack` to combine batch results)



