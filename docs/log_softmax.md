# Log-Softmax Numerical Stability Tutorial

## Overview

This tutorial explains why we use a numerically stable implementation of `log_softmax` and how it works. This is critical for training language models where logits can have large values that would cause numerical overflow in naive implementations.

## The Problem: Numerical Instability

The mathematical definition of log-softmax is:

```
log_softmax(x_i) = log(exp(x_i) / Σ_j exp(x_j))
                 = x_i - log(Σ_j exp(x_j))
```

However, directly computing `exp(x)` for large values of `x` causes **numerical overflow**. For example:
- In float32/bfloat16, `exp(88.7) ≈ 3.4e38` (near the maximum representable value)
- In float16, `exp(11)` already overflows to `inf`

When training large language models, logits can easily exceed these thresholds, causing:
- `inf` values in probabilities
- `NaN` values propagating through gradients
- Training instability and crashes

## The Solution: Log-Sum-Exp Trick

The numerically stable implementation uses the **log-sum-exp trick**:

```python
# Numerically stable log_softmax implementation
max_input = max(input, dim=dim)           # Find maximum along dimension
tmp = exp(input - max_input)              # Subtract max before exp
logsum = log(sum(tmp, dim=dim))           # Log of sum of exponentials
output = input - max_input - logsum       # Final log-softmax
```

### Why This Works

The key insight is that we can subtract the maximum value before exponentiating without changing the mathematical result:

```
exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
= exp(x_i) / exp(max(x)) / (Σ_j exp(x_j) / exp(max(x)))
= exp(x_i) / Σ_j exp(x_j)
```

**Benefits:**
1. **Prevents overflow**: `exp(x_i - max(x))` is always ≤ 1, so it can't overflow
2. **Prevents underflow**: Values are shifted to a reasonable range
3. **Mathematically equivalent**: Produces the same result as the naive implementation

### Step-by-Step Example

Let's trace through an example with logits `[100, 102, 98]`:

```python
# Naive (UNSTABLE - would overflow):
exp(100) + exp(102) + exp(98)  # exp(102) ≈ 2.7e44 → OVERFLOW!

# Stable implementation:
max_input = 102
tmp = [exp(100-102), exp(102-102), exp(98-102)]
     = [exp(-2), exp(0), exp(-4)]
     = [0.135, 1.0, 0.018]  # All values ≤ 1, no overflow!
logsum = log(0.135 + 1.0 + 0.018) = log(1.153) ≈ 0.142
output = [100-102-0.142, 102-102-0.142, 98-102-0.142]
       = [-2.142, -0.142, -4.142]
```

## PyTorch Implementation

PyTorch's `torch.log_softmax()` uses exactly this stable implementation internally. When you call:

```python
log_probs = logits.log_softmax(dim=-1)
```

PyTorch internally performs:
1. `max_val = logits.max(dim=-1, keepdim=True)[0]`
2. `shifted = logits - max_val`
3. `exp_shifted = shifted.exp()`
4. `logsum = exp_shifted.sum(dim=-1, keepdim=True).log()`
5. `result = shifted - logsum`

This is equivalent to the manual implementation shown above.

## Usage in This Codebase

In `grpo_ref_split.py` and `ref_server.py`, we use PyTorch's built-in `log_softmax`:

```python
def get_per_token_logps(logits, input_ids):
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)  # Numerically stable!
        token_log_prob = torch.gather(log_probs, dim=1, 
                                     index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)
```

### Why This Matters for GRPO

In Group Relative Policy Optimization (GRPO), we compute:
- **Per-token log probabilities** for the training model
- **Per-token log probabilities** for the reference model
- **KL divergence** between them: `KL = exp(ref_logp - train_logp) - (ref_logp - train_logp) - 1`

If log probabilities contain `inf` or `NaN`:
- KL divergence becomes undefined
- Loss becomes `NaN`
- Training crashes

Using the stable `log_softmax` ensures:
- All log probabilities are finite
- Gradients remain stable
- Training proceeds smoothly

## Manual Implementation (for reference)

If you need to implement log_softmax manually (e.g., for custom CUDA kernels), here's the stable version:

```python
def stable_log_softmax(x, dim=-1):
    """
    Numerically stable log_softmax implementation.
    
    Args:
        x: Input tensor of shape (..., vocab_size)
        dim: Dimension along which to compute softmax
    
    Returns:
        log_softmax(x) of same shape as x
    """
    # Step 1: Find maximum along the specified dimension
    max_input = x.max(dim=dim, keepdim=True)[0]
    
    # Step 2: Subtract max before exponentiating (prevents overflow)
    tmp = (x - max_input).exp()
    
    # Step 3: Compute log of sum
    logsum = tmp.sum(dim=dim, keepdim=True).log()
    
    # Step 4: Final result
    output = x - max_input - logsum
    
    return output
```

## Comparison: Stable vs Unstable

```python
import torch

# Large logits that would cause overflow
logits = torch.tensor([100.0, 102.0, 98.0])

# Unstable (naive) implementation
def naive_log_softmax(x):
    exp_x = x.exp()  # exp(102) → inf in float32!
    return x - exp_x.sum().log()

# Stable implementation (PyTorch's default)
stable_result = logits.log_softmax(dim=-1)
# Result: tensor([-2.142, -0.142, -4.142])

# Compare
print("Stable:", stable_result)
# Stable: tensor([-2.142, -0.142, -4.142])

# Naive would produce: tensor([nan, nan, nan]) or crash
```

## Key Takeaways

1. **Always use `log_softmax` instead of `log(softmax())`** - it's more stable and efficient
2. **PyTorch's implementation is stable** - you don't need to implement it manually
3. **The log-sum-exp trick** prevents overflow by subtracting the maximum before exponentiating
4. **This is critical for large language models** where logits can have extreme values
5. **Stability matters for GRPO** - unstable log probabilities cause NaN in KL divergence and loss

## Further Reading

- [PyTorch Documentation: log_softmax](https://pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html)
- [Numerical Stability in Deep Learning](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/numerical-stability.html)
- [Log-Sum-Exp Trick](https://en.wikipedia.org/wiki/LogSumExp)



