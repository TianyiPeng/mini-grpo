# Multi-GPU Transfer Best Practices: Serialization on CPU

## The Problem

When working with distributed training across multiple GPUs, tensor serialization and deserialization can cause unexpected issues, especially with `bfloat16` precision.

### The Issue We Encountered

In our GRPO training setup, we observed NaN values appearing when transferring tensors directly between GPUs:

```
BEFORE .to(): 0 NaN/Inf, device=cuda:0, dtype=torch.bfloat16
AFTER .to(): 4096 NaN/Inf, device=cuda:3, dtype=torch.bfloat16  # ❌ Problem!
```

**Root Cause**: Direct cross-GPU transfers (`cuda:0` → `cuda:X`) can corrupt `bfloat16` data due to:
- PyTorch's cross-GPU transfer implementation issues
- PCIe/NVLink hardware limitations
- CUDA driver bugs with `bfloat16` precision

## The Solution: Serialize on CPU

The **correct practice** is to always serialize tensors on CPU, regardless of where they're computed.

### Why CPU Serialization Works

1. **CPU is the safe intermediate**: CPU ↔ GPU transfers are well-tested and stable
2. **No cross-GPU issues**: Avoids direct GPU-to-GPU transfers
3. **Device-agnostic**: Works regardless of which GPU computed the tensor
4. **Compatible with distributed training**: Each rank can load from CPU and move to its own GPU

## Implementation

### ✅ Correct: Serialize on CPU

```python
def tensor_to_bytes(t):
    buffer = io.BytesIO()
    # Force move to CPU before serialization
    torch.save(t.detach().to("cpu"), buffer)
    return buffer.getvalue()

def bytes_to_tensor(b):
    # Force load to CPU, regardless of where tensor was saved
    return torch.load(io.BytesIO(b), map_location="cpu", weights_only=True)
```

### ❌ Incorrect: Serialize on GPU

```python
def tensor_to_bytes(t):
    buffer = io.BytesIO()
    torch.save(t, buffer)  # ❌ Saves device information (e.g., cuda:0)
    return buffer.getvalue()

def bytes_to_tensor(b):
    return torch.load(io.BytesIO(b), weights_only=True)  
    # ❌ Restores to original device (cuda:0), causing cross-GPU transfer issues
```

## How It Works in Distributed Training

### Scenario: Reference Server + Multi-GPU Training

```
Reference Server (cuda:0)
    ↓ compute per_token_logps
    ↓ tensor_to_bytes() → force to CPU
    ↓ HTTP transfer
Training Rank 0 (cuda:0)
    ↓ bytes_to_tensor() → load to CPU
    ↓ .to(cuda:0) → CPU → GPU (safe!)
Training Rank 1 (cuda:1)
    ↓ bytes_to_tensor() → load to CPU
    ↓ .to(cuda:1) → CPU → GPU (safe!)
Training Rank 2 (cuda:2)
    ↓ bytes_to_tensor() → load to CPU
    ↓ .to(cuda:2) → CPU → GPU (safe!)
```

### Key Points

1. **All serialization happens on CPU**: No device information in serialized data
2. **Each rank loads to CPU**: `map_location="cpu"` ensures consistent behavior
3. **Safe GPU transfer**: CPU → GPU transfers are stable and well-tested
4. **No cross-GPU issues**: Avoids problematic `cuda:X` → `cuda:Y` transfers

## Code Example: Complete Implementation

```python
import torch
import io

def tensor_to_bytes(t):
    """Serialize tensor to bytes, forcing CPU storage.
    
    Args:
        t: PyTorch tensor (can be on any device)
    
    Returns:
        bytes: Serialized tensor data
    """
    buffer = io.BytesIO()
    # Critical: Move to CPU before serialization
    # This ensures no device information is stored
    torch.save(t.detach().to("cpu"), buffer)
    return buffer.getvalue()

def bytes_to_tensor(b):
    """Deserialize bytes to tensor, always loading to CPU.
    
    Args:
        b: bytes: Serialized tensor data
    
    Returns:
        torch.Tensor: Tensor on CPU device
    """
    # Critical: map_location="cpu" forces load to CPU
    # regardless of where tensor was originally saved
    return torch.load(io.BytesIO(b), map_location="cpu", weights_only=True)

# Usage in distributed training
def get_batch():
    """Get batch from server, all tensors loaded to CPU."""
    r = requests.get(f"{ref_server}/get").content
    dd = bytes_list_to_list(r)
    data = {}
    data['inputs'] = bytes_to_tensor(dd[1])      # On CPU
    data['rewards'] = bytes_to_tensor(dd[2])    # On CPU
    data['refs'] = bytes_to_tensor(dd[3])        # On CPU
    return data

def training_step(batch):
    """Move batch to correct GPU for this rank."""
    device = torch.device(f"cuda:{torch.distributed.get_rank()}")
    
    # Safe transfer: CPU → GPU (no cross-GPU issues)
    inputs = batch['inputs'].to(device)
    refs = batch['refs'].to(device)
    # ... rest of training ...
```

## Best Practices Summary

### ✅ DO

1. **Always serialize on CPU**: `t.detach().to("cpu")` before `torch.save()`
2. **Always load to CPU**: Use `map_location="cpu"` in `torch.load()`
3. **Transfer CPU → GPU**: Move tensors to target GPU after loading
4. **Use `.detach()`**: Prevents gradient tracking in serialized data

### ❌ DON'T

1. **Don't serialize GPU tensors directly**: Avoids device information in serialized data
2. **Don't rely on default device mapping**: Always specify `map_location`
3. **Don't do direct cross-GPU transfers**: Use CPU as intermediate
4. **Don't serialize with gradients**: Use `.detach()` to save memory

## Why This Matters for `bfloat16`

`bfloat16` has reduced precision compared to `float32`, making it more susceptible to:
- Rounding errors during transfer
- Precision loss in cross-GPU transfers
- Hardware-specific handling differences

By using CPU as an intermediate step:
- CPU ↔ GPU transfers are well-tested and stable
- No precision loss from cross-GPU hardware differences
- Consistent behavior across different GPU architectures

## Performance Considerations

### Overhead

- **CPU serialization**: Minimal overhead (tensors are already computed)
- **CPU → GPU transfer**: Standard operation, well-optimized
- **Total overhead**: Negligible compared to training time

### Benefits

- **Reliability**: Eliminates NaN/Inf corruption issues
- **Compatibility**: Works across different GPU architectures
- **Debugging**: Easier to debug when all data starts on CPU
- **Portability**: Serialized data can be loaded on any device

## Related Issues

This pattern also helps with:
- **Model checkpointing**: Save models on CPU for portability
- **Data loading**: Load datasets to CPU, then move to GPU
- **Distributed inference**: Share results across ranks via CPU
- **Mixed precision training**: Stable transfers for `bfloat16`/`float16`

## Conclusion

**Always serialize tensors on CPU** when working with distributed training or multi-GPU setups. This simple practice:
- Prevents cross-GPU transfer issues
- Ensures data integrity (no NaN/Inf corruption)
- Works reliably with `bfloat16` precision
- Simplifies debugging and portability

The small overhead of CPU serialization is far outweighed by the reliability and correctness benefits.



