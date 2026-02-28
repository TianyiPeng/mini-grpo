# MiniGRPO Implementation Guide

## Overview

This document provides a walkthrough of the MiniGRPO implementation, explaining how each component works together.

## Project Structure

```
miniGRPO/
├── src/
│   └── minigrpo.py      # Main training script
├── docs/                 # Documentation
└── requirements.txt      # Dependencies
```

## Key Components

### 1. Model Setup

```python
# Load tokenizer and models
tokenizer = AutoTokenizer.from_pretrained(os.getenv("HF_MODEL_ID"))
model = AutoModelForCausalLM.from_pretrained(os.getenv("HF_MODEL_ID"), dtype=torch.bfloat16)
ref_model = AutoModelForCausalLM.from_pretrained(os.getenv("HF_MODEL_ID"), dtype=torch.bfloat16)

# Move both models to the same device
model = model.to(device)
ref_model = ref_model.to(device)
ref_model.eval()
ref_model.requires_grad_(False)
```

**Key points:**
- Both models start from the same checkpoint
- Reference model is frozen (no gradients)
- Both models are on the same GPU (single-GPU setup)

### 2. Per-Token Log Probability Extraction

```python
def get_per_token_logps(logits, input_ids):
    """
    Compute per-token log probabilities using memory-efficient loop.
    """
    per_token_logps = []  # Use a loop to reduce memory peak
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)
```

**Why the loop?**
- Reduces peak memory by ~44% during inference
- Processes one sample at a time
- See `memory_efficiency.md` for details

### 3. Batch Generation

```python
def generate_batch():
    """Generate a training batch."""
    # 1. Sample prompts
    inputs = random.sample(QAs, 1)
    
    # 2. Generate answers
    prompt_inputs, output_ids, rewards, answers = gen_samples(inputs)
    
    # 3. Merge prompt and output
    merged_ids = torch.cat([Qrep, output_ids], dim=1).to(device)
    
    # 4. Compute reference log probs (on same GPU!)
    with torch.inference_mode():
        ref_logits = ref_model(merged_ids).logits
        ref_per_token_logps = get_per_token_logps(ref_logits[:, :-1, :], merged_ids[:, 1:])
    
    # 5. Compute generation log probs (if needed)
    if compute_gen_logps:
        with torch.inference_mode():
            gen_logits = model(merged_ids).logits
            gen_logps = get_per_token_logps(gen_logits[:, :-1, :], merged_ids[:, 1:])
    
    return batch
```

**Key differences from multi-GPU version:**
- No HTTP server needed
- Reference model runs directly on GPU
- No serialization/deserialization
- Simpler and faster

### 4. GRPO Loss Computation

```python
def GRPO_step(batch):
    """Compute GRPO loss for a batch."""
    # 1. Forward pass
    logits = model(inputs).logits
    per_token_logps = get_per_token_logps(logits[:, :-1, :], inputs[:, 1:])
    
    # 2. KL divergence penalty
    diff = ref_per_token_logps - per_token_logps
    per_token_kl = torch.exp(diff) - diff - 1  # Schulman's estimator
    
    # 3. Policy gradient term (PPO-style)
    if 'gen_logps' in batch:
        ratio = torch.exp(per_token_logps - batch['gen_logps'])
        clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
    
    # 4. Combine terms
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    
    return loss
```

**See `grpo_objective.md` for detailed explanation.**

### 5. Training Loop

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

for step in progress:
    # Generate batch
    batch = generate_batch()
    while batch is None:
        batch = generate_batch()
    
    # Training step
    loss = GRPO_step(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Save checkpoint
    if step % save_steps == 0:
        model.save_pretrained(f"./step_{step}")
        tokenizer.save_pretrained(f"./step_{step}")
```

**Simplified compared to multi-GPU:**
- No DeepSpeed
- No distributed training
- Standard PyTorch optimizer
- Simpler checkpointing

## Differences from Multi-GPU Version

| Aspect | Multi-GPU (`grpo_ref_split.py`) | Single-GPU (`minigrpo.py`) |
|--------|--------------------------------|----------------------------|
| **Reference Model** | Separate server process | Same process, same GPU |
| **Communication** | HTTP + serialization | Direct GPU memory |
| **Distributed** | DeepSpeed + ZeRO | Standard PyTorch |
| **Complexity** | High (server, sync, etc.) | Low (single script) |
| **Memory** | Split across GPUs | Single GPU |
| **Speed** | Faster (parallel) | Slower (sequential) |

## Memory Considerations

### Single GPU Setup

Both models on the same GPU:
- Training model: ~3GB (1.5B params in bfloat16)
- Reference model: ~3GB (frozen, no gradients)
- Activations: ~2GB
- **Total: ~8GB** (fits on 16GB+ GPUs)

### Memory Optimizations

1. **Loop-based log probs**: Reduces peak memory
2. **Inference mode**: Reference model uses `torch.inference_mode()`
3. **Gradient checkpointing**: Can be added if needed

## Hyperparameters

```python
beta = 0.04              # KL penalty coefficient
num_pre_Q = 8           # Samples per prompt
all_steps = 1000        # Training steps
max_prompt_length = 400 # Max prompt tokens
save_steps = 200        # Checkpoint frequency
compute_gen_logps = True # Use importance sampling
clip_param = 0.2        # PPO clipping parameter
```

## Running the Code

```bash
# Set environment variable
export HF_MODEL_ID="Qwen/Qwen2.5-1.5B"

# Run training
python src/minigrpo.py
```

## Debugging Tips

1. **Check device placement**:
   ```python
   print(model.device)
   print(ref_model.device)
   ```

2. **Monitor memory**:
   ```python
   print(torch.cuda.memory_allocated() / 1e9, "GB")
   ```

3. **Check for NaN**:
   ```python
   if torch.isnan(loss):
       print("NaN detected!")
   ```

## Next Steps

- See `grpo_objective.md` for objective function details
- See `log_softmax.md` for numerical stability
- See `torch_gather.md` for per-token extraction
- See `memory_efficiency.md` for memory optimizations


