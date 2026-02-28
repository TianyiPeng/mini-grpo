# MiniGRPO Documentation

Welcome to the MiniGRPO documentation! This project provides a simplified, single-GPU implementation of Group Relative Policy Optimization (GRPO) for training language models.

## What is GRPO?

Group Relative Policy Optimization (GRPO) is a reinforcement learning algorithm designed for training language models. It combines:

- **Policy gradient methods** with PPO-style clipping
- **KL divergence penalties** to prevent policy collapse
- **Group-relative advantages** for more stable training

## Project Structure

```
miniGRPO/
├── minigrpo.py          # Single-GPU GRPO implementation
├── docs/                 # Documentation
│   ├── index.md
│   ├── grpo_objective.md
│   ├── log_softmax.md
│   └── ...
└── mkdocs.yml           # MkDocs configuration
```

## Key Features

- ✅ **Single GPU**: Both reference and training models on the same device
- ✅ **Simplified**: No distributed training complexity
- ✅ **Educational**: Well-documented with detailed explanations
- ✅ **Complete**: Full GRPO implementation with all key components

## Quick Start

```bash
# Install dependencies using uv
uv sync

# Run training (uses Qwen2.5-1.5B by default)
uv run python minigrpo.py
```

## Documentation Sections

### GRPO Algorithm
- [Objective Function](grpo_objective.md) - Mathematical formulation and KL divergence
- [Implementation](implementation.md) - Code walkthrough

### PyTorch Concepts
- [log_softmax](log_softmax.md) - Numerical stability and implementation
- [torch.gather](torch_gather.md) - Tensor indexing explained
- [torch.stack](torch_stack.md) - Combining tensors
- [Memory Efficiency](memory_efficiency.md) - Optimizing memory usage

### Mathematical Foundations
- [KL Divergence](kl_divergence.md) - Why the specific form is used

### Best Practices
- [Multi-GPU Transfer](multi_gpu_transfer.md) - Serialization best practices

## Learning Resources

This documentation explains:
- Why KL divergence uses `exp(diff) - diff - 1` (Schulman's unbiased estimator)
- How `log_softmax` provides numerical stability
- How `torch.gather` works for per-token log probabilities
- Memory-efficient implementations for large models
- Best practices for multi-GPU training

## Acknowledgments

This project is heavily based on the excellent work from **[simple_GRPO](https://github.com/lsdefine/simple_GRPO)** by the KnowledgeWorks Lab at Fudan University. We extend our gratitude to:

- **Dr. Jiaqing Liang** and **Professor Yanghua Xiao** (project leaders)
- **Jinyi Han**, **Xinyi Wang**, **Zishang Jiang**, and other contributors

The core loss calculation formula is referenced from Hugging Face's [trl](https://github.com/huggingface/trl) library.

## Contributing

This is an educational project. Feel free to:
- Report issues
- Suggest improvements
- Add more explanations

## Citation

If you find this code useful, please consider citing the original simple_GRPO work:

```bibtex
@misc{KW-R1,
  author = {Jiaqing Liang, Jinyi Han, Xinyi Wang, Zishang Jiang, Chengyuan Xiong, Boyu Zhu, Jie Shi, Weijia Li, Tingyun Li, Yanghua Xiao},
  title = {KW-R1: A Simple Implementation of the GRPO Algorithm},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/lsdefine/simple_GRPO}},
}
```


