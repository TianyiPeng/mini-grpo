# KL Divergence: Why This Specific Form?

## Overview

This document explains why GRPO uses the specific form `exp(diff) - diff - 1` for KL divergence estimation, based on Schulman's unbiased estimator.

## The Standard KL Divergence

The KL divergence from distribution $q$ to distribution $p$ is:

$$\text{KL}[q || p] = \sum_x q(x) \log \frac{q(x)}{p(x)} = \mathbb{E}_{x \sim q}\left[\log \frac{q(x)}{p(x)}\right]$$

For continuous distributions, this becomes an integral.

## The Problem with Direct Estimation

In GRPO, we want to estimate:

$$\text{KL}[\pi_{\text{ref}} || \pi_\theta] = \mathbb{E}_{x \sim \pi_{\text{ref}}}\left[\log \frac{\pi_{\text{ref}}(x)}{\pi_\theta(x)}\right]$$

A naive Monte Carlo estimator would be:

$$k_1 = \log \frac{\pi_{\text{ref}}(x)}{\pi_\theta(x)} = \log r$$

where $r = \frac{\pi_{\text{ref}}(x)}{\pi_\theta(x)}$ and $x \sim \pi_{\text{ref}}$.

**Problem**: This estimator has high variance because:
- It can be negative (when $\pi_{\text{ref}}(x) < \pi_\theta(x)$)
- KL divergence is always non-negative
- Half the samples contribute negative values, increasing variance

## Schulman's Unbiased Estimator

John Schulman proposed an unbiased, low-variance estimator in his blog post: [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)

For $\text{KL}[q || p]$ with samples $x \sim q$, the estimator is:

$$k_3 = (r - 1) - \log r$$

where $r = \frac{p(x)}{q(x)}$.

**Key properties**:
1. **Unbiased**: $\mathbb{E}_{x \sim q}[k_3] = \text{KL}[q || p]$
2. **Low variance**: Much lower variance than the naive estimator
3. **Always positive**: Uses the fact that $\log(x) \leq x - 1$ (Jensen's inequality)

## Derivation

The estimator comes from using a control variate. We start with the unbiased estimator:

$$k_1 = -\log r = -\log \frac{p(x)}{q(x)} = \log \frac{q(x)}{p(x)}$$

We can add any term with zero expectation. The natural choice is $r - 1 = \frac{p(x)}{q(x)} - 1$, which has expectation:

$$\mathbb{E}_{x \sim q}\left[\frac{p(x)}{q(x)} - 1\right] = \sum_x q(x) \left(\frac{p(x)}{q(x)} - 1\right) = \sum_x p(x) - q(x) = 1 - 1 = 0$$

So for any $\lambda$, the expression $-\log r + \lambda(r - 1)$ is unbiased. Setting $\lambda = 1$ gives:

$$k_3 = (r - 1) - \log r$$

This is always positive because $\log(x) \leq x - 1$ (with equality only at $x = 1$), so:

$$k_3 = (r - 1) - \log r \geq 0$$

## Application to GRPO

In GRPO, we compute:

```python
diff = ref_per_token_logps - per_token_logps  # log(π_ref/π_θ)
per_token_kl = torch.exp(diff) - diff - 1
```

This computes:
- `diff = log(π_ref/π_θ)`
- `exp(diff) = π_ref/π_θ`
- `per_token_kl = (π_ref/π_θ) - log(π_ref/π_θ) - 1`

This matches Schulman's estimator form: $r - \log(r) - 1$ where $r = \frac{\pi_{\text{ref}}}{\pi_\theta}$.

## Why This Form is Better

### 1. Always Positive

The form `exp(diff) - diff - 1` is always positive because:

$$e^x \geq x + 1 \quad \forall x$$

with equality only at $x = 0$. This ensures:
- Valid divergence measure (KL divergence is always non-negative)
- No negative contributions that increase variance

### 2. Low Variance

Compared to the naive estimator `log(r)`, Schulman's estimator has much lower variance:
- Uses control variate to reduce variance
- Always positive contributions
- Better convergence properties

### 3. Numerically Stable

When implemented with clipping:

```python
diff_clipped = torch.clamp(diff, min=-50.0, max=50.0)
per_token_kl = torch.exp(diff_clipped) - diff_clipped - 1
```

This prevents:
- Overflow: `exp(88) ≈ 1e38` exceeds `bfloat16` maximum
- Underflow: Clipping ensures reasonable values
- NaN/Inf: Controlled range prevents numerical issues

## Visual Comparison

### Naive Estimator
```
log(r) where r = π_ref/π_θ
- Can be negative when π_ref < π_θ
- High variance
- Not guaranteed to be positive
```

### Schulman's Estimator
```
r - log(r) - 1 where r = π_ref/π_θ
- Always positive
- Low variance
- Unbiased
```

## Code Implementation

```python
# Compute KL divergence using Schulman's estimator
diff = ref_per_token_logps - per_token_logps  # log(π_ref/π_θ)
diff_clipped = torch.clamp(diff, min=-50.0, max=50.0)  # Prevent overflow
per_token_kl = torch.exp(diff_clipped) - diff_clipped - 1  # KL estimator

# This is always positive and has low variance
assert (per_token_kl >= 0).all()  # Always true!
```

## Key Takeaways

1. **Schulman's estimator**: `r - log(r) - 1` is unbiased and low-variance
2. **Always positive**: Guaranteed by mathematical property $e^x \geq x + 1$
3. **Numerically stable**: Works well with clipping to prevent overflow
4. **Better than naive**: Much lower variance than `log(r)` estimator
5. **Standard form**: Used in GRPO and other RL algorithms

## References

- [Schulman, J. (2020): Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)
- GRPO paper: Uses this estimator for KL penalty
- See also: `grpo_objective.md` for full GRPO objective explanation


