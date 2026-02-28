"""
MiniGRPO: Single-GPU GRPO Training

A simplified implementation of Group Relative Policy Optimization (GRPO) 
that runs on a single GPU with both reference and training models on the same device.

This implementation is heavily based on the excellent work from:
https://github.com/lsdefine/simple_GRPO

We extend our gratitude to the simple_GRPO team for their open-source contribution.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from math_verify import parse, verify, ExprExtractionConfig
from torch.utils.tensorboard import SummaryWriter
import json, os, re, random, sys, time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# Model configuration
# Change this to use a different model (e.g., "Qwen/Qwen2.5-3B")
MODEL_ID = "Qwen/Qwen2.5-1.5B"

# Hyperparameters
beta = 0.04
num_pre_Q = 8
batch_size = 1  # Number of questions per batch
all_steps = 2000
max_prompt_length = 500   
save_steps = 300
clip_param = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load dataset
from datasets import load_dataset
dataset = load_dataset("openai/gsm8k", "main", split="train")
QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(dataset['question'], dataset['answer'])]

# Load tokenizer and models
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# Ensure pad_token is set (Qwen uses eos_token as pad_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
ref_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)

# Move both models to the same device
model = model.to(device)
ref_model = ref_model.to(device)
ref_model.eval() # this disables random dropout and fixing the batch norm parameters
ref_model.requires_grad_(False) # this disables the gradient calculation since ref_model is only for inference

# Generation config
from transformers import GenerationConfig

generation_config = GenerationConfig(
    max_new_tokens=512,
    do_sample=True, 
    temperature=0.9, 
    num_return_sequences=num_pre_Q,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

### The example in the system prompt is important to make Qwen2.5-1.5B follow the format. 
system_prompt = """You are a helpful assistant. When the user asks a question, you MUST respond using ONLY the following format:

<think>
[Your step-by-step reasoning process here; You might examine your reasoning here before answering.]
</think><answer>
[Your final answer here]
</answer>

Example:
Question: 
What is 5 * 2 + 3?

Answer:
<think>
Let me solve this step by step. First, I need to calculate 5 * 2 = 10. Then I add 3 to get 13. I am checking again and this is correct. So I will provide my final answer. 
</think><answer>
13
</answer>

Now the user asked the following question."""

def get_per_token_logps(logits, input_ids, debug_name=""):
    """
    Compute per-token log probabilities using memory-efficient loop.
    
    Args:
        logits: (B, L-1, V) logits from model
        input_ids: (B, L-1) token IDs
        debug_name: Optional name for debugging
        
    Returns:
        per_token_logps: (B, L-1) log probabilities for each token
    """    
    per_token_logps = []  # Use a loop to reduce memory peak
    for i, (logits_row, input_ids_row) in enumerate(zip(logits, input_ids)):
        # logits_row: (L-1, V)
        # input_ids_row: (L-1,)
        log_probs = logits_row.log_softmax(dim=-1) # log_probs: (L-1, V)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1) # token_log_prob: (L-1,)
        per_token_logps.append(token_log_prob)
    
    result = torch.stack(per_token_logps) # result: (B, L-1)
    return result

def gen_answers(prompts):
    """Generate answers for given prompts. Returns output_text, prompt_text, full_ids, prompt_ids, output_ids."""
    prompt_text = [f"{system_prompt}\n\nQuestion:\n{x}\n\nAnswer:\n" for x in prompts] ## a list of strings (size = batch_size)
    tip_inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=True)
    # prompt_ids: (batch_size, max_prompt_length), left padded
    prompt_ids = tip_inputs["input_ids"]
    prompt_length = prompt_ids.shape[-1]
    if prompt_length > max_prompt_length: # if one prompt is too long, let's sample again
        return [], None, None, None, None
    tip_inputs_device = {k: v.to(device) for k, v in tip_inputs.items()}
    with torch.inference_mode():
        full_ids = model.generate(**tip_inputs_device, generation_config=generation_config)
        # full_ids: (batch_size * num_pre_Q, prompt_len + max_output_len) - includes prompt + completion
    # Clone to allow gradients in training step (tensors created in inference_mode cannot be used in autograd)
    full_ids = full_ids.clone()
    # Extract just the completion part for decoding
    output_ids = full_ids[:, prompt_length:]
    # output_ids: (batch_size * num_pre_Q, max_output_len)
    output_text = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
    # output_text: list of length (batch_size * num_pre_Q)
    return output_text, prompt_text, full_ids, prompt_ids, output_ids

# Reward functions using math_verify for accurate mathematical verification
def reward_correct(item, answer):
    """Check if answer is numerically correct using math_verify."""
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(pattern, answer)  # Find all numbers in answer
    if len(nums) == 0: 
        return -1.0
    lastnum = nums[-1]  # Use the last number in answer to compare with ground truth
    try:
        # Parse the answer number using math_verify
        ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
        # Parse the ground truth
        ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
        # Verify if they match
        return 1.0 if verify(ans, ground_truth) else -1.0
    except:
        return -1.0

def reward_format(item, answer):
    """Check if answer follows the required format."""
    pattern = r"<think>.*?</think><answer>.*?</answer>$"
    return 0.5 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) else -0.5

def gen_samples(inputs):
    """Generate samples and compute rewards."""
    prompts = [x["Q"] for x in inputs]
    output_text, prompt_text, full_ids, prompt_ids, output_ids = gen_answers(prompts)
    # full_ids: (batch_size * num_pre_Q, prompt_len + output_len) - includes prompt + completion
    # prompt_ids: (batch_size, prompt_len)
    # output_text: list of length (batch_size * num_pre_Q)
    # output_ids: (batch_size * num_pre_Q, output_len)
    if len(output_text) == 0: 
        return None, None, None, None, None, None, None, None
    
    # Compute rewards
    rewards, format_rewards, answer_rewards = [], [], []
    for i, inp in enumerate(inputs):
        for a in output_text[i*num_pre_Q:(i+1)*num_pre_Q]:
            answer_reward = reward_correct(inp, a)
            format_reward = reward_format(inp, a)
            format_rewards.append(format_reward)
            answer_rewards.append(answer_reward)
            rewards.append(answer_reward + format_reward)
    # rewards, format_rewards, answer_rewards: lists of length (batch_size * num_pre_Q)
    
    # Use the original full_ids directly (already includes prompt + completion)
    # This avoids potential tokenization differences from decode/re-tokenize
    
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    # rewards_tensor: (batch_size * num_pre_Q,)
    return prompt_ids, output_ids, full_ids, rewards_tensor, output_text, format_rewards, answer_rewards, prompt_text

def generate_batch():
    """Generate a training batch."""
    inputs = random.sample(QAs, batch_size)
    result = gen_samples(inputs)
    if result[0] is None:
        # Debug: no answers generated (likely prompt too long)
        return None
    
    prompt_ids, output_ids, full_ids, rewards, output_text, format_rewards, answer_rewards, prompt_text = result
    # prompt_ids: (batch_size, prompt_len)
    # output_ids: (batch_size * num_pre_Q, output_len)
    # full_ids: (batch_size * num_pre_Q, prompt_len + output_len)
    # rewards: (batch_size * num_pre_Q,)
    
    # Normalize rewards per prompt group
    # Reshape rewards: (batch_size * num_pre_Q,) -> (batch_size, num_pre_Q)
    rewards_reshaped = rewards.view(batch_size, num_pre_Q)
    
    # Check if any prompt group has sufficient reward variance
    reward_ranges = (rewards_reshaped.max(dim=1)[0] - rewards_reshaped.min(dim=1)[0])
    # reward_ranges: (batch_size,)
    if (reward_ranges < 0.01).all():
        # Debug: rewards too similar (all samples have similar quality)
        # This happens when all generated answers are equally good/bad
        # Skip this batch to avoid training on uninformative data
        return None
    
    # Normalize each prompt group separately
    rewards_mean = rewards_reshaped.mean(dim=1, keepdim=True)
    # rewards_mean: (batch_size, 1)
    rewards_std = rewards_reshaped.std(dim=1, keepdim=True)
    # rewards_std: (batch_size, 1)
    rewards_normalized = (rewards_reshaped - rewards_mean) / (rewards_std + 1e-4)
    # rewards_normalized: (batch_size, num_pre_Q)
    
    # Reshape back to (batch_size * num_pre_Q,)
    rewards = rewards_normalized.view(-1)
    # rewards: (batch_size * num_pre_Q,)
    
    # Get prompt_length from full_ids (already computed in gen_samples)
    prompt_length = prompt_ids.shape[1]  # prompt_length: int
    # full_ids is already on device from gen_samples
    
    # Compute reference log probs
    with torch.inference_mode():
        ref_logits = ref_model(full_ids).logits
        # ref_logits: (batch_size * num_pre_Q, prompt_len + output_len, vocab_size)
        ref_logits = ref_logits[:, :-1, :]
        # ref_logits: (batch_size * num_pre_Q, prompt_len + output_len - 1, vocab_size)
        ref_per_token_logps = get_per_token_logps(ref_logits, full_ids[:, 1:], "ref_model")
        # ref_per_token_logps: (batch_size * num_pre_Q, prompt_len + output_len - 1)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length-1:]
        # ref_per_token_logps: (batch_size * num_pre_Q, output_len)
    
    # Compute generation log probs
    with torch.inference_mode():
        gen_logits = model(full_ids).logits
        # gen_logits: (batch_size * num_pre_Q, prompt_len + output_len, vocab_size)
        gen_logits = gen_logits[:, :-1, :]
        # gen_logits: (batch_size * num_pre_Q, prompt_len + output_len - 1, vocab_size)
        gen_logps = get_per_token_logps(gen_logits, full_ids[:, 1:], "gen_model")
        # gen_logps: (batch_size * num_pre_Q, prompt_len + output_len - 1)
        gen_logps = gen_logps[:, prompt_length-1:]
        # gen_logps: (batch_size * num_pre_Q, output_len)
    
    # Calculate accuracies
    format_accuracy = sum(1 for r in format_rewards if r > 0) / len(format_rewards) if format_rewards else 0.0
    answer_accuracy = sum(1 for r in answer_rewards if r > 0) / len(answer_rewards) if answer_rewards else 0.0
    
    batch = {
        'inputs': full_ids,
        'rewards': rewards.to(device),
        'refs': ref_per_token_logps,
        'plen': prompt_length,
        'gen_logps': gen_logps,
        'format_accuracy': format_accuracy,
        'answer_accuracy': answer_accuracy,
        'prompt_text': prompt_text[0] if prompt_text else "",  # Original prompt text
        'output_text': output_text,  # List of generated output text
    }
    
    return batch

def GRPO_step(batch):
    """Compute GRPO loss for a batch."""
    prompt_length = batch['plen']  # prompt_length: int
    inputs = batch['inputs']
    # inputs: (batch_size * num_pre_Q, prompt_len + output_len)
    advantages = batch['rewards'].unsqueeze(1)  # normalized in generation
    # advantages: (batch_size * num_pre_Q, 1)
    
    # Forward pass
    logits = model(inputs).logits
    # logits: (batch_size * num_pre_Q, prompt_len + output_len, vocab_size)
    logits = logits[:, :-1, :]
    # logits: (batch_size * num_pre_Q, prompt_len + output_len - 1, vocab_size)
    input_ids = inputs[:, 1:]
    # input_ids: (batch_size * num_pre_Q, prompt_len + output_len - 1)
    
    per_token_logps = get_per_token_logps(logits, input_ids, "GRPO_step")
    # per_token_logps: (batch_size * num_pre_Q, prompt_len + output_len - 1)
    per_token_logps = per_token_logps[:, prompt_length-1:]
    # per_token_logps: (batch_size * num_pre_Q, output_len)
    ref_per_token_logps = batch['refs']
    # ref_per_token_logps: (batch_size * num_pre_Q, output_len)
    
    # KL divergence penalty (Schulman's unbiased estimator)
    # Clip difference to prevent exp overflow
    diff = ref_per_token_logps - per_token_logps
    # diff: (batch_size * num_pre_Q, output_len)
    diff_clipped = torch.clamp(diff, min=-20.0, max=20.0)
    # diff_clipped: (batch_size * num_pre_Q, output_len)
    per_token_kl = torch.exp(diff_clipped) - diff_clipped - 1
    # per_token_kl: (batch_size * num_pre_Q, output_len)
    
    # Policy gradient term with PPO-style clipping (using importance sampling)
    # Clip log ratio to prevent exp overflow
    gen_logps = batch['gen_logps']
    # gen_logps: (batch_size * num_pre_Q, output_len)
    log_ratio = per_token_logps - gen_logps
    # log_ratio: (batch_size * num_pre_Q, output_len)
    log_ratio_clipped = torch.clamp(log_ratio, min=-20.0, max=20.0)
    # log_ratio_clipped: (batch_size * num_pre_Q, output_len)
    ratio = torch.exp(log_ratio_clipped)
    # ratio: (batch_size * num_pre_Q, output_len)
    clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
    # clipped_ratio: (batch_size * num_pre_Q, output_len)
    
    # Policy gradient term (clipped)
    policy_grad_term_per_token = torch.min(ratio * advantages, clipped_ratio * advantages)
    # policy_grad_term_per_token: (batch_size * num_pre_Q, output_len)
    
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()
    # completion_mask: (batch_size * num_pre_Q, output_len)
    mask_sum = completion_mask.sum(dim=1)
    # mask_sum: (batch_size * num_pre_Q,)
    mask_sum = torch.clamp(mask_sum, min=1)  # Avoid division by zero
    
    # Compute per-sequence policy gradient and KL terms
    policy_grad_term = (policy_grad_term_per_token * completion_mask).sum(dim=1) / mask_sum
    # policy_grad_term: (batch_size * num_pre_Q,)
    kl_term = (per_token_kl * completion_mask).sum(dim=1) / mask_sum
    # kl_term: (batch_size * num_pre_Q,)
    
    # Compute losses separately
    # GRPO objective: maximize (policy_gradient - beta * KL) = minimize -(policy_gradient - beta * KL)
    policy_loss = -policy_grad_term.mean()  # Negative because we want to maximize policy gradient
    # policy_loss: scalar
    kl_loss = (beta * kl_term).mean()
    # kl_loss: scalar
    
    # Total loss is the sum
    loss = policy_loss + kl_loss
    # loss: scalar
    
    return loss, policy_loss, kl_loss

# Initialize optimizer
learning_rate = 1e-6
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Initialize TensorBoard writer
# Create a new run directory with timestamp for each training session
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"./logs/tensorboard/run_{timestamp}"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

# Initialize sample log file
sample_log_file = f"./logs/samples_{timestamp}.log"
os.makedirs(os.path.dirname(sample_log_file), exist_ok=True)
sample_log = open(sample_log_file, 'w', encoding='utf-8')

# Training loop
print("Starting GRPO training...")
print(f"TensorBoard logs will be saved to: {log_dir}")
print(f"View with: tensorboard --logdir {log_dir}")
progress = tqdm(range(1, all_steps+1))
skipped_batches = 0

import random
random.seed(0)
for step in progress:
    # Generate batch
    batch = generate_batch()
    attempts = 1
    while batch is None:
        skipped_batches += 1
        attempts += 1
        batch = generate_batch()
        # Warn if too many attempts (might indicate a problem)
        if attempts > 10:
            print(f"\nWarning: Skipped {attempts} batches. This might indicate:")
            print("  - Prompts are too long (increase max_prompt_length)")
            print("  - Model generates similar quality answers (check reward function)")
            attempts = 0  # Reset counter
    
    # Training step
    loss, policy_loss, kl_loss = GRPO_step(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Log metrics to TensorBoard
    writer.add_scalar('Loss/Total', loss.item(), step)
    writer.add_scalar('Loss/Policy', policy_loss.item(), step) 
    writer.add_scalar('Loss/KL', kl_loss.item(), step)
    writer.add_scalar('Accuracy/Format', batch['format_accuracy'], step)
    writer.add_scalar('Accuracy/Answer', batch['answer_accuracy'], step)
    writer.add_scalar('Metrics/Skipped_Batches', skipped_batches, step)
    
    progress.set_description(f"Loss: {loss.item():.6f} | Format: {batch['format_accuracy']:.2%} | Answer: {batch['answer_accuracy']:.2%} | Skipped: {skipped_batches}")
    
    # Log a sampled prompt + output to file
    # Use the first output as a sample
    sample_output = batch['output_text'][0] if batch['output_text'] else ""
    prompt_text = batch['prompt_text']
    
    # Write to log file
    sample_log.write(f"\n{'='*80}\n")
    sample_log.write(f"Step {step}\n")
    sample_log.write(f"Loss: {loss.item():.6f} | Format Accuracy: {batch['format_accuracy']:.2%} | Answer Accuracy: {batch['answer_accuracy']:.2%}\n")
    sample_log.write(f"{'-'*80}\n")
    sample_log.write(f"Prompt:\n{prompt_text}\n")
    sample_log.write(f"{'-'*80}\n")
    sample_log.write(f"{sample_output}\n")
    sample_log.write(f"{'-'*80}\n")
    sample_log.flush()  # Ensure it's written immediately
    
    # Save checkpoint
    if step % save_steps == 0:
        print(f'Saving model at step {step}')
        save_name = f"./step_{step}"
        model.save_pretrained(save_name)
        tokenizer.save_pretrained(save_name)

# Close TensorBoard writer and sample log
writer.close()
sample_log.close()
print("Training complete!")
print(f"TensorBoard logs saved to: {log_dir}")
print(f"View with: tensorboard --logdir {log_dir}")
print(f"Sample logs saved to: {sample_log_file}")


