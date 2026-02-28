#!/usr/bin/env python3
"""Plot metrics from TensorBoard logs."""
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import sys
import os

# Get log directory from command line argument or use default
if len(sys.argv) > 1:
    log_dir = sys.argv[1]
else:
    # Default to most recent run
    tensorboard_dir = 'logs/tensorboard'
    if os.path.exists(tensorboard_dir):
        runs = [d for d in os.listdir(tensorboard_dir) if os.path.isdir(os.path.join(tensorboard_dir, d)) and d.startswith('run_')]
        if runs:
            runs.sort(reverse=True)
            log_dir = os.path.join(tensorboard_dir, runs[0])
        else:
            print("No runs found in logs/tensorboard/")
            sys.exit(1)
    else:
        print(f"TensorBoard directory not found: {tensorboard_dir}")
        sys.exit(1)

# Extract run name for title and output paths
run_name = os.path.basename(log_dir)

print(f"Loading data from: {log_dir}")
print(f"Run name: {run_name}")

# Load data
ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()

# Extract data
loss_data = ea.Scalars('Loss/Train')
format_acc_data = ea.Scalars('Accuracy/Format')
answer_acc_data = ea.Scalars('Accuracy/Answer')

loss_steps = [s.step for s in loss_data]
loss_values = [s.value for s in loss_data]

format_steps = [s.step for s in format_acc_data]
format_values = [s.value for s in format_acc_data]

answer_steps = [s.step for s in answer_acc_data]
answer_values = [s.value for s in answer_acc_data]

# Create figure with subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle(f'Training Metrics - {run_name}', fontsize=14, fontweight='bold')

# Plot 1: Loss
ax1 = axes[0]
ax1.plot(loss_steps, loss_values, 'b-', linewidth=1.5, alpha=0.7, label='Loss')
ax1.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Zero line')
ax1.set_xlabel('Step')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_ylim([min(loss_values) * 1.1, max(loss_values) * 1.1])

# Add text annotation for negative values
negative_count = sum(1 for v in loss_values if v < 0)
if negative_count > 0:
    ax1.text(0.02, 0.98, f'Negative values: {negative_count}/{len(loss_values)} ({negative_count/len(loss_values)*100:.1f}%)',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: Format Accuracy
ax2 = axes[1]
ax2.plot(format_steps, format_values, 'g-', linewidth=1.5, alpha=0.7, label='Format Accuracy')
ax2.axhline(y=0.5, color='r', linestyle='--', linewidth=1, alpha=0.5, label='50% baseline')
ax2.set_xlabel('Step')
ax2.set_ylabel('Accuracy')
ax2.set_title('Format Accuracy')
ax2.set_ylim([0, 1.05])
ax2.grid(True, alpha=0.3)
ax2.legend()

# Add mean line
mean_format = np.mean(format_values)
ax2.axhline(y=mean_format, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Mean: {mean_format:.3f}')
ax2.legend()

# Add text annotation
ax2.text(0.02, 0.98, f'Mean: {mean_format:.3f}\nRange: {min(format_values):.3f} - {max(format_values):.3f}',
         transform=ax2.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Plot 3: Answer Accuracy
ax3 = axes[2]
ax3.plot(answer_steps, answer_values, 'm-', linewidth=1.5, alpha=0.7, label='Answer Accuracy')
ax3.axhline(y=0.5, color='r', linestyle='--', linewidth=1, alpha=0.5, label='50% baseline')
ax3.set_xlabel('Step')
ax3.set_ylabel('Accuracy')
ax3.set_title('Answer Accuracy')
ax3.set_ylim([0, 1.05])
ax3.grid(True, alpha=0.3)
ax3.legend()

# Add mean line
mean_answer = np.mean(answer_values)
ax3.axhline(y=mean_answer, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Mean: {mean_answer:.3f}')
ax3.legend()

# Add trend line
if len(answer_values) > 10:
    z = np.polyfit(answer_steps, answer_values, 1)
    p = np.poly1d(z)
    ax3.plot(answer_steps, p(answer_steps), "r--", alpha=0.5, linewidth=1, label=f'Trend: {z[0]:.6f}/step')
    ax3.legend()

# Add text annotation
first_10_mean = np.mean(answer_values[:10])
last_10_mean = np.mean(answer_values[-10:])
ax3.text(0.02, 0.98, f'Mean: {mean_answer:.3f}\nFirst 10: {first_10_mean:.3f}\nLast 10: {last_10_mean:.3f}\nChange: {(last_10_mean-first_10_mean)/first_10_mean*100:+.1f}%',
         transform=ax3.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

plt.tight_layout()
output_file = os.path.join(log_dir, 'training_metrics.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {output_file}")

# Also create a combined accuracy plot
fig2, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(format_steps, format_values, 'g-', linewidth=2, alpha=0.7, label='Format Accuracy', marker='o', markersize=3)
ax.plot(answer_steps, answer_values, 'm-', linewidth=2, alpha=0.7, label='Answer Accuracy', marker='s', markersize=3)
ax.axhline(y=0.5, color='r', linestyle='--', linewidth=1, alpha=0.5, label='50% baseline')
ax.set_xlabel('Step', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Accuracy Comparison - Format vs Answer', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

# Add mean lines
ax.axhline(y=mean_format, color='g', linestyle=':', linewidth=1.5, alpha=0.5)
ax.axhline(y=mean_answer, color='m', linestyle=':', linewidth=1.5, alpha=0.5)

plt.tight_layout()
output_file2 = os.path.join(log_dir, 'accuracy_comparison.png')
plt.savefig(output_file2, dpi=150, bbox_inches='tight')
print(f"Combined accuracy plot saved to: {output_file2}")

plt.show()

