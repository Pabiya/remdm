#!/bin/bash
#SBATCH -J ar                         # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu               # Request partition
#SBATCH --constraint="[3090|a5000|a6000|a100]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

checkpoint_path=/home/junsu0115/remdm/outputs/checkpoints/ar.ckpt
T=0
sampling_steps=1024
p=0.9
generated_seqs_path=/home/junsu0115/remdm/outputs/ar.json

export HYDRA_FULL_ERROR=1
mkdir -p "$(dirname "$generated_seqs_path")" "$PWD/outputs" "$PWD/.hf_cache" ### added

# python -u -m main \
#     mode=sample_eval \
#     loader.batch_size=1 \
#     loader.eval_batch_size=1 \
#     eval.perplexity_batch_size=1 \
#     data=openwebtext-split \
#     model=small-ar \
#     parameterization=ar \
#     backbone=ar \
#     model.length=1024 \
#     eval.checkpoint_path=${checkpoint_path} \
#     time_conditioning=false \
#     +wandb.offline=true \
#     hydra.run.dir="${PWD}/outputs/ar" \
#     T=${T} \
#     sampling.steps=${sampling_steps} \
#     seed=1 \
#     sampling.num_sample_batches=5000 \
#     sampling.generated_seqs_path=${generated_seqs_path} \
#     sampling.nucleus_p=${p}

# python -u -m main \
#   mode=sample_eval \
#   data=openwebtext-split \
#   model=small-ar backbone=ar parameterization=ar model.length=1024 \
#   eval.checkpoint_path="${checkpoint_path}" \
#   sampling.generated_seqs_path="${generated_seqs_path}" \
#   loader.batch_size=1 loader.eval_batch_size=1 eval.perplexity_batch_size=1 \
#   T=0 sampling.steps=1024 seed=1 sampling.num_sample_batches=1 sampling.nucleus_p=0.9 \
#   hydra.run.dir="$PWD/outputs/ar" \
#   data.cache_dir="$PWD/.hf_cache"

python -u -m main \
  mode=sample_eval \
  data=openwebtext-streaming \
  model=small-ar backbone=ar parameterization=ar model.length=1024 \
  eval.checkpoint_path="/home/junsu0115/remdm/outputs/checkpoints/ar.ckpt" \
  sampling.generated_seqs_path="$PWD/outputs/ar_samples.json" \
  loader.batch_size=1 loader.eval_batch_size=1 eval.perplexity_batch_size=1 \
  T=0 sampling.steps=1024 sampling.num_sample_batches=1 sampling.nucleus_p=0.9 \
  hydra.run.dir="$PWD/outputs/ar_stream" \
  data.cache_dir="$PWD/.hf_cache" \
  eval.compute_generative_perplexity=false \
  +eval.compute_mauve=false