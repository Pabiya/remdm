#!/bin/bash
#SBATCH -J remdm-loop                 # Job name
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

checkpoint_path=YOUR-BASE-PATH/remdm/outputs/checkpoints/mdlm.ckpt
T=0
sampling_steps=1024
p=0.9
eta=0.02
t_on=0.55
t_off=0.05
alpha_on=0.9
generated_seqs_path=YOUR-BASE-PATH/remdm/outputs/remdm-loop_T-${sampling_steps}_eta-${eta}_ton-${t_on}_toff-${t_off}_alphaon-${alpha_on}_topp-${p}.json

export HYDRA_FULL_ERROR=1

srun python -u -m main \
    mode=sample_eval \
    loader.batch_size=1 \
    loader.eval_batch_size=1 \
    eval.perplexity_batch_size=1 \
    data=openwebtext-split \
    model=small \
    parameterization=subs \
    backbone=dit \
    model.length=1024 \
    eval.checkpoint_path=${checkpoint_path} \
    time_conditioning=false \
    +wandb.offline=true \
    hydra.run.dir="${PWD}/outputs/remdm-loop" \
    T=${T} \
    sampling.steps=${sampling_steps} \
    seed=1 \
    sampling.num_sample_batches=5000 \
    sampling.generated_seqs_path=${generated_seqs_path} \
    sampling.nucleus_p=${p} \
    sampling.sampler="remdm-loop" \
    sampling.eta=${eta} \
    sampling.t_on=${t_on} \
    sampling.t_off=${t_off} \
    sampling.alpha_on=${alpha_on}
