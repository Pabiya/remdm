#!/bin/bash
#SBATCH -J refine-ent-3               # (선택) Job name 변경
#SBATCH -o watch_folder/%x_%j.out
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=32000
#SBATCH -t 960:00:00
#SBATCH --partition=gpu
#SBATCH --constraint="[3090|a5000|a6000|a100]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH --requeue

checkpoint_path=/home/junsu0115/remdm/outputs/checkpoints/mdlm.ckpt
T=0
sampling_steps=1024
p=0.9

k=1

generated_seqs_path=/home/junsu0115/remdm/outputs/refine-ent-3_T-${sampling_steps}_topp-${p}_k-${k}.json

export HYDRA_FULL_ERROR=1

python -u -m main \
    mode=sample_eval \
    loader.batch_size=1 \
    loader.eval_batch_size=1 \
    eval.perplexity_batch_size=1 \
    data=openwebtext-streaming \
    model=small \
    parameterization=subs \
    backbone=dit \
    model.length=1024 \
    eval.checkpoint_path=${checkpoint_path} \
    time_conditioning=false \
    +wandb.offline=true \
    hydra.run.dir="${PWD}/outputs/refine-ent-3" \
    T=${T} \
    sampling.steps=${sampling_steps} \
    seed=1 \
    sampling.num_sample_batches=10 \
    sampling.generated_seqs_path=${generated_seqs_path} \
    sampling.nucleus_p=${p} \
    sampling.sampler="refine-ent-3" \
    +sampling.refine_every=${k} \
    +sampling.entropy_remove_mask_prob=true \
    +sampling.print_prob_entropy_stats=false \
    +sampling.print_every=100 \
    +sampling.log_unmask_remask_order=false \
    +sampling.log_unmask_remask_example_idx=0
