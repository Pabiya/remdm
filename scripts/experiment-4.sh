
p=0.9
T=0

N=100

sampling_steps=1024
L=1024
# mirage
k=1

BASE_PATH=/home/junsu0115
checkpoint_path=${BASE_PATH}/remdm/outputs/checkpoints/mdlm.ckpt
generated_seqs_path_mirage=${BASE_PATH}/remdm/outputs/experiment_2/refine-ent-3_T-${sampling_steps}_L-${L}_k-${k}.json
generated_seqs_path_remdm_ent_2=${BASE_PATH}/remdm/outputs/remdm-ent-2_T-${sampling_steps}_L-${L}.json
generated_seqs_path_refine_conf=${BASE_PATH}/remdm/outputs/refine-conf_T-${sampling_steps}_L-${L}_k-${k}.json

# mirage
python -u -m main \
    mode=sample_eval \
    loader.batch_size=1 \
    loader.eval_batch_size=1 \
    eval.perplexity_batch_size=1 \
    data=openwebtext-streaming \
    model=small \
    parameterization=subs \
    backbone=dit \
    model.length=${L} \
    eval.checkpoint_path=${checkpoint_path} \
    time_conditioning=false \
    +wandb.offline=true \
    hydra.run.dir="${PWD}/outputs/refine-ent-3" \
    T=${T} \
    sampling.steps=${sampling_steps} \
    seed=1 \
    sampling.num_sample_batches=${N} \
    sampling.generated_seqs_path=${generated_seqs_path_mirage} \
    sampling.nucleus_p=${p} \
    sampling.sampler="refine-ent-3" \
    +sampling.refine_every=${k} \
    +sampling.entropy_remove_mask_prob=true \
    +sampling.print_prob_entropy_stats=false \
    +sampling.print_every=100 \
    +sampling.log_unmask_remask_order=false \
    +sampling.log_unmask_remask_example_idx=0

# remdm-ent-2
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
    hydra.run.dir="${PWD}/outputs/remdm-ent-2" \
    T=${T} \
    sampling.steps=${sampling_steps} \
    seed=1 \
    sampling.num_sample_batches=1 \
    sampling.generated_seqs_path=${generated_seqs_path_remdm_ent_2} \
    sampling.nucleus_p=${p} \
    sampling.sampler="remdm-ent-2" \
    +sampling.entropy_remove_mask_prob=true \
    +sampling.print_prob_entropy_stats=true \
    +sampling.print_every=100 \
    +sampling.log_unmask_remask_order=true \
    +sampling.log_unmask_remask_example_idx=0

# refine-conf
python -u -m main \
    mode=sample_eval \
    loader.batch_size=1 \
    loader.eval_batch_size=1 \
    eval.perplexity_batch_size=1 \
    data=openwebtext-streaming \
    model=small \
    parameterization=subs \
    backbone=dit \
    model.length=${L} \
    eval.checkpoint_path=${checkpoint_path} \
    time_conditioning=false \
    +wandb.offline=true \
    hydra.run.dir="${PWD}/outputs/refine-conf" \
    T=${T} \
    sampling.steps=${sampling_steps} \
    seed=1 \
    sampling.num_sample_batches=${N} \
    sampling.generated_seqs_path=${generated_seqs_path_refine_conf} \
    sampling.nucleus_p=${p} \
    sampling.sampler="refine-conf" \
    +sampling.refine_every=${k} \
    +sampling.entropy_remove_mask_prob=true \
    +sampling.print_prob_entropy_stats=true \
    +sampling.print_every=100 \
    +sampling.log_unmask_remask_order=true \
    +sampling.log_unmask_remask_example_idx=0