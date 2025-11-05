
p=0.9
T=0

N=100

sampling_steps=256
L=1024
# mirage
k=1
# remdm-loop
eta=0.02
t_on=0.55
t_off=0.05
alpha_on=0.9

BASE_PATH=/home/junsu0115
checkpoint_path=${BASE_PATH}/remdm/outputs/checkpoints/mdlm.ckpt
generated_seqs_path_mirage=${BASE_PATH}/remdm/outputs/experiment_2/refine-ent-3_T-${sampling_steps}_L-${L}_k-${k}.json
generated_seqs_path_conf=${BASE_PATH}/remdm/outputs/experiment_2/remdm-conf_T-${sampling_steps}_L-${L}.json
generated_seqs_path_loop=${BASE_PATH}/remdm/outputs/experiment_2/remdm-loop_T-${sampling_steps}_L-${L}_eta-${eta}_ton-${t_on}_toff-${t_off}_alphaon-${alpha_on}.json
generated_seqs_path_mdlm=${BASE_PATH}/remdm/outputs/experiment_2/mdlm_T-${sampling_steps}_L-${L}.json
generated_seqs_path_sedd=${BASE_PATH}/remdm/outputs/experiment_2/sedd_T-${sampling_steps}_L-${L}.json


export HYDRA_FULL_ERROR=1

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

# remdm-conf
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
    hydra.run.dir="${PWD}/outputs/remdm-conf" \
    T=${T} \
    sampling.steps=${sampling_steps} \
    seed=1 \
    sampling.num_sample_batches=${N} \
    sampling.generated_seqs_path=${generated_seqs_path_conf} \
    sampling.nucleus_p=${p} \
    sampling.sampler="remdm-conf"

# remdm-loop
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
    hydra.run.dir="${PWD}/outputs/remdm-loop" \
    T=${T} \
    sampling.steps=${sampling_steps} \
    seed=1 \
    sampling.num_sample_batches=${N} \
    sampling.generated_seqs_path=${generated_seqs_path_loop} \
    sampling.nucleus_p=${p} \
    sampling.sampler="remdm-loop" \
    sampling.eta=${eta} \
    sampling.t_on=${t_on} \
    sampling.t_off=${t_off} \
    sampling.alpha_on=${alpha_on}

# baseline mdlm
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
    hydra.run.dir="${PWD}/outputs/mdlm" \
    T=${T} \
    sampling.steps=${sampling_steps} \
    seed=1 \
    sampling.num_sample_batches=${N} \
    sampling.generated_seqs_path=${generated_seqs_path_mdlm} \
    sampling.nucleus_p=${p} \
    sampling.sampler="mdlm"

# sedd
python -u -m main \
    mode=sample_eval \
    loader.batch_size=1 \
    loader.eval_batch_size=1 \
    eval.perplexity_batch_size=1 \
    data=openwebtext-streaming \
    model=small \
    parameterization=sedd \
    backbone=dit \
    model.length=${L} \
    eval.checkpoint_path=${checkpoint_path} \
    time_conditioning=true \
    +wandb.offline=true \
    hydra.run.dir="${PWD}/outputs/sedd" \
    T=${T} \
    sampling.steps=${sampling_steps} \
    seed=1 \
    sampling.num_sample_batches=${N} \
    sampling.generated_seqs_path=${generated_seqs_path_sedd} \
    sampling.predictor="analytic"