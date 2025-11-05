L=256
sampling_steps=256

N=100

BASE_PATH=/home/junsu0115
checkpoint_path=${BASE_PATH}/remdm/outputs/checkpoints/sedd.ckpt
generated_seqs_path_sedd=${BASE_PATH}/remdm/outputs/experiment_1/sedd_T-${sampling_steps}_L-${L}.json

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
    T=0 \
    sampling.steps=${sampling_steps} \
    seed=1 \
    sampling.num_sample_batches=${N} \
    sampling.generated_seqs_path=${generated_seqs_path_sedd} \
    sampling.predictor="analytic"