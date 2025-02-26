# Remasking Discrete Diffusion Models with Inference-Time Scaling

[![arXiv](https://img.shields.io/badge/arXiv-2406.07524-red.svg)](https://arxiv.org/abs/2406.07524)
[![deploy](https://img.shields.io/badge/Blog%20%20-8A2BE2)](https://remdm.github.io)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18nC6q7dWq154fI1BXPLwmtnS7Zvbrv6p?usp=sharing/)

![graphical_abstract](./assets/graphical_abstract.png)

We introduce *ReMDM*, a simple and general framework to design remasking samplers for masked discrete diffusion models. In this repo, we provide our implementation of different ReMDM strategies for unconditional text generation on OpenWebText. We also provide a demo in this [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18nC6q7dWq154fI1BXPLwmtnS7Zvbrv6p?usp=sharing/) notebook showing how to download the MDLM checkpoint and implement ReMDM-loop on top of it.


Our main add-ons from MDLM are:
* **Evaluation Metrics**
  1. Add MAUVE computation code.
  2. Add entropy computation code.
* **Sampling Tricks** 
  1. Replace fp32 gumbel noise with the correct fp64 gumbel noise.
  2. Implement nucleus sampling.
* **ReMDM Strategies**
  1. Implement different ReMDM strategies, including ReMDM-cap, ReMDM-rescale, ReMDM-conf, and ReMDM-loop.
* **Predictor-Corrector Samplers**
  1. Implement forward-backward and discrete flow matching corrector samplers as extra baselines.


<a name="getting_started"></a>

## Getting started

To get started, create a conda environment containing the required dependencies.

```bash
conda env create -f requirements.yaml
conda activate remdm
```

Create the following directories to store saved models and slurm logs:
```bash
mkdir outputs
mkdir watch_folder
```

Download checkpoints from this [Google Drive folder](https://drive.google.com/drive/folders/16LuuptK7Xfk-vzhQYZBZ0SA-B-BFluau?usp=sharing) released by the MDLM repo and put them under
the following directory `./outputs/checkpoints`

## Reproducing Experiments

Below, we describe the steps required for reproducing the experiments in the paper.
Throughout, the main entry point for running experiments is the [`main.py`](./main.py) script.
We also provide sample `slurm` scripts for launching pre-training and downstream fine-tuning experiments in the [`scrips/`](./scripts) directory.


### Generate Samples
<a name="sample-gen"></a>
The argument to `sampling.predictor` specifies the sampler which takes one of the following values:
* `ddpm_cache`: our proposed sampler that's **~3-4x** faster than the samplers propsed in D3PM and SEDD.
* `ddpm`: Ancestral sampling proposed in D3PM.
* `analytic`: Analytic sampler proposed in SEDD.

In the following table we report wall clock time to generate 64 samples on a single A5000 GPU with `batch_size=1`. $T$ denotes the time discretization of the reverse process.
|                         | $T=5k (\downarrow)$ | $T=10k (\downarrow)$ |
|-------------------------|---------------------|----------------------|
| **SEDD**                | 127.1               | 229.3                |
| **MDLM** + `ddpm`       | 113.8               | 206.6                |
| **MDLM** +`ddpm_cache`  | **40.1**            | **60.4**             |


To generate samples from a pre-trained model use one of the following commands:
#### Huggingface model
```bash
python main.py \
  mode=sample_eval \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  data=openwebtext-split  \
  model.length=1024  \
  sampling.predictor=ddpm_cache  \
  sampling.steps=1000 \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=10 \
  backbone=hf_dit
```
#### Local checkpoint
```bash
python main.py \
  mode=sample_eval \
  eval.checkpoint_path=/path/to/checkpoint/mdlm.ckpt \
  data=openwebtext-split  \
  model.length=1024  \
  sampling.predictor=ddpm_cache  \
  sampling.steps=10000 \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=1 \
  backbone=dit
```

### Semi-AR sample generation
<a name="semi-ar-gen"></a>
MDLM can also generate samples of arbitrary length in a semi-autoregressive (SAR) manner.
We generate 200 sequences of length 2048 tokens on a single `3090` GPU and evaluate generative perplexity under a pre-trained GPT-2 model. In the below table we find that in addition to achieving better generative perplexity, MDLM enables **25-30x** faster SAR decoding relative to [SSD-LM](https://arxiv.org/abs/2210.17432).

|                     | Gen. PPL ($\downarrow$) | Sec/Seq ($\downarrow$) |
|---------------------|-------------------------|------------------------|
| **SSD-LM**          | 35.43                   | 2473.9                 |
| **MDLM** +`ddpm_cache`  | **27.18**               | **89.3**               |

*Gen. PPL: Generation Perplexity, Sec/Seq: Seconds per Sequence*

```bash
python main.py \
  mode=sample_eval \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  data=openwebtext-split \
  parameterization=subs \
  model.length=1024  \
  sampling.predictor=ddpm_cache  \
  sampling.steps=1000 \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=2 \
  sampling.semi_ar=True \
  sampling.stride_length=512 \
  sampling.num_strides=2 \
  backbone=hf_dit
```

### Train
To train MDLM from scratch on OpenWebText use the following command:
```
python main.py \
  model=small \
  data=openwebtext-split \
  wandb.name=mdlm-owt \
  parameterization=subs \
  model.length=1024 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000
```
The arguments `loader.batch_size` and `loader.eval_batch_size` allow you to control the global batch size and the batch size per GPU. If `loader.batch_size * num_gpus` is less than the global batch size, PyTorch Lightning will resort to gradient accumulation. You can also launch a training job on Slurm using the command: `sbatch scripts/train_owt_mdlm.sh`. The slurm scripts to train the Auto-regressive and SEDD baselines are as follows respectively: [`scripts/train_lm1b_ar.sh`](scripts/train_lm1b_ar.sh), [`scripts/train_owt_sedd.sh`](scripts/train_owt_sedd.sh).

### Eval 
To compute test perplexity, use `mode=ppl_eval`. Example scripts provided in `scripts/`. An example command for perplexity evaluation on OpenWebText is:
```
python main.py \
  mode=ppl_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  data=openwebtext-split \
  model=small \
  parameterization=subs \
  backbone=dit \
  model.length=1024 \
  eval.checkpoint_path=/path/to/checkpoint/mdlm.ckpt \
  +wandb.offline=true
```

### Baseline evaluation
<a name="baselines"></a>
We release the checkpoints for the baselines: SEDD and AR trained on OpenWebText in this [Google Drive folder](https://drive.google.com/drive/folders/16LuuptK7Xfk-vzhQYZBZ0SA-B-BFluau?usp=sharing). Download the checkpoints: `ar.ckpt`, `sedd.ckpt` and use the following commands to compute test perplexity:
#### AR
```bash
python main.py \
  mode=ppl_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  data=openwebtext-split \
  model=small-ar \
  parameterization=ar \
  backbone=ar \
  model.length=1024 \
  eval.checkpoint_path=/path/to/checkpoint/ar.ckpt \
  +wandb.offline=true
```
#### SEDD
```bash
python main.py \
  mode=ppl_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  data=openwebtext-split \
  model=small \
  parameterization=sedd \
  backbone=dit \
  model.length=1024 \
  eval.checkpoint_path=/path/to/checkpoint/sedd.ckpt \
  time_conditioning=True \
  sampling.predictor=analytic \
  +wandb.offline=true
```

### Acknowledgements
This repository was built off of [MDLM] (https://github.com/kuleshov-group/mdlm) which was based on [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion).

## Citation
```

```
