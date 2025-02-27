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

Below, we demonstrate how to generate text samples using different models and samplers. It should be as easy as replacing the "YOUR-BASE-PATH" with your path to the repository and running the following SLURM bash scripts.

* AR
```bash
sbatch scripts/ar.sh
```

* SEDD
```bash
sbatch scripts/sedd.sh
```

* MDLM
```bash
sbatch scripts/mdlm.sh
```

* Forward-Backward corrector
```bash
sbatch scripts/fb.sh
```

* Discrete flow matching corrector
```bash
sbatch scripts/dfm.sh
```

* ReMDM
```bash
sbatch scripts/remdm-{YOUR-CHOSEN-STRATEGY}.sh
```

### Acknowledgements
This repository was built off of [MDLM](https://github.com/kuleshov-group/mdlm) which was based on [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion).

## Citation
```

```
