# High-Resolution Rainy Image Synthesis: Learning From Rendering - HRIGNet


## Requirements
A suitable conda environment named `hrig` can be created and activated with:

```
conda env create -f environment.yaml
conda activate hrig
```

---

## How To Run

* Get the first stage models：`bash scrpit/download_first_stages.sh`
* Train models：`bash train.sh`
* Image generation：`bash predict.sh`

## Comment

* Our codebase for the diffusion models builds heavily on [Latent Diffusion](https://github.com/CompVis/latent-diffusion) and [DiT](https://github.com/facebookresearch/DiT). Thanks for open-sourcing!
