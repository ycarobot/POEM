# Explore Unexplored Reliable Samples to Enhance Test-Time Adaptation


This is the official implementation of [Explore Unexplored Reliable Samples to Enhance Test-Time Adaptation]

This implementation is based on [DeYO implementation 🔗](https://github.com/mr-eggplant/SAR).

## Environments  
You should modify [username] and [env_name] in environment.yaml, then  
> $ conda env create --file environment.yaml  

## Baselines  
[TENT 🔗](https://arxiv.org/abs/2006.10726) (ICLR 2021)  
[EATA 🔗](https://arxiv.org/abs/2204.02610) (ICML 2022)  
[SAR 🔗](https://arxiv.org/abs/2302.12400) (ICLR 2023)  
[DeYO 🔗](https://openreview.net/forum?id=9w3iw8wDuE) (ICLR 2024)  
## Dataset
You can download ImageNet-C from a link [ImageNet-C 🔗](https://zenodo.org/record/2235448).  
After downloading the dataset, move to the root directory ([data_root]) of datasets.  
Your [data_root] will be as follows:
```bash
data_root
├── ImageNet-C
│   ├── brightness
│   ├── contrast
│   └── ...
```
## Experiment

You can run most of the experiments in our paper by  
shell: python main.py

Moreover, we also prepare code for the plug-and-play, check the tentPOEM-I tentPOEM-II, and tentPOEM in methods filepackage.

## Acknowledgment
The code is inspired by the [Tent 🔗](https://github.com/DequanWang/tent), [EATA 🔗](https://github.com/mr-eggplant/EATA), [SAR 🔗](https://github.com/mr-eggplant/SAR) and [DeYO 🔗](https://openreview.net/forum?id=9w3iw8wDuE) (ICLR 2024) .
