# Getting Started

## Installation
* Clone this repo:
```
git clone https://github.com/loganbrenningmeyer/diffusion.git
cd diffusion
```

* Install dependencies or create/activate a new Conda environment with:
```
conda env create -f environment.yml
conda activate diffusion
```

## Training
* Login to wandb / set environment variables
```
wandb login

export WANDB_ENTITY="<entity>"
export WANDB_PROJECT="<project>"
```

* Configure training run and launch train script
```
sh scripts/train.sh --config <config>.yml
```