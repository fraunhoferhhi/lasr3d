# LaSR-3D

This repository is the implementation of the Paper "3D-Aware Latent-Space Reenactment: Combining Expression Transfer and Semantic Editing".

![Teaser image](./example/teaser.png)

<div align="center">CVMP Best Paper Award 2025! 🎉<b></b></div>

## Install

Clone the repository.

```
git clone git@github.com:fraunhoferhhi/lasr3d.git
cd lasr_3d
```

Before obtaining the models, make sure that you have an Account to access the [FLAME Model](https://flame.is.tue.mpg.de/login.php). We provided an `install.sh` script to build and configure the environment and obtain the models. It was tested with CUDA 11.7 and CUDA 11.8

```
source ./install.sh -a
conda activate lasr_3d
```

Alternatively, you can run `./install.sh -m` to exclusively download the models, `./install.sh -e` to exclusively set up the environment, or execute the individual steps in the installation script manually.

## Run

First, a model needs to be tuned on two individuals. Fine tuning the model takes about one hour of time.

```
python tune.py --id1 ./example/001.mp4 --id2 ./example/002.mp4 --out ./outputs/example/
```

Then, the tuned model can be used to edit the attributes of the people

```
python edit.py --model ./outputs/example/model.pkl --source 1 --target 0 --out ./outputs/example/ -e "glasses" +1.3 -e "sentiment" +0.3
```


