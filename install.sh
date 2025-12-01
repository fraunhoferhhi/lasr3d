#!/bin/bash

while getopts "ame" opt; do
  case $opt in
    a ) models=1
        env=1;;
    m ) models=1;;
    e ) env=1;;
    \? ) echo 'usage: install [-a all] [-m models] [-e environment]'
      exit 1
  esac
done
shift $(($OPTIND -1))

function download_models
{
    mkdir -p models
    cd models
    echo -e "\nYou need to be registered to download the FLAME model. If you do not have an account you can register at https://flame.is.tue.mpg.de/ following the installation instruction."
    read -p "Username (FLAME):" username
    read -p "Password (FLAME):" password
    echo -e "\nDownloading FLAME..."
    mkdir -p data/FLAME2020/
    wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1' -O './FLAME2020.zip' --no-check-certificate --continue
    unzip FLAME2020.zip -d data/FLAME2020/
    mv ./data/FLAME2020/generic_model.pkl "./generic_flame_model.pkl"
    rm -rf FLAME2020.zip
    rm -r data

    gdown 1cDvUHPTZQAEWvfiK9C0nDuI9C3Qdgbbp
    unzip pretrained_models.zip -d . 
    mv ./pretrained_models/79999_iter.pth . 
    mv ./pretrained_models/shape_predictor_68_face_landmarks.dat . 
    rm -r pretrained_models
    rm pretrained_models.zip
    
    gdown 1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz
    gdown 1iK5lD49E_gCn9voUjWDPj2ItGKvM10GI
    gdown 1vmWX6QmXGPnXTXWFgj67oHzOoOmxBh6B
    curl -L -o "./param_mean_std_62d_120x120.pkl" "https://github.com/cleardusk/3DDFA_V2/raw/refs/heads/master/configs/param_mean_std_62d_120x120.pkl"
    curl -L -o "./mb1_120x120.pth" "https://github.com/cleardusk/3DDFA_V2/raw/refs/heads/master/weights/mb1_120x120.pth"
    curl -L -o "cgs-gan-512.pkl" "https://huggingface.co/Fubei/cgs_gan_checkpoints/resolve/main/ffhqc_512.pkl"
    curl -L -o "./model_ir_se50.pth" "https://huggingface.co/Fubei/splatviz_inversion_checkpoints/resolve/main/model_ir_se50.pth"
    cd ..

}

function write_config
{
    local home=$HOME
    local project=$PWD
    config_path=configs/config.py
    echo "HOME = '$HOME'" > $config_path
    echo "PROJECT = '$PWD'" >> $config_path
    echo "DEBUG = False" >> $config_path
    echo "LOG = False" >> $config_path

}

function build_env
{
    local conda_bin=$(which conda)
    local conda_base=$(dirname $(dirname $conda_bin))
    local conda_sh="${conda_base}/etc/profile.d/conda.sh"
    if [ ! -f "$conda_sh" ]; then
        echo "Could not find conda installation"
        exit 1
    fi
    source "$conda_sh"
    if [ -n "$(conda env list | grep 'lasr_3d')" ]; then
        echo "A conda env with the name 'lasr_3d' already exists"
        exit 1
    fi

    conda create -y -n lasr_3d python=3.8.13
    conda activate lasr_3d
    conda install -y pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
    pip install -r requirements.txt
    git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
    export PATH=$PATH:/usr/local/bin/aws
    pip install ./diff-gaussian-rasterization
    rm -rf ./diff-gaussian-rasterization
    cd ./external/
    git clone https://github.com/fraunhoferhhi/cgs-gan.git --single-branch
    mv cgs-gan cgsgan
    git clone https://github.com/filby89/spectre.git
    git clone https://github.com/afruehstueck/VIVE3D.git
    cd ..
}

function compile_preprocessor
{
    cd ./preprocess_cgsgan/3DDFA_V2/
    ./build.sh
}

if [ ! -z $models ];then
    download_models
fi
if [ ! -z $env ];then
    build_env
    write_config
    compile_preprocessor
fi
