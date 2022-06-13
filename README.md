# CAT
## 1. hardware configuration:
For training, we used a server with two Intel(R) Xeon(R) Gold 5215 2.5 GHz CPUs, 256 GB RAM, and three NVIDIA Corporation GV100GL GPUs.

## 2. software configuration:
linux   Ubuntu

CUDA  10.1  Download:https://developer.nvidia.com/downloads


anaconda   Anaconda3-2020.07-Linux-x86_64   Download:https://www.anaconda.com/products/distribution#macos

python  3.7

Pytorch  1.4.0 Installation commands: conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch

scikit-learn  0.24 Installation commands: pip install scikit-learn

scipy  1.7.1 Installation commands: pip install scipy  

mol2vec   0.1 Installation commands: pip install git+https://github.com/samoturk/mol2vec

rdkit  2021.09.3 Installation commands: conda install rdkit

numpy  1.20.3 Installation commands: pip install numpy

## 3.  Instructions for running the CAT
To run CAT on DDI network, execute the following command from the project home directory:

`python train.py`

**Input:**

The supported input format are three Matrixes:

DDI adjacency matrix with n*n(int)

drug SMILEs matrix with n*1024(int)

drug similarity matrix with n*n(float)

**Output:**

The existence probability matrix of DDIs with n*n (float)

**Result:**

The  DDIs prediction performance AUROC&AUPR&AP&F1 score :

| Dataset | AUROC | AUPR | AP | F1 |

| :-------: | :-------: | :------: | :---: | :---: |

| DDI | 0.932 | 0.934 | 0.934 | 0.802|
