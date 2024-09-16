# Improving Robustness of Hyperbolic Neural Networks by Lipschitz analysis

This repository provides the official implementation of Lipschitz regularization of HNNs from the following paper.

```
Yuekang Li, Yidan Mao, Yifei Yang, and Dongmian Zou. 2024. Improving Robustness of Hyperbolic Neural Networks by Lipschitz Analysis. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’24), August 25–29, 2024, Barcelona, Spain. ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3637528.3671875
```


## 1. Environment
* numpy 1.21.6
* scikit-learn 0.20.3
* torch 1.8.1
* torchvision 0.9.1
* networkx 2.2
For more specific information, please see `environment.yml`.


## 2. Usage


Before training, run 

`source set_env.sh`

to create environment variables that are used in the code.


## 3. Examples
We provide examples of scripts to perform Lipschitz regularized training for node classification. In the examples below, we used a fixed random seed set to 1234 for reproducibility. Results may vary slightly based on the machine used. To reproduce results in the paper, run each script for 10 random seeds and average the results.


* 2-layer HNN for Poincaré ball with noise std=0.001

`CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --task nc --dataset texas --model HNN --lr 0.01 --dim 16 --num_layers 2 --act None --bias 0 --dropout 0.5 --weight-decay 0 --manifold PoincareBall --log-freq 1 --noise_std 0.001 --multiplier 1e-5 --lip 0 --noise_in_training 0 > test.file 2>&1 >&1 &`

* 3-layer HGCN for hyperboloid with noise std=0.01

`CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --task nc --dataset texas --model HGCN --lr 0.01 --dim 16 --num_layers 2 --act None --bias 0 --dropout 0.5 --weight-decay 0 --manifold Hyperboloid --log-freq 1 --noise_std 0.01 --multiplier 1e-5 --lip 0 --noise_in_training 0 > test.file 2>&1 >&1 &`


## 4. File Descriptions
`data/`: Datasets

`layers/`: Hyperbolic layers

`manifolds/`: Manifold calculations

`models/`: Hyperbolic models
* `models/base_model.py` contains our Lipschitz bounds calculation and implementation of Lipschitz regularized training.
* `models/encoders.py` contains the injection of noise perturbations.

`optimizers/`: Optimization on manifolds

`utils/`: Utility files

`train.py`: Training script

`config.py`: Hyperparameter settings


## Citation

```
Yuekang Li, Yidan Mao, Yifei Yang, and Dongmian Zou. 2024. Improving Robustness of Hyperbolic Neural Networks by Lipschitz Analysis. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’24), August 25–29, 2024, Barcelona, Spain. ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3637528.3671875
```

or 

```
@inproceedings{
li2024improving,
title={Improving Robustness of Hyperbolic Neural Networks by Lipschitz Analysis},
author={Li, Yuekang, Mao, Yidan, Yang, Yifei and Zou, Dongmian},
booktitle={30th SIGKDD Conference on Knowledge Discovery and Data Mining - Research Track},
year={2024}}
```

## Reference

For the construction of hyperbolic models, we utilized the code available at https://github.com/HazyResearch/hgcn.
