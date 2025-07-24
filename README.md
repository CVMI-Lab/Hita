
# Holistic Tokenizer for Autoregressive Image Generation <br><sub>Official PyTorch Implementation</sub>

[![arXiv](https://img.shields.io/badge/arXiv-2507.02358-b31b1b.svg)](https://arxiv.org/abs/2507.02358)&nbsp;
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Hita-yellow)](https://huggingface.co/yexiguafu/hita-gen)&nbsp;


<p align="center">
<img src="assets/application.png" width=95%>
<p>

This is a PyTorch/GPU implementation of the paper **Holistic Tokenizer for Autoregressive Image Generation**:

This repo contains:

* 🪐 A simple PyTorch implementation of Hita tokenizer and various AR generative models.
* ⚡️ Pre-trained Hita tokenizers and AR generative models trained on ImageNet.
* 🛸 Training and evaluation scripts for tokenizer and generative models, which were also provided in [here](./scripts).
* 🎉 Hugging Face for easy access to pre-trained models.


## Release

- [2025/07/24] 🔥 [Image tokenizers](https://huggingface.co/yexiguafu/hita-gen/tree/main) and [AR models](https://huggingface.co/yexiguafu/hita-gen/tree/main) for class-conditional image generation are released. 🔥
- [2025/07/24] 🔥 All codes of Hita have been released. 🔥
- [2024/07/03] 🔥 **Hita** has been released. Checkout the [paper](https://arxiv.org/pdf/2410.09575) for details.🔥
- [2025/06/26] 🔥 **Hita has been accepted by ICCV 2025!** 🔥


## Contents
- [Install](#install)
- [Model Zoo](#model-zoo)
- [Performance](#performance)
- [Train](#train)
- [Evaluation](#evaluation)

## Install

If you are not using Linux, do *NOT* proceed.

1. Clone this repository and navigate to Hita folder
```bash
git clone https://github.com/CVMI-Lab/Hita.git
cd Hita
```

2. Install Package
```Shell
conda create -n hita python=3.10 -y
conda activate hita
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases as required.
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Model Zoom

In this repo, we release:
* Two image tokenizers: Hita-V(anilla) and Hita-U(ltra).
* Class-conditional autoregressive generative models ranging from 100M to 3B parameters.

### 1. VQ-VAE models
In this repo, we release two image tokenizers: Hita-V(anilla) and Hita-U(ltra). Hita-V is utilized in the original paper, while Hita-U is an updated version that uses more advanced techniques, such as the DINO discriminator and the learning objective of pre-trained vision foundation model reconstruction proposed in [VFMTok](https://arxiv.org/pdf/2507.08441), which exhibits better image reconstruction and generation quality. 

Method | tokens | rFID (256x256) | rIS (256x256)    | weight
---    | :---:  |:---:|:---:   | :---: 
Hita-V |  569   | 1.03  | 198.5   | [hita-vanilla.pt](https://huggingface.co/yexiguafu/hita-gen/blob/main/vanilla/tokenizer/hita-tok.pt)
Hita-U |  569   | **0.57**  | **221.8**   | [hita-ultra.pt](https://huggingface.co/yexiguafu/hita-gen/blob/main/ultra/tokenizer/hita-ultra.pt)

### 2. AR generation models with Hita-V
Method   | params | epochs | FID  |  IS   | weight 
---      | :---:  | :---:  | :---:|:---:  |:---:|
HitaV-B  | 111M   |   50   | 5.85 | 212.3 | [HitaV-B-50e.pt](https://huggingface.co/yexiguafu/hita-gen/blob/main/vanilla/GPT-B/GPT-B-50e.pt)
HitaV-B  | 111M   |  300   | 4.33 | 238.9 | [HitaV-B-50e.pt](https://huggingface.co/yexiguafu/hita-gen/blob/main/vanilla/GPT-B/GPT-B-300e.pt)
HitaV-L  | 343M   |   50   | 3.75 | 262.1 | [HitaV-L-50e.pt](https://huggingface.co/yexiguafu/hita-gen/blob/main/vanilla/GPT-L/GPT-L-50e.pt)
HitaV-L  | 343M   |  300   | 2.86 | 267.3 | [HitaV-L-50e.pt](https://huggingface.co/yexiguafu/hita-gen/blob/main/vanilla/GPT-L/GPT-L-300e.pt)
HitaV-XL | 775M   |   50   | 2.98 | 253.4 | [HitaV-XL-50e.pt](https://huggingface.co/yexiguafu/hita-gen/blob/main/vanilla/GPT-XL/GPT-XL-50e.pt)
HitaV-XXL| 1.4B   |   50   | 2.70 | 274.8 | [HitaV-XXL-50e.pt](https://huggingface.co/yexiguafu/hita-gen/blob/main/vanilla/GPT-XXL/GPT-XXL-50e.pt)
HitaV-2B | 2.0B   |   50   | 2.59 | 281.9 | [HitaV-2B-50e.pt](https://huggingface.co/yexiguafu/hita-gen/blob/main/vanilla/GPT-2B/GPT-2B-50e.pt)

### 3. AR generation with Hita-U
Method  | params | epochs  | FID |  IS | weight 
---     |:---:|:---:| :---: | :---: |:---:|
HitaU-B  | 111M | 50  | 4.21 | 229.0 | [HitaU-B-50e.pt](https://huggingface.co/yexiguafu/hita-gen/blob/main/ultra/GPT-B/GPT-B-50e.pt)
HitaU-B  | 111M | 250 | 3.49 | 237.5 | [HitaU-B-250e.pt](https://huggingface.co/yexiguafu/hita-gen/blob/main/ultra/GPT-B/GPT-B-250e.pt)
HitaU-L  | 343M | 50  | 2.97 | 273.3 | [HitaU-L-50e.pt](https://huggingface.co/yexiguafu/hita-gen/blob/main/ultra/GPT-L/GPT-L-50e.pt)
HitaU-L  | 343M | 250 | 2.44 | 274.6 | [HitaU-L-250e.pt](https://huggingface.co/yexiguafu/hita-gen/blob/main/ultra/GPT-L/GPT-L-250e.pt)
HitaU-XL | 775M | 50  | 2.40 | 276.3 | [HitaU-XL-50e.pt](https://huggingface.co/yexiguafu/hita-gen/blob/main/ultra/GPT-XL/GPT-XL-50e.pt)
HitaU-XL | 775M | 100 | 2.16 | 275.3 | [HitaU-XL-100e.pt](https://huggingface.co/yexiguafu/hita-gen/blob/main/ultra/GPT-XL/GPT-XL-100e.pt)
HitaU-XXL| 1.4B | 50  | 2.07 | 273.8 | [HitaU-XXL-50e.pt](https://huggingface.co/yexiguafu/hita-gen/blob/main/ultra/GPT-XXL/GPT-XXL-50e.pt)
HitaU-XXL| 1.4B | 100 | 2.01 | 276.4 | [HitaU-XXL-100e.pt](https://huggingface.co/yexiguafu/hita-gen/blob/main/ultra/GPT-XXL/GPT-XXL-100e.pt)
HitaU-2B | 2.0B | 50  | 1.93 | 286.0 | [HitaU-2B-50e.pt](https://huggingface.co/yexiguafu/hita-gen/blob/main/ultra/GPT-2B/GPT-2B-50e.pt)
HitaU-2B | 2.0B | 50  | 1.82 | 282.9 | [HitaU-2B-100e.pt](https://huggingface.co/yexiguafu/hita-gen/blob/main/ultra/GPT-2B/GPT-2B-100e.pt)

### 4. AR generation with CFG-free guidance
Once the pre-trained VFM features and the original image reconstruction are simultaneously conducted, we found that the trained Hita-U(ltra), when integrated into the AR generation models, can achieve image generation without CFG-guidance.

Method  | params | epochs  | FID |  IS| weight 
---     | :---: |:---:| :---:|:---:  |:---:|
HitaU-B  | 111M | 50  | 8.32 | 108.5 | [HitaU-B-50e.pt](https://huggingface.co/yexiguafu/hita-gen/blob/main/ultra/GPT-B/GPT-B-50e.pt)
HitaU-B  | 111M | 250 | 5.19 | 138.9 | [HitaU-B-250e.pt](https://huggingface.co/yexiguafu/hita-gen/blob/main/ultra/GPT-B/GPT-B-250e.pt)
HitaU-L  | 343M | 50  | 3.96 | 151.8 | [HitaU-L-50e.pt](https://huggingface.co/yexiguafu/hita-gen/blob/main/ultra/GPT-L/GPT-L-50e.pt)
HitaU-L  | 343M | 250 | 2.46 | 188.9 | [HitaU-L-250e.pt](https://huggingface.co/yexiguafu/hita-gen/blob/main/ultra/GPT-L/GPT-L-250e.pt)
HitaU-XL | 775M | 50  | 2.66 | 178.9 | [HitaU-XL-50e.pt](https://huggingface.co/yexiguafu/hita-gen/blob/main/ultra/GPT-XL/GPT-XL-50e.pt)
HitaU-XL | 775M | 100 | 2.21 | 195.8 | [HitaU-XL-100e.pt](https://huggingface.co/yexiguafu/hita-gen/blob/main/ultra/GPT-XL/GPT-XL-100e.pt)
HitaU-XXL| 1.4B | 50  | 2.21 | 196.0 | [HitaU-XXL-50e.pt](https://huggingface.co/yexiguafu/hita-gen/blob/main/ultra/GPT-XXL/GPT-XXL-50e.pt)
HitaU-XXL| 1.4B | 100 | 1.84 | 217.2 | [HitaU-XXL-100e.pt](https://huggingface.co/yexiguafu/hita-gen/blob/main/ultra/GPT-XXL/GPT-XXL-100e.pt)
HitaU-2B | 2.0B | 50  | 1.97 | 208.6 | [HitaU-2B-50e.pt](https://huggingface.co/yexiguafu/hita-gen/blob/main/ultra/GPT-2B/GPT-2B-50e.pt)
HitaU-2B | 2.0B | 50  | 1.69 | 233.0 | [HitaU-2B-100e.pt](https://huggingface.co/yexiguafu/hita-gen/blob/main/ultra/GPT-2B/GPT-2B-100e.pt)

## Training

### 1. Preparation

1. Download the [DINOv2-L](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth) pre-trained foundation model from the official [model zoo](https://github.com/facebookresearch/dinov2).
2. Create symbolic links that point from the locations of the pretrained DINOv2-L model and the ImageNet training dataset folders to this directory.
3. Create dataset script for your own dataset. Here, we provide a template for training tokenizers and AR generative models using the ImageNet dataset in [LMDB](https://www.symas.com/mdb) format.

```bash
ln -s DINOv2-L_folder init_models
ln -s ImageNetFolder imagenet
```

### 2.Hita Tokenizer Training

1. Training Hita-V tokenizer:

```bash
export NODE_COUNT=1
export NODE_RANK=0
export PROC_PER_NODE=8
scripts/torchrun.sh vq_train.py --image-size 336 --results-dir output --mixed-precision bf16 --codebook-embed-dim 8 --disc-type patchgan  \
    --data-path imagenet/lmdb/train_lmdb --global-batch-size 256 --num-workers 4 --ckpt-every 5000 --epochs 50 --log-every 1 --lr 1e-4    \
    --transformer-config configs/hita_vqgan.yaml --ema --z-channels 512
```

2. Training Hita-V tokenizer:

```bash
scripts/torchrun.sh vq_train.py --image-size 336 --results-dir output --mixed-precision bf16 --codebook-embed-dim 8 --disc-type dinogan  \
    --data-path imagenet/lmdb/train_lmdb --global-batch-size 256 --num-workers 4 --ckpt-every 5000 --epochs 50 --log-every 1 --lr 1e-4   \
    --transformer-config configs/hita_vqgan_ultra.yaml --ema --z-channels 512 --enable-vfm-recon
```

### 3. AR generative model training

1. Training AR generative models

```bash
model_type='GPT-L' # 'GPT-B' 'GPT-XL' 'GPT-XXL' 'GPT-2B'
scripts/torchrun.sh  \
    train_c2i.py --gpt-type c2i --image-size 336 --gpt-model ${model_type} --downsample-size 16 --num-workers 4     \
    --anno-file imagenet/lmdb/train_lmdb --global-batch-size 256 --ckpt-every 10000 --ema --log-every 1             \
    --results-dir output/vanilla --vq-ckpt pretrained_models/hita-tok.pt --epochs 300 --codebook-embed-dim 8        \
    --codebook-slots-embed-dim 12 --transformer-config-file configs/hita_vqgan.yaml --mixed-precision bf16 --lr 1e-4
```

2. Resume from an AR generative checkpoint
```bash
model_type='GPT-L'
scripts/torchrun.sh  \
    train_c2i.py --gpt-type c2i --image-size 336 --gpt-model ${model_type} --downsample-size 16 --num-workers 4     \
    --anno-file imagenet/lmdb/train_lmdb --global-batch-size 270 --ckpt-every 10000 --ema --log-every 1             \
    --results-dir output/vanilla --vq-ckpt pretrained_models/hita-tok.pt --epochs 300 --codebook-embed-dim 8        \
    --codebook-slots-embed-dim 12 --transformer-config-file configs/hita_vqgan.yaml --mixed-precision bf16          \
    --lr 1e-4 --gpt-ckpt output/vanilla/${model_type}/${model_type}-{ckpt_name}.pt
```

### 4. Evaluation (ImageNet 256x256)

1. Evaluated a pretrained Hita-V tokenizer

```bash
scripts/torchrun.sh  \
        vqgan_test.py --vq-model VQ-16 --image-size 336 --output_dir recons --batch-size 50   \
        --transformer-config-file configs/hita_vqgan.yaml --z-channels 512                    \
        --vq-ckpt pretrained_models/hita-tok.pt
```

2. Evaluate a pretrained Hita-U tokenizer:

```bash
scripts/torchrun.sh  \
        vqgan_test.py --vq-model VQ-16 --image-size 336 --output_dir recons --batch-size 50   \
        --transformer-config-file configs/hita_vqgan_ultra.yaml --z-channels 512              \
        --vq-ckpt pretrained_models/hita-ultra.pt
```

3. Evaluate a pretrained AR generative model

```bash
model_type='GPT-L' # 'GPT-B' 'GPT-XL' 'GPT-XXL' 'GPT-2B'
scripts/torchrun.sh  \
         test_net.py --vq-ckpt pretrained_models/hita-ultra.pt --gpt-ckpt output/ultra/${model_type}/${model_type}-$1.pt      \
         --num-slots 128 --gpt-model ${model_type} --image-size 336 --compile --sample-dir samples --cfg-scale $2             \
         --image-size-eval 256 --precision bf16 --per-proc-batch-size $3 --codebook-embed-dim 8 --codebook-slots-embed-dim 12 \
         --transformer-config-file configs/hita_vqgan_ultra.yaml
```
## Citation

If you find Hita useful for your research and applications, please kindly cite using this BibTeX:
```
@article{zheng2025holistic,
  title={Holistic Tokenizer for Autoregressive Image Generation},
  author={Zheng, Anlin and Wang, Haochen and Zhao, Yucheng and Deng, Weipeng and Wang, Tiancai and Zhang, Xiangyu and Qi, Xiaojuan},
  journal={arXiv preprint arXiv:2507.02358},
  year={2025}
}
@article{zheng2025vision,
  title={Vision Foundation Models as Effective Visual Tokenizers for Autoregressive Image Generation},
  author={Zheng, Anlin and Wen, Xin and Zhang, Xuanyang and Ma, Chuofan and Wang, Tiancai and Yu, Gang and Zhang, Xiangyu and Qi, Xiaojuan},
  journal={arXiv preprint arXiv:2507.08441},
  year={2025}
}
```

## License
The majority of this project is licensed under MIT License. Portions of the project are available under separate license of referred projects, detailed in corresponding files.


## Acknowledgement

Our codebase builds upon several excellent open-source projects, including [LlamaGen](https://github.com/FoundationVision/LlamaGen) and [Paintmind](https://github.com/Qiyuan-Ge/PaintMind). We are grateful to the communities behind them.

## Contact
This codebase has been cleaned up but has not undergone extensive testing. If you encounter any issues or have questions, please open a GitHub issue. We appreciate your feedback!