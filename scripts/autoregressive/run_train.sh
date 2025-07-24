# !/bin/bash
mkdir -p output/logs
model_type='GPT-L'
export NODE_COUNT=1
export NODE_RANK=0
export PROC_PER_NODE=4
scripts/torchrun.sh  \
    train_c2i.py --gpt-type c2i --image-size 336 --gpt-model ${model_type} --downsample-size 16 --num-workers 4     \
    --anno-file imagenet/lmdb/train_lmdb --global-batch-size 64 --ckpt-every 10000 --ema --log-every 1             \
    --results-dir output/vanilla --vq-ckpt pretrained_models/hita-ultra.pt --epochs 300 --codebook-embed-dim 8  \
    --codebook-slots-embed-dim 12 --transformer-config-file configs/hita_vqgan_ultra.yaml --mixed-precision bf16    \
    --lr 1e-4 2>&1 | tee 'train.log'
