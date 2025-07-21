# !/bin/bash
mkdir -p output/logs
export NODE_COUNT=$1
export NODE_RANK=$2
export PROC_PER_NODE=8
export MASTER_PORT=65519
/jobutils/scripts/torchrun.sh  \
    vq_train.py --image-size 336 --results-dir output --mixed-precision bf16 --codebook-embed-dim 8 --disc-type dinogan  \
    --data-path imagenet/imagenet.train.oss.list --global-batch-size 64 --num-workers 4 --ckpt-every 5000 --epochs 50     \
    --transformer-config configs/hita_vqgan_ultra.yaml --log-every 1 --lr 1e-4 --ema --z-channels 512 \
    --enable-vfm-recon 2>&1 | tee 'debug.log'