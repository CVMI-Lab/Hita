# !/bin/bash
export NODE_COUNT=1
export NODE_RANK=0
export PROC_PER_NODE=8
/jobutils/scripts/torchrun.sh  \
        vqgan_test.py --vq-model VQ-16 --image-size 336 --output_dir recons --batch-size $2   \
        --transformer-config-file configs/hita_vqgan_ultra.yaml --z-channels 512              \
        --vq-ckpt pretrained_models/hita-ultra.pt  2>&1 | tee 'test_vq.log'
