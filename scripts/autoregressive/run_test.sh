# !/bin/bash
export NODE_COUNT=1
export NODE_RANK=0
export PROC_PER_NODE=8
model_type='GPT-L'
scripts/torchrun.sh  \
         test_net.py --vq-ckpt pretrained_models/hita-tok.pt --gpt-ckpt output/vanilla/${model_type}/${model_type}-$1e.pt  \
         --num-slots 128 --gpt-model ${model_type} --image-size 336 --compile --sample-dir ultra-samples --cfg-scale $2   \
         --image-size-eval 256 --precision bf16 --per-proc-batch-size $3 --codebook-embed-dim 8    \
         --transformer-config-file configs/hita_vqgan.yaml --codebook-slots-embed-dim 12 2>&1 | tee 'hello.log'
