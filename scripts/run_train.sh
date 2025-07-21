# !/bin/bash
set -x

# torchrun \
# 	--nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank \
# 	--master_addr=$master_addr --master_port=$master_port \
# 	autoregressive/train/train_c2i.py "$@"

# torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr 100.100.119.85 --master_port 42021 \
# NCCL_IB_HCA=$(pushd /sys/class/infiniband/ > /dev/null; for i in mlx5_*; do cat $i/ports/1/gid_attrs/types/* 2>/dev/null | grep v >/dev/null && echo $i ; done; popd > /dev/null)
# export NCCL_IB_HCA
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_TC=106
# torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr 100.100.119.85 --master_port 42019 \
NCCL_IB_HCA=$(pushd /sys/class/infiniband/ > /dev/null; for i in mlx5_*; do cat $i/ports/1/gid_attrs/types/* 2>/dev/null | grep v >/dev/null && echo $i ; done; popd > /dev/null)
export NCCL_IB_HCA
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=106
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr 100.100.113.138  --master_port 42019 \
    	train_c2i.py --cloud-save-path output --num-workers 2 --global-batch-size 256 \
		--code-path codes/imagenet/imagenet_code_c2i_flip_ten_crop --epochs 50 \
		--vq-ckpt ./models/vq_ds16_c2i.pt  \
		--anno-file '/home/zhenganlin/june/datasets/imagenet/imagenet.train.nori.list' --ten-crop \
		--global-batch-size 8 --mixed-precision fp16 \
		--image-size 384 --log-every 20 --gpt-model GPT-B 2>&1 | tee 'train.log'
