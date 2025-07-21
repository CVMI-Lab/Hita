# Modified from:
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample_ddp.py
import argparse
import numpy as np
import math, torch
from tqdm import tqdm
from PIL import Image
import os.path as osp
import torch, os, pdb, time
from omegaconf import OmegaConf
import torch.nn.functional as F
import torch.distributed as dist
import tensorflow.compat.v1 as tf
torch.backends.cudnn.allow_tf32 = True
from evaluations.c2i.evaluator import Evaluator
torch.backends.cuda.matmul.allow_tf32 = True
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate
from hita.engine.util import instantiate_from_config
from hita.tokenizer.tokenizer_image.vq_model import VQ_models
from hita.engine.misc import (is_main_process, get_rank, get_world_size,
                                    init_distributed_mode, concat_all_gather)


def main(args):
    
    # Setup PyTorch:
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={get_world_size()}.")

    #* Create and load model
    transformer_config = OmegaConf.load(args.transformer_config_file)
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        codebook_slots_embed_dim=args.codebook_slots_embed_dim,
        transformer_config = transformer_config,
        z_channels=args.z_channels)

    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    m1, u1 = vq_model.load_state_dict(checkpoint["ema"], strict=False)
    assert sum(['backbone' in p for p in m1]) == len(m1) 
    assert len(u1) < 1
    del checkpoint

    vq_model.to(device)
    vq_model.eval()
    vq_model.freeze()

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        num_slots=args.num_slots,
    )

    print(f'Loading pre-trained GPT model {args.gpt_ckpt}.')
    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    if args.from_fsdp: # fsdp
        model_weight = checkpoint
    elif 'ema' in checkpoint:
        model_weight = checkpoint['ema']
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint['ema']
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight, maybe add --from-fsdp to run command")

    m1, u1 = gpt_model.load_state_dict(model_weight, strict=True)
    
    gpt_model.eval()
    gpt_model.to(device=device, dtype=precision)
    del checkpoint

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True,
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no model compile") 

    # Create folder to save samples:
    model_string_name = args.gpt_model.replace("/", "-")
    if args.from_fsdp:
        ckpt_string_name = args.gpt_ckpt.split('/')[-2]
    else:
        ckpt_string_name = osp.basename(args.gpt_ckpt).replace(".pth", "").replace(".pt", "")
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-size-{args.image_size_eval}-ViT-VQGAN-" \
                  f"topk-{args.top_k}-topp-{args.top_p}-temperature-{args.temperature}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(args.sample_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)

    if is_main_process():
        print(f"Total number of images that will be sampled: {total_samples}")
        
    assert total_samples % get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total, count = 0, 0
    gen_samples = []
    for idx in pbar:
        # Sample inputs:
        c_indices = torch.randint(0, args.num_classes, (n,), device=device)
        qzshape = [n, args.codebook_embed_dim, latent_size, latent_size]


        if idx > 20:
            tic = time.time()
        index_sample = generate(
            gpt_model, c_indices, args.num_slots + latent_size ** 2,
            cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
            temperature=args.temperature, top_k=args.top_k,
            top_p=args.top_p, sample_logits=True, 
            )
        sizes = [args.num_slots, index_sample.size(1) - args.num_slots]
        q_indices, z_indices = index_sample.split(sizes, dim=1)
        samples, _ = vq_model.decode_from_indices(z_indices.flatten(), q_indices.flatten(), qzshape)

        if args.image_size_eval != args.image_size:
            samples = F.interpolate(samples, size=(args.image_size_eval, args.image_size_eval), mode='bicubic')
        
        samples = concat_all_gather(samples)
        if idx > 20:
            total += time.time() - tic
            count += 1
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        gen_samples.append(samples)

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if is_main_process():

        gen_samples = np.concatenate(gen_samples, axis=0)[:50_000]
        print(f'generated samples: {gen_samples.shape[0]}')

        config = tf.ConfigProto(
                allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
        )
        config.gpu_options.allow_growth = True

        evaluator = Evaluator(tf.Session(config=config),batch_size=64)
        evaluator.warmup()


        print("computing reference batch activations...")
        ref_acts = evaluator.read_activations(args.ref_batch)
        print("computing/reading reference batch statistics...")
        ref_stats, ref_stats_spatial = evaluator.read_statistics(args.ref_batch, ref_acts)
        

        print("computing sample batch activations...")
        sample_acts = evaluator.read_activations(gen_samples)
        print("computing/reading sample batch statistics...")
        sample_stats, sample_stats_spatial = evaluator.read_statistics(samples, sample_acts)
        FID = sample_stats.frechet_distance(ref_stats)
        sFID = sample_stats_spatial.frechet_distance(ref_stats_spatial)

        IS = evaluator.compute_inception_score(sample_acts[0])
        prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])
        speed = total / max(1e-6, count * args.per_proc_batch_size)
        print("Computing evaluations...")
        print("Inception Score:", IS)
        print("FID:", FID)
        print("sFID:", sFID)
        print("Precision:", prec)
        print("Recall:", recall)
        print('Speed: ', speed)
        print('Batch-size: ', args.per_proc_batch_size)
        print('Total: ', total)
        print('count:', count)
        txt_path = sample_folder_dir + '.txt'
        print("writing to {}".format(txt_path))
        with open(txt_path, 'w') as f:
            print("Inception Score:", IS, file=f)
            print("FID:", FID, file=f)
            print("sFID:", sFID, file=f)
            print("Precision:", prec, file=f)
            print("Recall:", recall, file=f)
            print('Speed:', speed, file=f)

        print("Done.")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #* GPT architecture
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument('--num-slots', type=int, default=128, choices=[128, 160])
    parser.add_argument("--compile", action='store_true', default=False)

    #* VQ tokenizer
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--transformer-config-file", type=str, default='configs/vit_transformer.yaml',)
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--codebook-slots-embed-dim", type=int, default=8, help="codebook dimension for slots quantization")
    parser.add_argument("--z-channels", type=int, default=512,)

    #* Dataset settings
    parser.add_argument("--image-size", type=int, choices=[256, 336, 384, 512], default=384)
    parser.add_argument("--image-size-eval", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=0,help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--ref-batch", type=str, default='imagenet/VIRTUAL_imagenet256_labeled.npz', help="path to reference batch npz file")

    args = parser.parse_args()
    main(args)