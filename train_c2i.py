# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import torch, pdb
import numpy as np
from PIL import Image
import os.path as osp
from tqdm import tqdm
from copy import deepcopy
from omegaconf import OmegaConf
import torch.distributed as dist
import os, time, inspect, argparse
from einops import repeat, rearrange
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from autoregressive.models.gpt import GPT_models
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from hita.engine.logger import create_logger
from hita.engine.util import instantiate_from_config
from hita.engine.ema import update_ema, requires_grad
from hita.engine.distributed import init_distributed_mode
from hita.tokenizer.tokenizer_image.vq_model import VQ_models
from hita.data.imagenet_lmdb import ImageNetLmdbDataset as ImageNetDataset
from hita.engine.misc import is_main_process, all_reduce_mean, concat_all_gather,get_world_size

#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def creat_optimizer(model, weight_decay, learning_rate, betas, logger):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer

#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    #* Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    
    #* Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        checkpoint_dir = osp.join(args.results_dir, args.gpt_model)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger_dir = osp.join(args.results_dir, 'logs')
        os.makedirs(logger_dir, exist_ok=True)
        logger = create_logger(logger_dir)
        logger.info(f"Experiment directory created at {args.results_dir}")
    else:
        logger = create_logger(None)

    #* training args & training enviroment
    logger.info(f"{args}")
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={get_world_size()}.")

    #* Setup model
    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p
    latent_size = args.image_size // args.downsample_size
    model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        num_slots=args.num_slots,
        drop_path_rate=args.drop_path_rate,
        token_dropout_p=args.token_dropout_p,
    ).to(device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    #* Setup VQ tokenizer
    transformer_config = OmegaConf.load(args.transformer_config_file)
    vq_model = VQ_models[args.vq_model](
        codebook_size = args.codebook_size,
        codebook_embed_dim = args.codebook_embed_dim,
        codebook_slots_embed_dim=args.codebook_slots_embed_dim,
        transformer_config = transformer_config,
        z_channels = args.z_channels,
        )

    vq_model.to(device)
    vq_model.freeze()
    vq_model.eval()
    
    assert osp.exists(args.vq_ckpt)
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    m1, u1 = vq_model.load_state_dict(checkpoint["ema"], strict=False)

    assert sum(['backbone' in p for p in m1]) == len(m1)
    del checkpoint

    if args.ema:
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")

    #* Setup optimizer
    optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)

    #* Setup dataset/dataloader:
    # dataset = ImageNetDataset(args.anno_file, args.image_size, True)
    dataset = ImageNetDataset(args.anno_file, args.image_size, True)

    sampler = DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    logger.info(f"Dataset contains {len(dataset):,} images ({args.anno_file}).")

    #* Prepare models for training:
    if args.gpt_ckpt:
        assert osp.exists(args.gpt_ckpt), f'Please ensure the existence of {args.gpt_ckpt}'
        checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")

        model.load_state_dict(checkpoint["model"], strict=False)
        if args.ema:
            ema.load_state_dict(checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.gpt_ckpt.split('/')[-1].split('.')[0])
        start_epoch = round(np.ceil(train_steps / int(len(dataset) / args.global_batch_size)))
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.gpt_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    
    model = DDP(model.to(device), device_ids=[args.gpu])
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    
    #* initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))

    logger.info(f"Training for {args.epochs} epochs...")
    model_dirs = osp.realpath(__file__).split('/')
    this_model_dir = '/'.join(model_dirs[2:-1])
            
    for epoch in range(start_epoch, args.epochs):

        loader.sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
       
        with tqdm(loader, dynamic_ncols=True, disable=not is_main_process()) as train_dl:

            for i, (imgs, labels) in enumerate(train_dl):
                
                imgs, labels = imgs.to(device), labels.to(device)
                bs, num_aug = imgs.shape[:2]

                if imgs.dim() > 4:
                    imgs = imgs.flatten(0, 1)
                    labels = repeat(labels, 'b -> (b f)', f = num_aug)
                else:
                    num_aug = 1

                with torch.no_grad():
                    (z_quant, _, _), _, (z_indices, q_indices) = vq_model.encode(imgs)

                z_indices, q_indices = z_indices.reshape(bs, -1), q_indices.reshape(bs, -1)

                # samples, _ = vq_model.decode_from_indices(z_indices.flatten(), q_indices.flatten(), z_quant.shape)
                # samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
                # saveDir = 'images'
                # os.makedirs(saveDir, exist_ok=True)
                # for j, sample in enumerate(samples):
                #     Image.fromarray(sample).save(f'{saveDir}/delete_{bs *i + j+1}.png')
                # pdb.set_trace()
                
                assert z_indices.shape[0] == labels.shape[0]
                with torch.cuda.amp.autocast(dtype=ptdtype):  
                    _, (loss1, loss2) = model(cond_idx=labels, idx=(q_indices, z_indices[:,:-1]), targets=(q_indices, z_indices))
                
                loss = loss1 + loss2
                # backward pass, with gradient scaling if training in fp16         
                scaler.scale(loss).backward()
                if args.max_grad_norm != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # step the optimizer and scaler if training in fp16
                scaler.step(optimizer)
                scaler.update()
                # flush the gradients as soon as we can, no need for this memory anymore
                optimizer.zero_grad(set_to_none=True)
                
                if isinstance(model, DistributedDataParallel):
                    gpt_model = model.module
                else:
                    gpt_model = model
                

                if args.ema:
                    update_ema(ema, gpt_model)
                # Log loss values:
                train_steps += 1

                #* Reduce loss over all processes:
                torch.cuda.synchronize()
                avg_loss1 = all_reduce_mean(loss1)
                avg_loss2 = all_reduce_mean(loss2)
                world_size = get_world_size()

                if (train_steps % args.log_every == 0) & is_main_process():
                    
                    train_dl.set_postfix(
                        ordered_dict={
                            "epoch"          : epoch,
                            "iters"          : train_steps,
                            'gpt-type'       : args.gpt_model,
                            "gpt_slot_loss"  : avg_loss1.item(),
                            "gpt_img_loss"   : avg_loss2.item(),
                            'this_model_dir' : this_model_dir,
                            "world_size"     : world_size,
                        }
                    )

                #* Save checkpoint.
                if (train_steps > 0) & (train_steps % args.ckpt_every == 0) & is_main_process():
         
                    checkpoint = {
                        "model": gpt_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "steps": train_steps,
                        "args": args,
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()
                    model_string_name = args.gpt_model
                    ckpt_path = osp.join(checkpoint_dir, '{}-{}.pt'.format(model_string_name, train_steps))
                    torch.save(checkpoint, ckpt_path)
                    logger.info(f"Saved checkpoint  to {ckpt_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    logger.info("Done!")
    dist.destroy_process_group()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno-file", type=str, required=True)
    parser.add_argument("--crop-range", type=float, default=1.1)
    parser.add_argument("--local-rank", type=int, default=0,)

    #* GPT model
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument('--num-slots', type=int, default=128, choices=[128, 160])
    parser.add_argument('--z-channels', type=int, default=512, choices=[256, 512])

    parser.add_argument("--ema", action='store_true', help="whether using ema training")
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path-rate", type=float, default=0.0, help="using stochastic depth decay")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default='imagenet_code')
    parser.add_argument("--image-size", type=int, choices=[256, 336, 384, 448, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 14, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 parameter for the Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.95, help="beta2 parameter for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])

    #* VQ tokenizer
    # parser.add_argument('--tokenizer-config', type=str, default='configs/vit_vqgan.yaml', required=True)
    # parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--codebook-slots-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--transformer-config-file", type=str, default='configs/vit_transformer.yaml',)
    
    args = parser.parse_args()
    main(args)
