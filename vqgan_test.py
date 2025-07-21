import json, time
import torch, timm
import numpy as np
import pandas as pd
import torch_fidelity
import os.path as osp
from glob import glob
from PIL import Image
from pathlib import Path
import argparse, datetime
import os, sys, pdb, time
from omegaconf import OmegaConf
import tensorflow.compat.v1 as tf
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from hita.evaluations.evaluator import Evaluator
from hita.data.augmentation import center_crop_arr
from hita.data.imagenet_nori import ImageNetNori as ImageNet
from hita.tokenizer.tokenizer_image.vq_model import VQ_models
from skimage.metrics import structural_similarity as ssim_loss
from skimage.metrics import peak_signal_noise_ratio as psnr_loss

from hita.engine.distributed import init_distributed_mode
from hita.engine.misc import (is_main_process, get_rank, get_world_size, concat_all_gather)

def get_args_parser():

    parser = argparse.ArgumentParser('SGM testing', add_help=False)
    parser.add_argument('--batch-size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    #* Dataset parameters
    parser.add_argument('--output_dir', default='recons',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='output/logs/',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pin_mem', action='store_false',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=False)

    #* Feature genration
    parser.add_argument('--evaluate', action='store_true', help="perform only evaluation")

    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--codebook-slots-embed-dim", type=int, default=12, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 336, 384, 448, 512, 1024], default=512)
    parser.add_argument("--transformer-config-file", type=str, default='configs/vit_transformer.yaml',)
    parser.add_argument("--z-channels", type=int, default=512,)
    parser.add_argument("--anno-file", type=str, default='imagenet/imagenet.val.oss.list')
    
    return parser

def build_imagenet_dataloader(anno_file, num_tasks, local_rank, args):

    assert osp.exists(anno_file)
    transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])

    samples = []
    with open(anno_file, "r") as f:
        for line in f:
            nori_id, target = line.strip().split()
            samples.append((nori_id, int(target)))

    num = len(samples)
    stride = int(np.ceil(num / num_tasks))
    s, t = local_rank * stride, min(num, (local_rank + 1) * stride)
    
    dataset = ImageNet(samples=samples[s:t], transform=transform,)

    return dataset

def imagenet_eval(args, gtDir = 'imagenet/imagenet-val'):

    name = osp.basename(args.vq_ckpt).split('.')[0]
    #* Perform evaluation on the generated images.
    gtDir = 'imagenet/imagenet-val'
    gen_names = os.listdir(args.output_dir)
    img_names = os.listdir(gtDir)

    assert len(gen_names) == len(img_names), \
        f"generate only {len(gen_names)} images, while there are {len(img_names)} in total!"

    metrics_dict = torch_fidelity.calculate_metrics(
        input1=args.output_dir,
        input2=gtDir,
        cuda=True,
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        verbose=True,
    )
    fid_score = metrics_dict['frechet_inception_distance']
    inception_score = metrics_dict['inception_score_mean']
    print('FID:{:.4f}, IS: {:.4f}'.format(fid_score, inception_score))
    with open('results.md', 'a') as fid:
        fid.write(f'\n{name}\n')
        fid.write('FID:{:.4f}, IS: {:.4f}\n'.format(fid_score, inception_score))

def main(args):

    init_distributed_mode(args)
    print('job dir: {}'.format(osp.dirname(osp.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = False

    num_tasks = get_world_size()
    global_rank = get_rank()
    
    print(f'num_tasks: {num_tasks}')
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # initialize the data loader
    dataset = build_imagenet_dataloader(args.anno_file, num_tasks, global_rank, args)
    
    data_loader = DataLoader(
        dataset,  
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False, drop_last=False,)

    transformer_config = OmegaConf.load(args.transformer_config_file)
    model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        codebook_slots_embed_dim=args.codebook_slots_embed_dim,
        transformer_config = transformer_config,
        z_channels=args.z_channels)

    model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")

    if "ema" in checkpoint:  # ema
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    missings, unexpected = model.load_state_dict(model_weight, strict=False)
    assert sum(['backbone' in p for p in missings]) == len(missings), 'Please check the state_dict since necessary parameters are missed.'

    del checkpoint
    
    #* Set arguments generation params
    model.to(device)

    #* Log parameters
    if is_main_process():
        print('#DataLoader: {}, num_tasks: {}'.format(len(data_loader), num_tasks))

    num_protos, num_slots, eps = 16384, 16384, 1e-6
    img_indices, slot_indices, (samples, gt, psnr_val_rgb, ssim_val_rgb) = gen_images(model, data_loader, device, args)
    
    if is_main_process():
        samples = np.stack(samples, axis=0)
        gt = np.stack(gt, axis=0)

        print(f'len(samples):{samples.shape[0]}, len(gt): {len(gt)}')
        config = tf.ConfigProto(
                allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
        )
        config.gpu_options.allow_growth = True

        evaluator = Evaluator(tf.Session(config=config),batch_size=64)
        evaluator.warmup()
        print("computing reference batch activations...")
        ref_acts = evaluator.read_activations(gt)
        print("computing/reading reference batch statistics...")
        ref_stats, _ = evaluator.read_statistics(gt, ref_acts)
        print("computing sample batch activations...")
        sample_acts = evaluator.read_activations(samples)
        print("computing/reading sample batch statistics...")
        sample_stats, _ = evaluator.read_statistics(samples, sample_acts)
        FID = sample_stats.frechet_distance(ref_stats)

        IS = evaluator.compute_inception_score(sample_acts[0])

        print(f"rFID: {FID:04f}, rIS: {IS:04f}.")

        usage_img = img_indices.size(0) / num_protos
        usage_slot = slot_indices.size(0) / num_slots

        psnr_val_rgb = sum(psnr_val_rgb) / (len(psnr_val_rgb) + eps)
        ssim_val_rgb = sum(ssim_val_rgb) / (len(ssim_val_rgb) + eps)
        print('usage_img:{:.4f}, usage_slot: {:.4f},  psnr: {:.4f}, ssim: {:.4f}.'.format(usage_img, usage_slot, psnr_val_rgb, ssim_val_rgb))
        filename = osp.basename(args.vq_ckpt).split('.')[0]
        with open('results.md', 'a') as fid:
            fid.write(f'\n{filename}:\n')
            fid.write(f'rFID: {FID:04f}, rIS: {IS:04f}.\n')
            fid.write('usage_img:{:.4f}, usage_slot: {:.5f}, PSNR: {:.4f}, SSIM: {:.4f}.\n'.format(usage_img, usage_slot, psnr_val_rgb, ssim_val_rgb))

@torch.no_grad()
def gen_images(model, dataloader, device, args):

    model.eval()
    saveDir = args.output_dir
    prev, total = 0, len(dataloader)

    this_model_dir = osp.dirname(osp.realpath(__file__))
    
    img_indices = torch.Tensor([]).to(device)
    slot_indices = torch.Tensor([]).to(device)
    psnr_val_rgb, ssim_val_rgb = [], []
    samples, gt = [], []
    times, count = [], 0
    for i, (images, targets) in enumerate(dataloader):
        
        images = images.to(device)
        bs = images.size(0)
        (gen_imgs, _, _), _, (z_indices, q_indices) = model(images)


        gen_images = concat_all_gather(gen_imgs)
        img_indices = torch.unique(torch.cat((img_indices, concat_all_gather(z_indices))))
        slot_indices = torch.unique(torch.cat((slot_indices, concat_all_gather(q_indices))))

        images = concat_all_gather(images)
        slot_indices = torch.unique(torch.cat((slot_indices, concat_all_gather(q_indices))))

        gen_images = torch.clamp(127.5 * gen_images.permute(0, 2, 3, 1) + 128.0, 0, 255).to('cpu').numpy()
        images = torch.clamp(127.5 * images.permute(0, 2, 3, 1) + 128.0, 0, 255).to('cpu').numpy()

        gemini = np.concatenate((images, gen_images), axis=2)
        if is_main_process():
            print('{}, iter-{}/{}, gen_imgs.shape:{}'.format(this_model_dir, i, total, gen_images.shape))
            for k, re in enumerate(gen_images):

                rec = Image.fromarray(np.uint8(re))
                img = Image.fromarray(np.uint8(images[k]))
                # rec = Image.fromarray(np.uint8(gemini[k]))
                
                # pdb.set_trace()
                rec = rec.resize((256, 256))
                img = img.resize((256, 256))


                rgb_restored = np.array(rec).astype(np.float32) / 255. # rgb_restored value is between [0, 1]
                rgb_gt = np.array(img).astype(np.float32) / 255.
                psnr = psnr_loss(rgb_restored, rgb_gt)
                ssim = ssim_loss(rgb_restored, rgb_gt, multichannel=True, data_range=2.0, channel_axis=-1)
                psnr_val_rgb.append(psnr)
                ssim_val_rgb.append(ssim)

                samples.append(np.array(rec))
                gt.append(np.array(img))

    return img_indices, slot_indices, (samples, gt, psnr_val_rgb, ssim_val_rgb)

if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()
 
    main(args)
