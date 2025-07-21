import socket
import numpy as np
import pickle as pkl
import os.path as osp
from PIL import Image
from glob import glob
import torch, time, io, boto3
import os, sys, pdb, argparse, megfile
from dataset.register import set_aws_a
from single_test import create_npz_from_sample_folder
from evaluations.c2i.evaluator import evaluate_fid, main
from paintmind.data.imagenet_nori import ImageNet
# from dataset.imagenet_nori import ImageNet, ImageNetNori, ImageNetNoriDataset

# s3://whc/imagenet1k/
def _find_free_port():
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

def create_npz_from_target_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    img_names = os.listdir(sample_dir)
    cur_name = sample_dir.split('/')[-1]
    for i in tqdm(range(len(img_names)), desc="Building .npz file from samples"):
        img_name = img_names[i]
        sample_pil = Image.open(osp.join(sample_dir, img_name))
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)

    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{cur_name}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def readlines(fpath):

    assert osp.exists(fpath), f"Please ensure the existence of {fpath}"
    samples = []
    with open(fpath, 'r') as fid:
        for line in fid.readlines():
            dirname, target, _ =  line.strip('\n').split(' ')
            samples.append((dirname, target))
    return samples

def gen_evaluate():

    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_batch", help="path to reference batch npz file")
    parser.add_argument("--sample_batch", help="path to sample batch npz file")

    args = parser.parse_args()
    if not osp.exists(args.sample_batch):
        dirpath = args.sample_batch.replace('.npz', '')
        assert osp.exists(dirpath)
        create_npz_from_sample_folder(dirpath)
    main(args)

def nori_demo():

    set_aws_a()
    # host = "http://oss.i.basemind.com"
    # s3_client = boto3.client('s3', endpoint_url=host)

    # os.environ['OSS_ENDPOINT'] = 'http://oss.i.basemind.com'
    # os.environ['AWS_ACCESS_KEY_ID'] = 'cbd9cc9eaa437b626dc3973f5220f7a3'
    # os.environ['AWS_SECRET_ACCESS_KEY'] = 'd409e0bfd0027e021b6b7707eeb3de3f'

    images = []
    oss = 's3://whc/imagenet1k/val/'
    samples = readlines('imagenet/map_clsloc.txt')

    formats = ['jpg', 'jpeg', 'JPEG', '.bmp']
    for i, (dirname, label) in enumerate(samples):
        dirpath = osp.join(oss, dirname)
        print(f'{i}-th/{len(samples)}: {dirpath}')
        names = []
        for fmt in formats:
            names.extend(megfile.smart_glob(f'{dirpath}/{dirname}_*.{fmt}'))
        
        for name in names:
            images.append((name, label))
    
    print('total imagenet samples is: {}'.format(len(images)))
    with open('imagenet.val.oss.list', 'w') as fid:
        for img_file, label in images:
            line = f'{img_file} {label}\n'
            fid.write(line)

def batch_gen_evaluation():


    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_batch", default='imagenet/VIRTUAL_imagenet256_labeled.npz', help="path to reference batch npz file")
    parser.add_argument("--sample_batch", default=None, help="path to sample batch npz file")

    args = parser.parse_args()

    sources = glob('final-chapter/*')
    finished = []
    histc = 'fid-finished.pkl'
    if osp.exists(histc):
        with open(histc, 'rb') as fid:
            finished = pkl.load(fid)
    
    for src_dir in sources:
        if not osp.isdir(src_dir):
            continue
        images = glob(osp.join(src_dir, '*.png'))
        if len(images) < 50_000:
            continue
        if src_dir in finished:
            continue
        finished.append(src_dir)
        create_npz_from_sample_folder(src_dir)
        args.sample_batch = src_dir + '.npz'
        main(args)
        os.remove(args.sample_batch)
        
    with open('fid-finished.pkl', 'wb') as fid:
        pkl.dump(finished, fid)
    print('Finished...')

if __name__ == '__main__':
    
    # gen_evaluate()
    anno_file = 'imagenet/imagenet.val.nori.list'
    loader = ImageNet(anno_file)
    for i, (image, label, _) in enumerate(loader):
        pdb.set_trace()
