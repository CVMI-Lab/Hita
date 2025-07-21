import lmdb, torch
import numpy as np
import os.path as osp
from PIL import Image
import io, pickle, pdb
from torchvision import transforms
from torch.utils.data import Dataset
from .augmentation import random_crop_arr, center_crop_arr

class LMDBImageNet(Dataset):
    def __init__(self, lmdb_path, transform=None):
        assert osp.exists(lmdb_path)
        self.env = lmdb.open(lmdb_path, 
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
        
        with self.env.begin() as txn:
            self.length = int(txn.get('num_samples'.encode()).decode())
        self.samples = np.arange(self.length)
        np.random.shuffle(self.samples)
        self.transform = transform
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        with self.env.begin() as txn:
            data = txn.get(f'{self.samples[index]}'.encode())
            if data is None:
                raise IndexError(f'Index {index} is out of bounds')
            
            data = pickle.loads(data)
            img_bin = data['image']
            label = data['label']
            
            buffer = io.BytesIO(img_bin)
            img = Image.open(buffer).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
                
            return img, label
    
    def __del__(self):
        self.env.close()


class ImageNetLmdbDataset(LMDBImageNet):

    def __init__(self, anno_file, image_size, is_train = False):

        super().__init__(anno_file,)
        if is_train:
            crop_size = image_size
            self.transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, crop_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        else:
            crop_size = image_size 
            self.transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
    
    def __getitem__(self, idx):

        image, target = super().__getitem__(idx)
        return image, target


if __name__ == '__main__':

    fpath = 'imagenet/lmdb/val_lmdb/'
    dataloader = LMDBImageNet(fpath)
    for i, (img, label) in enumerate(dataloader):
        pdb.set_trace()