import kornia, cv2
import numpy as np
import torch.nn as nn
from typing import Union 
from einops import rearrange
import math, kornia, torch, pdb
from torch.nn import functional as F
from hita.engine.util import disabled_train
import hita.modules.pretrained_enc.models_pretrained_enc as models_pretrained_enc

class AbstractEmbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._is_trainable = None
        self._ucg_rate = None
        self._input_key = None

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    @property
    def ucg_rate(self) -> Union[float, torch.Tensor]:
        return self._ucg_rate

    @property
    def input_key(self) -> str:
        return self._input_key

    @is_trainable.setter
    def is_trainable(self, value: bool):
        self._is_trainable = value

    @ucg_rate.setter
    def ucg_rate(self, value: Union[float, torch.Tensor]):
        self._ucg_rate = value

    @input_key.setter
    def input_key(self, value: str):
        self._input_key = value

    @is_trainable.deleter
    def is_trainable(self):
        del self._is_trainable

    @ucg_rate.deleter
    def ucg_rate(self):
        del self._ucg_rate

    @input_key.deleter
    def input_key(self):
        del self._input_key

class PLACEHOLDEREmbedder(AbstractEmbModel):

    def __init__(self):
        super().__init__()
        self.input_key = ''
        self.is_trainable = False

class SelfSupervisedCondtionEmbedder(AbstractEmbModel):

    def __init__(self, pretrained_enc_arch, pretrained_enc_path,
                 pretrained_enc_withproj = False, proj_dim = 768, 
                 pretrained_enc_pca_path = None, antialias = True):

        super().__init__()
        self.input_key = ''
        self.is_trainable = False
        self.antialias = antialias
        if 'dinov2' in pretrained_enc_arch:
            self.pretrained_encoder = models_pretrained_enc.__dict__[pretrained_enc_arch]()
        elif 'moco' in pretrained_enc_arch:
            self.pretrained_encoder = models_pretrained_enc.__dict__[pretrained_enc_arch](
                proj_dim = proj_dim)
        else:
            raise NotImplementedError
        
        if 'dinov2' in pretrained_enc_arch:
            self.pretrained_encoder = models_pretrained_enc.load_pretrained_dino_v2(self.pretrained_encoder, pretrained_enc_path)
        elif 'moco' in pretrained_enc_arch:
            self.pretrained_encoder = models_pretrained_enc.load_pretrained_moco(self.pretrained_encoder, pretrained_enc_path)
        else:
            raise NotImplementedError

        if pretrained_enc_pca_path is not None:
            pca = np.load(pretrained_enc_pca_path, allow_pickle=True).item()
            self.pca_component = torch.Tensor(pca["components"]).cuda()
            self.pca_mean = torch.Tensor(pca["mean"]).cuda()
            self.pretrained_enc_use_pca = True
        else:
            self.pretrained_enc_use_pca = False

        self.pretrained_encoder.cuda()
        self.pretrained_encoder.eval()
        self.pretrained_encoder.train = disabled_train
        try:
            self.pretrained_enc_withproj = pretrained_enc_withproj
        except:
            self.pretrained_enc_withproj = False
    
    def preprocess(self, x):
        
        #* Normalize to [0,1]
        x = kornia.geometry.resize(
            x,
            (224, 224),
            interpolation="bicubic",
            align_corners=True,
            antialias=self.antialias,
        )
        x = x.clamp(min=-1, max=1)
        x = (x + 1.0) / 2.0
        return x

    def forward(self, x):

        x = self.preprocess(x)
        mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        std = torch.Tensor([0.229, 0.224, 0.225]).cuda().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        x_normalized = (x - mean) / std
        output = self.pretrained_encoder.forward_features(x_normalized)

        rep = torch.cat((output['x_norm_clstoken'].unsqueeze(1), output['x_norm_patchtokens']), dim=1)
        if self.pretrained_enc_withproj:
            rep = self.pretrained_encoder.head(rep)
        
        return rep