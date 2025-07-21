from torch import nn
from copy import deepcopy
import torch, math, pdb, time
from ..modules.mlp import build_mlp
from torch.nn import functional as F
from einops import rearrange, repeat
from .transformer import Transformer
from .transformer import build_perceptron
from einops.layers.torch import Rearrange
from ..engine.util import instantiate_from_config


class Encoder(nn.Module):

    def __init__(self, image_size, layer_type, n_carrier,
                 patch_size, dim, depth, num_head, mlp_dim,
                 in_channels=1024, d_model=256, dim_head=64, 
                 visual_encoder_config = None, dropout=0.):

        super().__init__()

        self.backbone = instantiate_from_config(visual_encoder_config)
        self.proj = build_mlp(in_channels, dim, dim, 2)

        self.image_size = image_size
        self.patch_size = patch_size
        
        self.n_carrier = n_carrier
        scale = dim ** -0.5
        self.carrier_tokens = nn.Parameter(torch.randn(1, self.n_carrier, dim) * scale)

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim) * scale)
        self.norm_pre1 = nn.LayerNorm(dim)
        self.transformer1 = Transformer(layer_type, dim, depth, num_head, dim_head, mlp_dim, dropout, xformer=True)
        
        self.transformer2 = Transformer(layer_type, dim, depth, num_head, dim_head, mlp_dim, dropout, xformer=False)
        self.norm_pre2 = nn.LayerNorm(dim)

        self.initialize_weights()

    def freeze_visual_encoder(self):

        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
            
    def freeze(self):

        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, imgs, h):

        self.backbone.eval()
        with torch.no_grad():
            latent = self.backbone(imgs)

        dinov2 = self.proj(latent)

        bs, _, H, W = h.shape
        x = self.prepare_tokens(h)

        carrier = self.carrier_tokens.repeat(bs, 1, 1)

        n_c, m, n = self.n_carrier, dinov2.size(1), x.size(1)
        x = torch.cat((carrier, dinov2, x), dim=1)

        x = self.norm_pre1(x)
        x = self.transformer1(x,)
        
        sizes = [n_c, m, n]
        carrier, _,  z = x.split(sizes, dim=1)

        x = torch.cat((carrier, z), dim=1)
        attn_mask = ~torch.tril(x.new_ones((n_c + n, n_c + n), dtype=torch.bool), diagonal=0)

        x = self.norm_pre2(x)
        x = self.transformer2(x, mask=attn_mask)

        sizes = [n_c, n]
        carrier, z = x.split(sizes, dim=1)

        z = rearrange(z, 'b (h w) c -> b c h w', h=H, w=W)
        return z, carrier, latent[:, 1:]

    def interpolate_pos_encoding(self, x, w, h):


        npatch = x.shape[1]
        N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed
        
        patch_pos_embed = self.pos_embed
        dim = x.shape[-1]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w + 0.1, h + 0.1
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        return patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

    def prepare_tokens(self, x):
        
        _, _, w, h = x.shape
        # add positional encoding to each token
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.interpolate_pos_encoding(x, w, h)
        
        return x

    def initialize_weights(self):

        if self.backbone:
            assert self.backbone.is_trainable is False
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.norm_pre1.apply(self._init_weights)
        self.norm_pre2.apply(self._init_weights)
        for m in self.transformer1.parameters():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        for m in self.transformer2.parameters():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class Decoder(nn.Module):
    def __init__(self, layer_type, image_size, patch_size, dim, n_carrier, shift_k,
                 depth, num_head, mlp_dim, dim_head=64, dropout=0., enable_vfm_recon=False):
        
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.shift_k = shift_k
        self.enable_vfm_recon = enable_vfm_recon
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.dim = dim
        scale = dim ** -0.5
        num_patches = (image_size // patch_size) ** 2

        self.position_embedding = nn.Parameter(torch.randn(1, num_patches, dim) * scale)
        self.slot_position_embedding = nn.Parameter(torch.randn(1, n_carrier, dim) * scale)

        self.transformer = Transformer(layer_type, dim, depth, num_head, dim_head, mlp_dim, dropout, xformer=False)
        self.norm = nn.LayerNorm(dim)

        if self.enable_vfm_recon:
            self.mask_embedding = nn.Parameter(torch.randn(1, 1, dim) * scale)
            self.norm_vfm = nn.LayerNorm(dim)
            self.proj  = nn.Linear(dim, 1024)
        
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def interpolate_pos_encoding(self, x, w, h):

        npatch = w * h
        N = self.position_embedding.size(1)
        if npatch == N and w == h:
            return self.position_embedding

        patch_pos_embed = self.position_embedding
        dim = x.shape[2]
        
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w + 0.1, h + 0.1

        m = patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        return patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

    def prepare_tokens(self, x):

        w = h = int(math.sqrt(x.size(1)))
        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        
        return x

    def recon_vfm_state(self, slots,):
        
        H = W = 16
        bs, n_c, _ = slots.shape
        patch_pos_embed = self.interpolate_pos_encoding(slots, W, H)
        latent = repeat(self.mask_embedding + patch_pos_embed, 'f ... -> (b f) ...', b=bs)
        num = patch_pos_embed.size(1)

        tl = ~torch.tril(latent.new_ones((n_c, n_c), dtype=torch.bool), diagonal=0)
        tr = latent.new_ones((n_c, num), dtype=torch.bool)

        top = torch.cat((tl, tr), dim = 1)
        bottom = latent.new_zeros((num, n_c + num), dtype=torch.bool)
        attn_mask = torch.cat((top, bottom), dim=0)

        x = torch.cat((slots, latent), dim = 1)
        x = self.norm_vfm(x)
        
        x = self.transformer(x, mask=attn_mask)
        z = self.proj(x[:, n_c:])

        return z

    def forward(self, z, slots):
        
        n_c = slots.size(1)
        H, W = z.shape[2:]

        memory = slots + self.slot_position_embedding
        patch_pos_embed = self.interpolate_pos_encoding(slots, W, H)
        h = rearrange(z, 'b c h w -> b (h w) c') + patch_pos_embed
        
        x = torch.cat((memory, h), dim=1)
        mask = ~torch.tril(x.new_ones((n_c + H * W, n_c + H * W), dtype=torch.bool), diagonal=0)
        
        x = self.norm(x)
        x = self.transformer(x, mask=mask)

        z = rearrange(x[:, n_c - self.shift_k: n_c - self.shift_k + H * W], 'b (h w) c -> b c h w', h=H, w=W)

        if self.enable_vfm_recon:
            sem = self.recon_vfm_state(slots,)
            return z, sem
        return z, None

