#vit.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=256, use_cls=True):
        super().__init__()
        self.patch_size = patch_size
        self.use_cls = use_cls
        self.n = (img_size//self.patch_size)** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_emb = nn.Parameter(torch.randn(1, 1 + self.n, embed_dim))
        else:
            self.pos_emb = nn.Parameter(torch.randn(1, self.n, embed_dim))


    def forward(self,X):
        X = self.proj(X)
        X = X.flatten(2).transpose(1,2)
        if self.use_cls:
            cls = self.cls_token.expand(X.size(0),-1,-1)
            X = torch.cat((cls,X),dim=1)
        X = X + self.pos_emb
        return X
    


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, elementwise_affine=False,ff_dim=4):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim,embed_dim*ff_dim)
        self.linear2 = nn.Linear(embed_dim*ff_dim,embed_dim)
        self.gelu = nn.GELU()
        self.multihead_attention = nn.MultiheadAttention(embed_dim,num_heads=num_heads,batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim,elementwise_affine=elementwise_affine)
        self.layer_norm2 = nn.LayerNorm(embed_dim,elementwise_affine=elementwise_affine)
        self.seq = nn.Sequential(
            self.linear1,
            self.gelu,
            self.linear2
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 6 * embed_dim)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def adaptive(self,x, shift, scale):
        return x * (1 + scale[:,None,:]) + shift[:,None,:]

    def forward(self,X,c):
        X_residual = X
        shift_gamma1, scale_beta1, scale_alpha1, shift_gamma2, scale_beta2, scale_alpha2 = self.adaLN_modulation(c).chunk(6, dim=1)
        X = self.adaptive(self.layer_norm1(X), shift_gamma1, scale_beta1)
        attn_output, _ = self.multihead_attention(X, X, X)
        attn_output = attn_output*scale_alpha1[:,None,:]
        X = X_residual + attn_output
        X_residual = X
        X = self.adaptive(self.layer_norm2(X), shift_gamma2, scale_beta2)
        seq_output = self.seq(X)
        seq_output = seq_output*scale_alpha2[:,None,:]
        X = X_residual + seq_output
        return X


class ViTModule(nn.Module):
        def __init__(self,img_size=224,patch=16,in_ch=3,num_classes=10,embed_dim=256,num_block=6,heads=8,ff_dim=4,use_cls=True,elementwise_affine=False, do_patch = False):
            super().__init__()
            self.use_cls = use_cls
            self.do_patch = do_patch
            self.img_size = img_size
            self.patch_embed = PatchEmbedding(img_size=img_size,patch_size=patch,in_ch=in_ch,embed_dim=embed_dim,use_cls=self.use_cls)
            self.enc_layers = nn.ModuleList(
                  TransformerEncoder(embed_dim=embed_dim,num_heads=heads,elementwise_affine=elementwise_affine,ff_dim=ff_dim) for _ in range(num_block)
            )
            self.norm = nn.LayerNorm(embed_dim,elementwise_affine=elementwise_affine)
            self.out = nn.Linear(embed_dim, in_ch*patch*patch)

            self.final_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim))   
            nn.init.constant_(self.final_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_adaLN_modulation[-1].bias, 0)
        
        def adaptive(self,x, shift, scale):
            return x * (1 + scale[:,None,:]) + shift[:,None,:]

        def forward(self,X,c):
            if self.do_patch:
                X = self.patch_embed(X)
            batch_size = X.shape[0]
            for block in self.enc_layers:
                 X = block(X,c)

            shift, scale = self.final_adaLN_modulation(c).chunk(2, dim=1)
            X = self.adaptive(self.norm(X),shift,scale)
            X = self.out(X)
            return X
    