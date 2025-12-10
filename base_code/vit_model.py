import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="data",
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    transform=transform,
    download=True
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=256, use_cls=True):
        super().__init__()
        self.patch_size = patch_size
        self.use_cls = use_cls
        self.n = (img_size//self.patch_size)** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
            self.pos_emb = nn.Parameter(torch.zeros(1,1+self.n,embed_dim))
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1,self.n,embed_dim))


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

    def forward(self,X):
        X = X + self.multihead_attention(self.layer_norm1(X),self.layer_norm1(X),self.layer_norm1(X))[0]
        X = X + self.seq(self.layer_norm2(X))
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

        def forward(self,X):
            if self.do_patch:
                X = self.patch_embed(X)
            batch_size = X.shape[0]
            for block in self.enc_layers:
                 X = block(X)
            X = self.norm(X)
            X = self.out(X)
            return X
        

x, y = next(iter(train_loader))
H = x[0].squeeze(dim=0).shape[0]
W = x[0].squeeze(dim=0).shape[1]


def main():
    H = 32
    W = 32

    model = ViTModule(
        img_size = H,
        patch = 7,
        in_ch = 1,
        num_classes = 10,
        embed_dim = 256,
        num_block = 6,
        heads = 8,
        use_cls = True,
        elementwise_affine = True,
        ff_dim = 4
    )

    print(model)

if __name__ == "__main__":
    main()
