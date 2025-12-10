import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from vit_model import *
import os

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(root="../data", train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root="../data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


class DiT:
    def __init__(self,timesteps,beta_start, beta_end):
        super().__init__()
        self.beta = torch.linspace(beta_start,beta_end,timesteps, device=device)
        self.alpha = torch.ones_like(self.beta) - self.beta
        self.alpha_bar = torch.cumprod(self.alpha,dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(torch.ones_like(self.alpha_bar)-self.alpha_bar)

    def add_noise(self,x_original,noise,t):
        sqrt_alpha_bar = self.sqrt_alpha_bar[t].view(-1,1,1,1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t].view(-1,1,1,1)
        return sqrt_alpha_bar*x_original + sqrt_one_minus_alpha_bar*noise

    def sample_prev_timestep(self, x_t, noise_pred, t):
        x0_hat = (x_t - self.sqrt_one_minus_alpha_bar[t]*noise_pred) / self.sqrt_alpha_bar[t]
        mean = x_t - (self.beta[t] / self.sqrt_one_minus_alpha_bar[t]) * noise_pred
        mean = mean / self.sqrt_alpha_bar[t]
        if t == 0:
            return mean, x0_hat
        variance = ((1 - self.alpha_bar[t-1])/(1 - self.alpha_bar[t])) * self.beta[t]
        z = torch.randn_like(x_t)
        return mean + torch.sqrt(variance)*z, x0_hat


def timeEmbedding(timesteps,emd_dim):
    factor = 10000 ** ((torch.arange(start=0, end=emd_dim // 2, dtype=torch.float32, device=device)
                        / (emd_dim // 2)))
    t_emb = timesteps[:, None].repeat(1, emd_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


class DiTViT(nn.Module):
    def __init__(self,img_size=28,patch=7,in_ch=1,embed_dim=256,num_block=6,heads=8,ff_dim=4,elementwise_affine=False):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch
        self.embed_dim = embed_dim

        self.vit = ViTModule(
            img_size=img_size,
            patch=patch,
            in_ch=in_ch,
            num_classes=img_size * img_size,
            embed_dim=embed_dim,
            num_block=num_block,
            heads=heads,
            ff_dim=ff_dim,
            use_cls=False,
            elementwise_affine=elementwise_affine
        )
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch,
            in_ch=in_ch,
            embed_dim=embed_dim,
            use_cls=False
        )
        self.timesteps = 100
        self.noisescheduler = DiT(
            timesteps=self.timesteps,
            beta_start=0.0001,
            beta_end=0.02
        )
        self.time_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self,X_t,sampled_timesteps,noise,add_noise=True):
        if add_noise:
            X_t = self.noisescheduler.add_noise(X_t,noise,sampled_timesteps)
        time_emb = timeEmbedding(sampled_timesteps, self.embed_dim)
        time_emb = self.time_proj(time_emb)
        X_t = self.patch_embed(X_t) + time_emb[:, None, :]
        vit = self.vit(X_t)
        B = X_t.shape[0]
        P = self.patch_size
        H = self.img_size
        W = self.img_size
        C = 1
        vit = vit.view(B, H//P, W//P, C, P, P)
        vit = vit.permute(0,3,1,4,2,5)
        vit = vit.reshape(B, C, H, W)
        return vit


timesteps = 100
model = DiTViT().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)


def train_one_epoch(model, loader, optimizer, epoch):
    model.train()
    total_loss = 0
    total = 0
    for x, y in tqdm(loader):
        x = x.to(device)
        batch_size = x.shape[0]
        sampled_timesteps = torch.randint(timesteps, size=(batch_size,), device=device)
        noise = torch.randn_like(x)
        optimizer.zero_grad()
        noise_hat = model(x,sampled_timesteps,noise)
        loss = criterion(noise,noise_hat)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total += y.size(0)
    avg_loss = total_loss / total
    print("epoch", epoch, "train loss", avg_loss)

epochs = 10
ckpt = f"dit_{epochs}.pt"
if os.path.exists(ckpt):
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print("loaded saved model")
else:
    for epoch in range(epochs):      
        train_one_epoch(model, train_loader, optimizer, epoch)
    torch.save(model.state_dict(), ckpt)
    print("saved model")


model = DiTViT().to(device)
model.load_state_dict(torch.load(f"dit_{epochs}.pt", map_location=device))
model.eval()


@torch.no_grad()
def test(model, num_samples,train_loader):
    x, _ = next(iter(train_loader))
    x = x.to(device)
    noise_scheduler = DiT(timesteps=timesteps, beta_start=0.0001, beta_end=0.02)
    samples = []
    for _ in range(num_samples):
        X = torch.randn_like(x)
        for t in range(timesteps-1,-1,-1):
            t_batch = t*torch.ones(X.size(0), device=device, dtype=torch.long)
            noise_hat = model(X,t_batch,None,add_noise=False)
            X, _ = noise_scheduler.sample_prev_timestep(X,noise_hat,t)
        samples.append(X.cpu())
    samples = torch.cat(samples, dim=0)
    return samples


samples = test(model, num_samples=4, train_loader=train_loader)
img = samples[0,0].clamp(0,1).numpy()
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()
