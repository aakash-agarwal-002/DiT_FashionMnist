#dit.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from vit_model import *
import os
import random
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
batch_size = 128
transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(root="../data", train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root="../data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

classes = train_dataset.classes

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
        mean = mean / torch.sqrt(self.alpha[t])
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
    def __init__(self,timesteps,num_classes,img_size=28,patch=7,in_ch=1,embed_dim=256,num_block=6,heads=8,ff_dim=4,elementwise_affine=False):
        super().__init__()
        self.img_size = img_size
        self.in_ch = in_ch
        self.patch_size = patch
        self.embed_dim = embed_dim

        self.vit = ViTModule(img_size=img_size,patch=patch,in_ch=in_ch,num_classes=img_size * img_size,embed_dim=embed_dim,num_block=num_block,heads=heads,ff_dim=ff_dim,use_cls=False,elementwise_affine=elementwise_affine)
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch,
            in_ch=in_ch,
            embed_dim=embed_dim,
            use_cls=False
        )
        self.timesteps = timesteps
        self.noisescheduler = DiT(
            timesteps=self.timesteps,
            beta_start=0.0001,
            beta_end=0.02
        )
        self.time_proj = nn.Linear(embed_dim, embed_dim)
        self.label_proj = nn.Embedding(num_classes, embed_dim)

    def forward(self,X_t,y,sampled_timesteps,noise,add_noise=True):
        if add_noise:
            X_t = self.noisescheduler.add_noise(X_t,noise,sampled_timesteps)
        time_emb = timeEmbedding(sampled_timesteps, self.embed_dim)
        time_emb = self.time_proj(time_emb)
        label_emb = self.label_proj(y)
        conditional_token = time_emb + label_emb

        tokens = self.patch_embed(X_t)
        vit = self.vit(tokens,conditional_token)
        B = X_t.shape[0]
        P = self.patch_size
        H = self.img_size
        W = self.img_size
        C = self.in_ch
        vit = vit.view(B, H//P, W//P, C, P, P)
        vit = vit.permute(0,3,1,4,2,5)
        vit = vit.reshape(B, C, H, W)
        return vit


timesteps = 1000
in_ch = train_dataset[0][0].shape[0]
H = train_dataset[0][0].shape[1]
W = train_dataset[0][0].shape[2]
emb_dim = 256
num_block=6
heads=8
ff_dim=4
epochs = 25
patch_size = 4

assert H%patch_size == 0

model = DiTViT(timesteps,num_classes=len(classes),img_size=H,patch=patch_size,in_ch=in_ch,embed_dim=emb_dim,num_block=num_block,heads=heads,ff_dim=ff_dim,elementwise_affine=False).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
start_epoch = 0


def train_one_epoch(model, loader, optimizer, epoch):
    model.train()
    total_loss = 0
    total = 0
    for x, y in tqdm(loader):
        x = x.to(device)
        y = y.to(device)
        batch_size = x.shape[0]
        sampled_timesteps = torch.randint(timesteps, size=(batch_size,), device=device)
        noise = torch.randn_like(x)
        optimizer.zero_grad()
        noise_hat = model(x,y,sampled_timesteps,noise)
        loss = criterion(noise,noise_hat)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total += y.size(0)
    avg_loss = total_loss / total
    print("epoch", epoch, "train loss", avg_loss)
    return avg_loss


@torch.no_grad()
def test(model, num_samples,test_loader,classes,timesteps,device=device):
    x, y = next(iter(test_loader))
    x = x.to(device)
    y = y.to(device)
    noise_scheduler = DiT(timesteps=timesteps, beta_start=0.0001, beta_end=0.02)
    samples = []
    labels = []
    for class_id in range(len(classes)):
        X = torch.randn_like(x)
        y = torch.tensor([class_id], device=device, dtype=torch.long)
        for t in range(timesteps-1, -1, -1):
            t_batch = t*torch.ones(X.size(0), device=device, dtype=torch.long)
            noise_hat = model(X, y, t_batch, None, add_noise=False)
            X, _ = noise_scheduler.sample_prev_timestep(X, noise_hat, t)
        samples.append(X.cpu())
        labels.append(classes[class_id])
    samples = torch.cat(samples, dim=0)
    return samples, labels

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results/test", exist_ok=True)
os.makedirs("results/loss", exist_ok=True)



latest = None
for f in os.listdir("checkpoints"):
    if f.startswith("dit_epoch_") and f.endswith(".pt"):
        try:
            ep = int(f[len("dit_epoch_"):-3])
            if latest is None or ep > latest:
                latest = ep
        except:
            pass

if latest is not None:
    ck = f"checkpoints/dit_epoch_{latest}.pt"
    model.load_state_dict(torch.load(ck, map_location=device))
    start_epoch = latest


loss_history = []


for epoch in range(start_epoch, epochs):
    avg_loss = train_one_epoch(model, train_loader, optimizer, epoch)
    loss_history.append(avg_loss)
    epochs_axis = np.arange(1, len(loss_history) + 1)
    plt.figure()
    plt.plot(epochs_axis,loss_history )
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("loss")
    plt.savefig(f"results/loss/loss_epoch_{epoch+1}.png")
    plt.close()

    if (epoch + 1) % 5 == 0:
        path = f"checkpoints/dit_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), path)
        print("saved", path)

        model.eval()
        with torch.no_grad():
            for T in [100,500,1000]:
                fig = plt.figure(figsize=(len(classes),1))
                s,labels = test(model, num_samples=len(classes),
                                test_loader=test_loader,
                                classes=classes,
                                timesteps=T,
                                device=device)
                for i in range(len(classes)):
                    ax = fig.add_subplot(1, len(classes), i+1)
                    ax.imshow(s[i,0].clamp(0,1).cpu().numpy(), cmap="gray")
                    ax.set_title(f"{labels[i]}")
                    ax.axis("off")
                fig.tight_layout()
                fig.savefig(f"results/test/e_{epoch+1}_T{T}.png")
                plt.close(fig)
        model.train()

    with open("results/loss.txt", "a") as f:
        f.write(f"epoch {epoch+1}: {avg_loss}\n")


# metrics
from torchmetrics.image.fid import FrechetInceptionDistance
@torch.no_grad()
def compute_fid(model, loader, num_samples=1000, device=device, timesteps=1000):
    print(f"--- Starting FID Calculation ({num_samples} samples, T={timesteps}) ---")
    model.eval()
    
    fid = FrechetInceptionDistance(feature=64).to('cpu') 
    
    print("Processing Real Images...")
    real_images_count = 0
    for x, _ in tqdm(loader, desc="Real Images"):

        x = x.to('cpu')
        

        x_rgb = x.repeat(1, 3, 1, 1)
        

        x_uint8 = (x_rgb * 255).to(torch.uint8)
        
        fid.update(x_uint8, real=True)
        real_images_count += x.shape[0]
        if real_images_count >= num_samples:
            break

    print("Generating Fake Images...")
    
    batch_size = 32 
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    noise_scheduler = DiT(timesteps=timesteps, beta_start=0.0001, beta_end=0.02)
    
    for _ in tqdm(range(num_batches), desc="Fake Images"):
        curr_batch = min(batch_size, num_samples - (_ * batch_size))
        
        x = torch.randn(curr_batch, 1, 28, 28).to(device)
        y = torch.randint(0, len(classes), (curr_batch,), device=device)

        for t in range(timesteps-1, -1, -1):
            t_batch = t * torch.ones(x.size(0), device=device, dtype=torch.long)
            noise_hat = model(x, y, t_batch, None, add_noise=False)
            x, _ = noise_scheduler.sample_prev_timestep(x, noise_hat, t)
        
        x = x.detach().cpu().clamp(0, 1)
        x_rgb = x.repeat(1, 3, 1, 1)
        x_uint8 = (x_rgb * 255).to(torch.uint8)
        fid.update(x_uint8, real=False)
    print("Computing Final Score...")
    fid_score = fid.compute()
    print(f"FID Score: {fid_score.item():.4f}")
    return fid_score


if latest is not None:
    print(f"Loading best checkpoint for evaluation: checkpoints/dit_epoch_{latest}.pt")
    model.load_state_dict(torch.load(f"checkpoints/dit_epoch_{latest}.pt", map_location=device))


fid_results = {}
for T in [100,500,1000]:
    fid_T = compute_fid(model, test_loader, num_samples=1000, device=device, timesteps=T)
    fid_results[T] = fid_T.item()

plt.figure()
plt.plot(sorted(fid_results.keys()),
         [fid_results[T] for T in sorted(fid_results.keys())])
plt.xlabel("timesteps")
plt.ylabel("FID")
plt.title("FID vs timesteps")
plt.savefig("results/fid_vs_timesteps.png")
plt.close()

with open("results/fid_scores.txt", "w") as f:
    for T in sorted(fid_results.keys()):
        f.write(f"T {T}: {fid_results[T]}\n")
