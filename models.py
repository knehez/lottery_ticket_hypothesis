import torch
import torch.nn as nn
import torch.nn.functional as F
    
# --- MLP-Mixer model ---
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class MixerBlock(nn.Module):
    def __init__(self, num_patches, dim, token_dim, channel_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mlp = MLP(num_patches, token_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mlp = MLP(dim, channel_dim)

    def forward(self, x):
        y = self.norm1(x).transpose(1, 2)
        x = x + self.token_mlp(y).transpose(1, 2)
        x = x + self.channel_mlp(self.norm2(x))
        return x

class MLPMixer(nn.Module):
    def __init__(self, image_size=28, patch_size=4, dim=64, depth=8, num_classes=10, token_dim=32, channel_dim=128):
        super().__init__()
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2
        patch_dim = patch_size * patch_size

        self.embedding = nn.Linear(patch_dim, dim)
        self.mixer_blocks = nn.Sequential(*[
            MixerBlock(num_patches, dim, token_dim, channel_dim)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.dim = dim

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p).contiguous().view(B, -1, p*p)
        x = self.embedding(x)
        x = self.mixer_blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.classifier(x)
