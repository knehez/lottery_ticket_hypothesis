import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy

# --- MLP-Mixer modulok ---
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

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
    def __init__(self, image_size=28, patch_size=7, dim=64, depth=4, num_classes=10, token_dim=32, channel_dim=128):
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

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p).contiguous().view(B, -1, p*p)
        x = self.embedding(x)
        x = self.mixer_blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# --- Adatok ---
transform = transforms.ToTensor()
train_loader = DataLoader(datasets.MNIST('.', train=True, download=True, transform=transform),
                          batch_size=128, shuffle=True)
test_loader = DataLoader(datasets.MNIST('.', train=False, transform=transform),
                         batch_size=128)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LTH komponensek ---
def train(model, loader, optimizer):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()

def test(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
    return correct / len(loader.dataset)

def get_initial_weights(model):
    return {name: param.clone() for name, param in model.named_parameters()}

def initialize_mask(model):
    return {name: torch.ones_like(param) for name, param in model.named_parameters() if 'weight' in name}

def apply_mask(model, mask):
    for name, param in model.named_parameters():
        if name in mask:
            param.data *= mask[name]

def prune_by_percentile(model, mask, percent):
    all_weights = torch.cat([
        param[mask[name] != 0].abs().flatten()
        for name, param in model.named_parameters()
        if name in mask
    ])
    threshold = torch.quantile(all_weights, percent)
    for name, param in model.named_parameters():
        if name in mask:
            mask[name] = (param.abs() > threshold).float()

def report_sparsity(mask):
    total = sum(m.numel() for m in mask.values())
    zeroed = sum((m == 0).sum().item() for m in mask.values())
    print(f"Sparsity: {zeroed}/{total} weights pruned ({100 * zeroed / total:.2f}%)")

# --- Lottery Ticket Pruning ciklus ---
num_iterations = 10
epochs_per_iteration = 3
prune_percent = 0.2
lr = 0.001

model = MLPMixer().to(device)
initial_weights = get_initial_weights(model)
mask = initialize_mask(model)

for iteration in range(num_iterations):
    print(f"\n--- Iteration {iteration+1} ---")

    # Újrainicializálás
    model = MLPMixer().to(device)
    for name, param in model.named_parameters():
        param.data = initial_weights[name].clone()

    apply_mask(model, mask)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Tréning
    for epoch in range(epochs_per_iteration):
        train(model, train_loader, optimizer)

    acc = test(model, test_loader)
    print(f"Test accuracy: {acc:.4f}")
    report_sparsity(mask)

    # Pruning
    prune_by_percentile(model, mask, prune_percent)
    apply_mask(model, mask)
