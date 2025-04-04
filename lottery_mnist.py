import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time  # Add time import

# Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs_per_iteration = 5
num_iterations = 15
prune_percent = 0.2
batch_size = 128
lr = 0.01

# CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)   # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 14x14 -> 14x14
        self.pool = nn.MaxPool2d(2, 2)                # Halves the size
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))          # 32x14x14
        x = self.pool(F.relu(self.conv2(x)))          # 64x7x7
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Data
transform = transforms.ToTensor()
train_loader = DataLoader(datasets.MNIST('.', train=True, download=True, transform=transform),
                          batch_size=batch_size, shuffle=True)
test_loader = DataLoader(datasets.MNIST('.', train=False, transform=transform),
                         batch_size=batch_size)

# Training
def train(model, optimizer, loader):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()

# Testing
def test(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
    return correct / len(loader.dataset)

# Save initialization
def get_initial_weights(model):
    return {name: param.clone() for name, param in model.named_parameters()}

# Initialize mask
def initialize_mask(model):
    return {name: torch.ones_like(param) for name, param in model.named_parameters() if 'weight' in name}

# Apply mask
def apply_mask(model, mask):
    for name, param in model.named_parameters():
        if name in mask:
            param.data *= mask[name]

# Pruning
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

# Sparsity report
def report_sparsity(mask):
    total = 0
    zeros = 0
    for m in mask.values():
        total += m.numel()
        zeros += (m == 0).sum().item()
    print(f"Weights zeroed: {zeros}/{total} ({zeros/total:.2%})")

# Dead neuron report
def report_dead_neurons_input_or_output(model, mask):
    print("\nDead neurons (input OR output zeroed):")

    # First, collect relevant layers (Linear and Conv2d)
    layers = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            layers.append((name, layer))

    for i, (name, layer) in enumerate(layers):
        weight_name = f"{name}.weight"
        current_mask = mask.get(weight_name)
        if current_mask is None:
            continue

        # --- INPUT side ---
        if isinstance(layer, nn.Linear):
            input_dead = (current_mask.sum(dim=1) == 0)  # [out_features]
        elif isinstance(layer, nn.Conv2d):
            input_dead = (current_mask.view(current_mask.shape[0], -1).sum(dim=1) == 0)  # [out_channels]

        # --- OUTPUT side (based on the next layer) ---
        if i + 1 < len(layers):
            next_name, next_layer = layers[i + 1]
            next_mask = mask.get(f"{next_name}.weight")
            if next_mask is not None:
                if isinstance(next_layer, nn.Linear):
                    output_dead = (next_mask.sum(dim=0) == 0)  # [in_features]
                elif isinstance(next_layer, nn.Conv2d):
                    output_dead = (next_mask.view(next_mask.shape[0], next_mask.shape[1], -1).sum(dim=(0, 2)) == 0)  # [in_channels]
                else:
                    output_dead = torch.zeros_like(input_dead, dtype=torch.bool)
            else:
                output_dead = torch.zeros_like(input_dead, dtype=torch.bool)
        else:
            output_dead = torch.zeros_like(input_dead, dtype=torch.bool)

        # --- OR logic ---
        # Important: input_dead and output_dead sizes may not match (e.g., after FC flattening)
        # In such cases, we can only compare based on input
        if input_dead.shape != output_dead.shape:
            dead_or = input_dead
        else:
            dead_or = input_dead | output_dead

        dead_count = dead_or.sum().item()
        total = dead_or.numel()

        print(f"{weight_name}: {dead_count}/{total} neurons dead (in OR out)")


# ===== MAIN PRUNING LOOP =====
model = SimpleCNN().to(device)
initial_weights = get_initial_weights(model)
mask = initialize_mask(model)

for iteration in range(num_iterations):
    print(f"\n--- Iteration {iteration + 1} ---")

    # Reinitialize model
    model = SimpleCNN().to(device)
    for name, param in model.named_parameters():
        param.data = initial_weights[name].clone()

    apply_mask(model, mask)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Training
    for epoch in range(epochs_per_iteration):
        train(model, optimizer, train_loader)

    start_time = time.time()  # Start timing
    acc = test(model, test_loader)
    end_time = time.time()  # End timing
    print(f"Test duration: {end_time - start_time:.2f} seconds")    
    print(f"Accuracy: {acc:.4f}")

    # Sparsity and dead neuron report
    report_sparsity(mask)
    report_dead_neurons_input_or_output(model, mask)

    # Pruning
    prune_by_percentile(model, mask, prune_percent)
    apply_mask(model, mask)
