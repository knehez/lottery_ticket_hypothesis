import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os

# Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
epochs_per_iteration = 5
num_iterations = 10
prune_percent = 0.2
batch_size = 128
lr = 0.01

# --- Modell ---
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- Modified model: parallel autoencoder-like MLP for fc2 and fc3 ---
class SimpleMLPWithParallel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)
        # Parallel autoencoder-like branch for fc2: 300->30->100
        self.fc2_parallel_enc = nn.Linear(300, 30)
        self.fc2_parallel_dec = nn.Linear(30, 100)
        # Parallel autoencoder-like branch for fc3: 100->10->10
        self.fc3_parallel_enc = nn.Linear(100, 10)
        self.fc3_parallel_dec = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x1 = F.relu(self.fc1(x))
        # fc2 + parallel autoencoder branch
        fc2_out = self.fc2(x1)
        fc2_par_out = self.fc2_parallel_dec(F.relu(self.fc2_parallel_enc(x1)))
        x2 = F.relu(fc2_out + fc2_par_out)
        # fc3 + parallel autoencoder branch
        fc3_out = self.fc3(x2)
        fc3_par_out = self.fc3_parallel_dec(F.relu(self.fc3_parallel_enc(x2)))
        out = fc3_out + fc3_par_out
        return out

# --- Kvantált modell visszatöltés, ha már létezik ---
model = SimpleMLP().to(device)
if os.path.exists("final_parallel_model.pt"):
    model = SimpleMLPWithParallel().to(device)
    model.load_state_dict(torch.load("final_parallel_model.pt"))
    print("✅ Kvantált modell betöltve.")
    
    # Export ONNX using the correct model and device
    dummy_input = torch.randn(1, 1, 28, 28, device=device)
    parallel_model = model.to(device)
    parallel_model.eval()
    torch.onnx.export(parallel_model, dummy_input, "simple_parallel.onnx")
    
    # Adat
    transform = transforms.ToTensor()
    test_loader = DataLoader(datasets.MNIST('.', train=False, transform=transform), batch_size=128)
    acc = 0.0
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            acc += (pred == y).sum().item()
    acc /= len(test_loader.dataset)
    print(f"🎯 Kvantált modell pontossága: {acc:.4f}")
    exit()

# --- Adat ---
transform = transforms.ToTensor()
train_loader = DataLoader(datasets.MNIST('.', train=True, download=True, transform=transform),
                          batch_size=batch_size, shuffle=True)
test_loader = DataLoader(datasets.MNIST('.', train=False, transform=transform),
                         batch_size=batch_size)

# --- Függvények ---
def train(model, optimizer, loader):
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
    # Exclude fc2_parallel and fc3_parallel from mask
    return {
        name: torch.ones_like(param)
        for name, param in model.named_parameters()
        if 'weight' in name and not name.startswith('fc1') and
           not name.startswith('fc2_parallel') and not name.startswith('fc3_parallel')
    }

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
    total = 0
    zeros = 0
    for m in mask.values():
        total += m.numel()
        zeros += (m == 0).sum().item()
    print(f"🧹 Sparsity: {zeros}/{total} = {zeros/total:.2%}")

def report_weight_counts_per_layer(model):
    print("\nWeight counts per layer:")
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name}: {param.numel()}")

# --- Ternáris kvantálás ---
def ternary_quantize(param_tensor, threshold=0.05):
    q_param = torch.zeros_like(param_tensor)
    q_param[param_tensor > threshold] = 1.0
    q_param[param_tensor < -threshold] = -1.0
    return q_param

def quantize_model_ternary(model, mask, threshold=0.05):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and name in mask:
                param.data = ternary_quantize(param.data) * mask[name]

def quantize_fc2_fc3_ternary(model, mask, threshold=0.05):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask and (name.startswith('fc2.weight') or name.startswith('fc3.weight')):
                param.data = ternary_quantize(param.data) * mask[name]

# --- MAIN ---
model = SimpleMLP().to(device)
report_weight_counts_per_layer(model)
initial_weights = get_initial_weights(model)
mask = initialize_mask(model)

# Set fc2_parallel and fc3_parallel weights to zero and freeze them
def zero_and_freeze_parallel_layers(model):
    for name, param in model.named_parameters():
        if name.startswith('fc2_parallel') or name.startswith('fc3_parallel'):
            param.data.zero_()
            param.requires_grad = False

# --- 1. Training and pruning phase ---
for iteration in range(num_iterations):
    print(f"\n--- Iteration {iteration + 1} ---")

    model = SimpleMLP().to(device)
    for name, param in model.named_parameters():
        param.data = initial_weights[name].clone()

    apply_mask(model, mask)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs_per_iteration):
        train(model, optimizer, train_loader)

    start = time.time()
    acc = test(model, test_loader)
    print(f"⏱️ Teszt idő: {time.time() - start:.2f} s")
    print(f"🎯 Pontosság: {acc:.4f}")
    report_sparsity(mask)

    prune_by_percentile(model, mask, prune_percent)
    apply_mask(model, mask)

# --- 2. Ternary quantization and saving ---
print("\n🔧 Ternáris kvantálás (fc2, fc3) és mentés...")
quantize_fc2_fc3_ternary(model, mask)
torch.save(model.state_dict(), "ternary_model.pt")
print("✅ Mentés kész: ternary_model.pt")

print("\n🧪 Test run after ternary quantization (fc2, fc3):")
acc = test(model, test_loader)
print(f"🎯 Accuracy after ternary quantization: {acc:.4f}")

# --- 3. Further training with SimpleMLPWithParallel, only parallel layers trainable ---
print("\n🚀 Further training with parallel layers only...")

# Load ternary weights into main branch, parallel layers zeroed and trainable
parallel_model = SimpleMLPWithParallel().to(device)
state_dict = torch.load("ternary_model.pt")
# Load only matching keys (fc1, fc2, fc3)
for name, param in parallel_model.named_parameters():
    if name in state_dict and not (name.startswith('fc2_parallel') or name.startswith('fc3_parallel')):
        param.data.copy_(state_dict[name])
# Zero and unfreeze parallel layers
for name, param in parallel_model.named_parameters():
    if name.startswith('fc2_parallel') or name.startswith('fc3_parallel'):
        param.data.zero_()
        param.requires_grad = True
    else:
        param.requires_grad = False

# Only optimize parallel layers
optimizer = torch.optim.SGD(
    [p for n, p in parallel_model.named_parameters() if p.requires_grad], lr=lr
)

epochs_finetune = 3
for epoch in range(epochs_finetune):
    train(parallel_model, optimizer, train_loader)
    acc = test(parallel_model, test_loader)
    print(f"Finetune epoch {epoch+1}/{epochs_finetune}, accuracy: {acc:.4f}")

print("\n✅ Done. Final accuracy with parallel layers:")
acc = test(parallel_model, test_loader)
print(f"🎯 Final accuracy: {acc:.4f}")

# Save the final parallel model
torch.save(parallel_model.state_dict(), "final_parallel_model.pt")
print("✅ Final parallel model saved: final_parallel_model.pt")

# Print weight counts per layer for the final model
report_weight_counts_per_layer(parallel_model)