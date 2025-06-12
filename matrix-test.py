import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Adatok: 100x100-as ritka mátrix
m, n = 100, 100
sparsity = 0.9
A_np = np.zeros((m, n))
mask = np.random.rand(m, n) > sparsity
A_np[mask] = np.random.uniform(-5, 5, size=mask.sum())

# PyTorch tensorra váltás
A = torch.tensor(A_np, dtype=torch.float32)

# Autoencoder modell
class Autoencoder(nn.Module):
    def __init__(self, input_dim, code_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, code_dim)
        self.decoder = nn.Linear(code_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

# Paraméterek
code_dim = 20
model = Autoencoder(input_dim=n, code_dim=code_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

# Tanítás
for epoch in range(500):
    optimizer.zero_grad()
    output, _ = model(A)
    loss = loss_fn(output, A)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.6f}")

# Low-rank mátrix előállítása
with torch.no_grad():
    _, W = model(A)                     # W: (100 × r)
    H = model.decoder.weight.data.T     # H: (r × 100)
    A_approx = W @ H                    # Rekonstrukció

print("W (100 x r) mátrix:")
print(W.numpy())

print("\nH (r x 100) mátrix:")
print(H.numpy())

rel_error = torch.norm(A - A_approx) / torch.norm(A)
print(f"Relatív hiba: {rel_error:.6f}")
