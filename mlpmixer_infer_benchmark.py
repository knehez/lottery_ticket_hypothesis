import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

class DeepMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim
            layers.append(nn.Linear(in_dim, out_dim, bias=False))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, num_classes, bias=False))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.seq(x)

def measure_inference(model, batch_size=128, repeat=30, device="cuda"):
    model.eval()
    dummy_input = torch.randn(batch_size, 3, 32, 32, device=device)
    with torch.no_grad():
        for _ in range(5):  # Warmup
            _ = model(dummy_input)
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeat):
            _ = model(dummy_input)
    torch.cuda.synchronize() if device == "cuda" else None
    return (time.perf_counter() - start) / repeat

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = 3 * 32 * 32
    num_layers = 10
    num_classes = 10
    hidden_dim_list = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
    batch_size = 128
    repeat = 30

    results = []
    pruned_results = []
    for hidden_dim in hidden_dim_list:
        # Eredeti
        model = DeepMLP(input_dim, hidden_dim, num_layers, num_classes).to(device)
        inf_time = measure_inference(model, batch_size=batch_size, repeat=repeat, device=device)
        params = sum(p.numel() for p in model.parameters())
        results.append((hidden_dim, inf_time, params))

        # "Pruned": minden hidden dim 0.05-ös arányban (tehát 95%-os pruning)
        pruned_dim = max(1, int(hidden_dim * 0.05))  # min. 1 neuron maradjon
        pruned_model = DeepMLP(input_dim, pruned_dim, num_layers, num_classes).to(device)
        pruned_time = measure_inference(pruned_model, batch_size=batch_size, repeat=repeat, device=device)
        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        pruned_results.append((hidden_dim, pruned_time, pruned_params))

        print(f"hidden_dim={hidden_dim}, orig params={params:,}, time={inf_time:.6f} s | pruned_dim={pruned_dim}, pruned params={pruned_params:,}, pruned_time={pruned_time:.6f} s")

    # Plot
    hidden_dims, times, param_counts = zip(*results)
    _, pruned_times, pruned_param_counts = zip(*pruned_results)

    plt.figure(figsize=(9,5))
    plt.plot(hidden_dims, times, marker='o', label='Eredeti MLP')
    plt.plot(hidden_dims, pruned_times, marker='o', label='95% pruning (hidden_dim*0.05)')
    plt.xlabel("hidden layer dim (neurons)")
    plt.ylabel("inference time (sec)")
    plt.title("MLP inference time vs. hidden dim (10 layers, batch=128)")
    plt.legend()
    plt.grid(True)
    for x, y, p in zip(hidden_dims, times, param_counts):
        plt.text(x, y, f"{p//1000}k", fontsize=8, ha='center', va='bottom', color='tab:blue')
    for x, y, p in zip(hidden_dims, pruned_times, pruned_param_counts):
        plt.text(x, y, f"{p//1000}k", fontsize=8, ha='center', va='top', color='tab:orange')
    plt.tight_layout()
    plt.show()
