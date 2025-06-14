from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import time
# Ugyanaz a transform, mint tanításnál
transform = transforms.ToTensor()

test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000)

def test(model, dataloader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return correct / len(dataloader.dataset)

model = torch.jit.load("pruned_model.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

start_time = time.time()
acc = test(model, test_loader, device)
end_time = time.time()
print(f"[TorchScript Inference] Accuracy: {acc:.4f}")
print(f"[TorchScript Inference] Elapsed time: {end_time - start_time:.4f} seconds")