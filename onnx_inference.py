import onnxruntime as ort
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Transzformációk CIFAR-hoz
transform = transforms.Compose([
    transforms.ToTensor(),  # alak: [0,1] float32, shape: [C, H, W]
    # további transform itt ha kell
])

# Tesztadat betöltése
test_loader = DataLoader(
    datasets.CIFAR10('.', train=False, download=True, transform=transform),
    batch_size=128
)

# ONNX model betöltés (GPU preferált)
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2 GB
    }),
    'CPUExecutionProvider'
]
sess = ort.InferenceSession('pruned_model2.onnx', providers=providers)
input_name = sess.get_inputs()[0].name

# Inference time mérés CIFAR-10 teszt batch-eken
all_times = []
total = 0
correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.numpy().astype(np.float32)  # [batch, 3, 32, 32]
        start = time.time()
        outputs = sess.run(None, {input_name: images})
        end = time.time()
        all_times.append(end - start)
        
        # Predikció és accuracy számolás
        pred = np.argmax(outputs[0], axis=1)
        correct += np.sum(pred == labels.numpy())
        total += labels.size(0)

avg_time_per_batch = np.mean(all_times)
print(f"Átlagos inference idő: {avg_time_per_batch*1000:.2f} ms/batch")
print(f"Pontosság a teljes teszt halmazon: {correct/total*100:.2f}%")
