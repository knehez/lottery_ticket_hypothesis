import onnxruntime as ort
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def test_onnx_model(model_path, providers, dataloader):
    """
    Tests an ONNX model for inference time and accuracy.

    Args:
        model_path (str): The path to the .onnx model file.
        providers (list): The list of ONNX Runtime execution providers.
        dataloader (DataLoader): The DataLoader containing the test dataset.

    Returns:
        tuple: A tuple containing (average_time_ms, accuracy_percent).
    """
    print(f"\n--- Testing: {model_path} ---")
    try:
        sess = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        print(f"Error loading model ({model_path}): {e}")
        return None, None

    input_name = sess.get_inputs()[0].name

    all_times = []
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images_np = images.numpy().astype(np.float32)
            start = time.perf_counter()
            outputs = sess.run(None, {input_name: images_np})
            end = time.perf_counter()
            all_times.append(end - start)

            # Prediction and accuracy calculation
            pred = np.argmax(outputs[0], axis=1)
            correct += np.sum(pred == labels.numpy())
            total += labels.size(0)

    if not all_times:
        return 0, 0

    avg_time_per_batch_ms = np.mean(all_times) * 1000
    accuracy_percent = (correct / total) * 100

    print(f"Average inference time: {avg_time_per_batch_ms:.2f} ms/batch")
    print(f"Accuracy on the full test set: {accuracy_percent:.2f}%")

    return avg_time_per_batch_ms, accuracy_percent

if __name__ == "__main__":
    # Transformations for CIFAR
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load test data
    test_loader = DataLoader(
        datasets.CIFAR10('.', train=False, download=True, transform=transform),
        batch_size=128
    )

    # Load ONNX model (GPU preferred)
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2 GB
        }),
        'CPUExecutionProvider'
    ]

    # Test models
    orig_time, orig_acc = test_onnx_model('orig_model.onnx', providers, test_loader)
    pruned_time, pruned_acc = test_onnx_model('pruned_model.onnx', providers, test_loader)

    # Comparison
    print("\n\n--- COMPARISON ---")
    if orig_time is not None and pruned_time is not None:
        print(f"{'Model':<18} | {'Avg. time (ms/batch)':<25} | {'Accuracy (%)':<15}")
        print("-" * 65)
        print(f"{'Original (orig)':<18} | {orig_time:<25.2f} | {orig_acc:<15.2f}")
        print(f"{'Pruned':<18} | {pruned_time:<25.2f} | {pruned_acc:<15.2f}")

        if pruned_time > 0:
            speedup = orig_time / pruned_time
            print(f"\nSpeedup: {speedup:.2f}x")
    else:
        print("Testing one or both models failed.")
