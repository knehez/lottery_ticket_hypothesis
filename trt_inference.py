import os
import sys
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ======== 0. Windows DLL útvonal beállítása (ha szükséges) ========
os.environ["PATH"] = r"c:\Users\karol\Downloads\torolni\TensorRT-10.12.0.36\lib" + ";" + os.environ["PATH"]

import tensorrt as trt

# ======== 1. TensorRT Engine létrehozása és mentése (ha nincs még meg) ========
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

onnx_path = "pruned_model.onnx"
with open(onnx_path, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parsing failed")

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

input_name = network.get_input(0).name
profile = builder.create_optimization_profile()
profile.set_shape(input_name, min=(1, 1, 28, 28), opt=(64, 1, 28, 28), max=(1000, 1, 28, 28))
config.add_optimization_profile(profile)

serialized_engine = builder.build_serialized_network(network, config)
assert serialized_engine is not None

with open("pruned_model.trt", "wb") as f:
    f.write(serialized_engine)

# ======== 2. Engine betöltése futtatáshoz ========
def load_engine(engine_file_path):
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine("pruned_model.trt")
context = engine.create_execution_context()

# ======== 3. MNIST tesztadatok betöltése ========
batch_size = 1000
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ======== 4. Tensor nevek lekérdezése (TensorRT 10 API) ========
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        input_name = name
    elif engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
        output_name = name

context.set_input_shape(input_name, (batch_size, 1, 28, 28))
input_shape = context.get_tensor_shape(input_name)
output_shape = context.get_tensor_shape(output_name)

# ======== 5. GPU memóriafoglalás ========
d_input = cuda.mem_alloc(int(np.prod(input_shape) * np.float32().itemsize))
d_output = cuda.mem_alloc(int(np.prod(output_shape) * np.float32().itemsize))

# ======== 6. Inference batch-ben ========
correct = 0
total = 0
stream = cuda.Stream()

import time
# Warmup (optional, helps with first-batch overhead)
for _ in range(2):
    for imgs, labels in test_loader:
        actual_batch_size = imgs.shape[0]
        np_imgs = imgs.numpy().astype(np.float32)
        context.set_input_shape(input_name, (actual_batch_size, 1, 28, 28))
        context.set_tensor_address(input_name, int(d_input))
        context.set_tensor_address(output_name, int(d_output))
        cuda.memcpy_htod_async(d_input, np_imgs, stream)
        context.execute_async_v3(stream.handle)
        output = np.empty((actual_batch_size, 10), dtype=np.float32)
        cuda.memcpy_dtoh_async(output, d_output, stream)
        stream.synchronize()
        break  # Only one batch for warmup

start_time = time.time()
for i, (imgs, labels) in enumerate(test_loader):
    actual_batch_size = imgs.shape[0]
    np_imgs = imgs.numpy().astype(np.float32)

    context.set_input_shape(input_name, (actual_batch_size, 1, 28, 28))
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    # Use synchronous memcpy for better timing accuracy
    cuda.memcpy_htod(d_input, np_imgs)
    context.execute_async_v3(stream.handle)  # Use execute_async_v3 instead of execute_v3
    output = np.empty((actual_batch_size, 10), dtype=np.float32)
    cuda.memcpy_dtoh_async(output, d_output)
    stream.synchronize()

    preds = np.argmax(output, axis=1)
    correct += np.sum(preds == labels.numpy())
    total += actual_batch_size
end_time = time.time()

print(f"\nAccuracy: {correct / total * 100:.2f}% ({correct}/{total})")
print(f"Inference loop execution time: {end_time - start_time:.4f} seconds")