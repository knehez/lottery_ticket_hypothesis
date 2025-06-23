import os
import sys
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ======== 0. Ha kell: DLL-ek elérési útjának beállítása (Windows only) ========
os.environ["PATH"] = r"c:\Users\karol\Downloads\torolni\TensorRT-10.12.0.36\lib" + ";" + os.environ["PATH"]

import tensorrt as trt

# ======== 1. TensorRT Engine betöltése ========
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

# ONNX fájl betöltése
onnx_path = "pruned_model.onnx"
with open(onnx_path, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parsing failed")

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)  # ha támogatja a GPU
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
# 🔧 Hozzáadjuk az optimization profile-t
input_name = network.get_input(0).name
profile = builder.create_optimization_profile()
profile.set_shape(input_name,
                  min=(1, 1, 28, 28),
                  opt=(8, 1, 28, 28),
                  max=(32, 1, 28, 28))
config.add_optimization_profile(profile)
# ✨ ÚJ: serialized engine
serialized_engine = builder.build_serialized_network(network, config)
assert serialized_engine is not None

# Írás fájlba
with open("pruned_model.trt", "wb") as f:
    f.write(serialized_engine)

def load_engine(engine_file_path):
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine("pruned_model.trt")
context = engine.create_execution_context()

# ======== 2. MNIST tesztadatok betöltése (PyTorch) ========
transform = transforms.Compose([
    transforms.ToTensor(),
])
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ======== 3. Inference TensorRT-tel (TensorRT 10 API) ========
# ======== Tensor nevek lekérdezése az új API-val ========
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        input_name = name
    elif engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
        output_name = name

# 2. Beállítjuk a dinamikus shape-et
context.set_input_shape(input_name, (1, 1, 28, 28))
input_shape = context.get_tensor_shape(input_name)
output_shape = context.get_tensor_shape(output_name)

# 3. GPU memóriaterületek foglalása
d_input = cuda.mem_alloc(int(np.prod(input_shape) * np.float32().itemsize))
d_output = cuda.mem_alloc(int(np.prod(output_shape) * np.float32().itemsize))

# 4. Eredmények összegzése
correct = 0
total = 0

stream = cuda.Stream()

for i, (img, label) in enumerate(test_loader):
    np_img = img.numpy().astype(np.float32)

    cuda.memcpy_htod_async(d_input, np_img, stream)

    # TensorRT 10: külön set_tensor_address() hívások
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    context.execute_async_v3(stream.handle)

    output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh_async(output, d_output, stream)
    stream.synchronize()

    pred = np.argmax(output)
    total += 1
    correct += int(pred == label.item())

    if i % 1000 == 0:
        print(f"Sample {i}, Prediction: {pred}, Ground truth: {label.item()}")

print(f"\nAccuracy: {correct / total * 100:.2f}% ({correct}/{total})")
