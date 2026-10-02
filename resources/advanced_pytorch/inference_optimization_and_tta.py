import os
import sys
from itertools import product
from pathlib import Path
from typing import Tuple

import timm
import torch
from prettytable import PrettyTable
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import hflip
from timed_decorator.simple_timed import timed
from tqdm import tqdm

try:
    import torch_tensorrt

    HAS_TORCH_TENSORRT = True
except ImportError:
    HAS_TORCH_TENSORRT = False

try:
    import openvino
    import openvino.torch

    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False


class ClassificationModel(nn.Module):
    def __init__(self, backbone_name: str = "resnet18", num_classes: int = 10):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.weight.size(1), num_classes)

    def forward(self, x):
        return self.backbone(x)


def create_model(model_path: str, device: torch.device, model_type: str):
    model_data = torch.load(model_path, map_location=device, weights_only=True)

    model = ClassificationModel(model_data["model_name"], model_data["num_classes"])
    model = model.to(device)
    model.load_state_dict(model_data["model_state_dict"])
    model.eval()

    if model_type == "raw model":
        return model
    if sys.version_info < (3, 14):
        # These do not work anymore on py >= 3.14
        if model_type == "scripted model":
            return torch.jit.script(model)
        if model_type == "traced model":
            return torch.jit.trace(model, torch.rand((5, 3, 32, 32), device=device))
        if model_type == "frozen model":
            return torch.jit.freeze(torch.jit.script(model))
        if model_type == "optimized for inference":
            return torch.jit.optimize_for_inference(torch.jit.script(model))
    if model_type == "compiled model":
        return torch.compile(model, backend="inductor", dynamic=False)
    if model_type == "compiled reduce-overhead":
        return torch.compile(model, backend="inductor", mode="reduce-overhead", dynamic=False)
    if model_type == "compiled max-autotune":
        return torch.compile(model, backend="inductor", mode="max-autotune", dynamic=False)
    if model_type == "TensorRT":
        if device.type != "cuda":
            raise RuntimeError("Torch-TensorRT requires a CUDA device")
        if not HAS_TORCH_TENSORRT:
            raise RuntimeError("torch_tensorrt is not installed")
        return torch.compile(
            model,
            backend="torch_tensorrt",
            dynamic=False, options={
                "optimization_level": 5,
            }
        )
    if model_type == "OpenVINO":
        if device.type != "cpu":
            raise RuntimeError("This OpenVINO experiment is configured for CPU only")
        if not HAS_OPENVINO:
            raise RuntimeError("OpenVINO is not installed")

        return torch.compile(
            model,
            backend="openvino",
            dynamic=False,
            options={
                "device": "CPU",
                "config": {
                    "PERFORMANCE_HINT": "LATENCY",
                },
            },
        )
    print(f"Model {model_type} not supported")
    return None


@timed(stdout=False, return_time=True, use_seconds=True)
def tta_inference(model, batches: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], device: torch.device,
                  tta_type: str) -> float:
    total = 0
    correct = 0

    for data, target in batches:
        data = data.to(device)

        predicted = model(data)
        if tta_type == "mirroring":
            predicted += model(hflip(data))
        elif tta_type == "translate":
            padding_size = 2
            image_size = 32
            # We pad using the same value the model has seen during training
            padded = v2.functional.pad(data, [padding_size], fill=0.5)
            for i in [-2, 0, 2]:
                for j in [-2, 0, 2]:
                    if i == 0 and j == 0:
                        continue
                    x = padding_size + i
                    y = padding_size + j
                    predicted += model(padded[:, :, x:x + image_size, y:y + image_size])
        elif tta_type == "mirroring_and_translate":
            padding_size = 2
            image_size = 32
            padded = v2.functional.pad(data, [padding_size], fill=0.5)
            for i in [-2, 0, 2]:
                for j in [-2, 0, 2]:
                    if i == 0 and j == 0:
                        continue
                    x = padding_size + i
                    y = padding_size + j
                    aux = padded[:, :, x:x + image_size, y:y + image_size]
                    predicted += model(aux)
                    predicted += model(hflip(aux))

        correct += (predicted.cpu().argmax(dim=1) == target).sum().item()
        total += data.size(0)

    return round(correct / total, 4)


def inference(model, batches: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], device: torch.device, tta_type: str,
              dtype: torch.dtype, model_type: str) -> Tuple[float, float]:
    enable_autocast = device.type == "cuda" and dtype != torch.float32
    # Autocast is slow for cpu, so we disable it.
    # Also, if the device type is mps, autocast might not work (?)
    accuracy, elapsed = "N/A", "N/A"
    try:

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=enable_autocast), torch.inference_mode():
            accuracy, elapsed = tta_inference(model, batches, device, tta_type)
    except Exception as e:
        # Debug only

        # import traceback
        # traceback.print_exc()
        print(f"Model type {model_type} failed on {dtype} on {device.type} because of {e}")

    return accuracy, elapsed


def prepare_data(data_path: str) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.491, 0.482, 0.446), std=(0.247, 0.243, 0.261), inplace=True)
    ])
    dataset = CIFAR10(root=data_path, train=False, transform=transforms, download=True)
    dataloader = DataLoader(dataset, batch_size=200)
    return tuple([x for x in dataloader])


def warmup(model, device: torch.device, dtype: torch.dtype):
    enable_autocast = device.type == "cuda" and dtype != torch.float32
    sample = torch.rand((200, 3, 32, 32), device=device)

    try:
        with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=enable_autocast,
        ):
            model(sample)
    except:
        # Exception will be printed during inference
        pass

    if device.type == "cuda":
        torch.cuda.synchronize()


def do_speed_test(data: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
                  model_types: Tuple[str, ...],
                  dtypes: Tuple[torch.dtype, ...],
                  tta_types: Tuple[str, ...],
                  devices: Tuple[torch.device | None, ...],
                  model_path: str):
    tta_type = "none"
    with tqdm(total=len(devices) * len(dtypes) * len(model_types), desc="Speed experiments") as tbar:
        for device, dtype in product(devices, dtypes):
            if device is None:
                tbar.update(len(model_types))
                continue
            speed_results = PrettyTable()
            speed_results.field_names = ["Device", "Dtype", "TTA Type", "Model Type", "Accuracy", "Elapsed"]

            for model_type in model_types:
                tbar.update()
                if model_type == "TensorRT" and device.type != "cuda":
                    continue
                if model_type == "OpenVINO" and device.type != "cpu":
                    continue
                model = create_model(model_path, device, model_type)
                if model is None:
                    continue
                warmup(model, device, dtype)
                accuracy, elapsed = inference(model, data, device, tta_type, dtype, model_type)
                speed_results.add_row([device, dtype, tta_type, model_type, accuracy, elapsed])

            print(speed_results)

    # Possible results:
    # +--------+----------------+----------+--------------------------+----------+-------------+
    # | Device |     Dtype      | TTA Type |        Model Type        | Accuracy |   Elapsed   |
    # +--------+----------------+----------+--------------------------+----------+-------------+
    # |  cuda  | torch.bfloat16 |   none   |        raw model         |  0.8628  | 0.154228217 |
    # |  cuda  | torch.bfloat16 |   none   |      scripted model      |  0.8628  |  0.18227771 |
    # |  cuda  | torch.bfloat16 |   none   |       traced model       |  0.8628  | 0.145980372 |
    # |  cuda  | torch.bfloat16 |   none   |       frozen model       |  0.8621  | 0.131838102 |
    # |  cuda  | torch.bfloat16 |   none   | optimized for inference  |   N/A    |     N/A     |
    # |  cuda  | torch.bfloat16 |   none   |      compiled model      |  0.8631  |  0.44783471 |
    # |  cuda  | torch.bfloat16 |   none   | compiled reduce-overhead |  0.8631  | 0.467748413 |
    # |  cuda  | torch.bfloat16 |   none   |  compiled max-autotune   |  0.8631  | 0.489904809 |
    # |  cuda  | torch.bfloat16 |   none   |         TensorRT         |   N/A    |     N/A     |
    # +--------+----------------+----------+--------------------------+----------+-------------+
    #
    # --------+---------------+----------+--------------------------+----------+-------------+
    # | Device |     Dtype     | TTA Type |        Model Type        | Accuracy |   Elapsed   |
    # +--------+---------------+----------+--------------------------+----------+-------------+
    # |  cuda  | torch.float16 |   none   |        raw model         |  0.8628  | 0.156887009 |
    # |  cuda  | torch.float16 |   none   |      scripted model      |  0.8629  | 0.171385859 |
    # |  cuda  | torch.float16 |   none   |       traced model       |  0.8628  |  0.16924638 |
    # |  cuda  | torch.float16 |   none   |       frozen model       |  0.8629  | 0.147212552 |
    # |  cuda  | torch.float16 |   none   | optimized for inference  |   N/A    |     N/A     |
    # |  cuda  | torch.float16 |   none   |      compiled model      |  0.8628  | 0.453247213 |
    # |  cuda  | torch.float16 |   none   | compiled reduce-overhead |  0.8628  |  0.17567726 |
    # |  cuda  | torch.float16 |   none   |  compiled max-autotune   |  0.8628  | 0.162664373 |
    # |  cuda  | torch.float16 |   none   |         TensorRT         |  0.8628  | 0.160568678 |
    # +--------+---------------+----------+--------------------------+----------+-------------+
    #
    # --------+---------------+----------+--------------------------+----------+-------------+
    # | Device |     Dtype     | TTA Type |        Model Type        | Accuracy |   Elapsed   |
    # +--------+---------------+----------+--------------------------+----------+-------------+
    # |  cuda  | torch.float32 |   none   |        raw model         |  0.8628  | 0.218422712 |
    # |  cuda  | torch.float32 |   none   |      scripted model      |  0.8628  | 0.250669589 |
    # |  cuda  | torch.float32 |   none   |       traced model       |  0.8628  | 0.213143711 |
    # |  cuda  | torch.float32 |   none   |       frozen model       |  0.8628  | 0.207792728 |
    # |  cuda  | torch.float32 |   none   | optimized for inference  |  0.8628  | 0.385064424 |
    # |  cuda  | torch.float32 |   none   |      compiled model      |  0.8628  | 0.217624749 |
    # |  cuda  | torch.float32 |   none   | compiled reduce-overhead |  0.8628  | 0.233234776 |
    # |  cuda  | torch.float32 |   none   |  compiled max-autotune   |  0.8628  | 0.229791399 |
    # |  cuda  | torch.float32 |   none   |         TensorRT         |  0.8628  | 0.220956168 |
    # +--------+---------------+----------+--------------------------+----------+-------------+
    #
    # --------+----------------+----------+--------------------------+----------+-------------+
    # | Device |     Dtype      | TTA Type |        Model Type        | Accuracy |   Elapsed   |
    # +--------+----------------+----------+--------------------------+----------+-------------+
    # |  cpu   | torch.bfloat16 |   none   |        raw model         |  0.8628  | 2.608457818 |
    # |  cpu   | torch.bfloat16 |   none   |      scripted model      |  0.8628  | 2.998978515 |
    # |  cpu   | torch.bfloat16 |   none   |       traced model       |  0.8628  | 4.291061425 |
    # |  cpu   | torch.bfloat16 |   none   |       frozen model       |  0.8628  |  3.32382194 |
    # |  cpu   | torch.bfloat16 |   none   | optimized for inference  |  0.8628  | 3.243459987 |
    # |  cpu   | torch.bfloat16 |   none   |      compiled model      |  0.8628  | 2.891890332 |
    # |  cpu   | torch.bfloat16 |   none   | compiled reduce-overhead |  0.8628  | 3.483068128 |
    # |  cpu   | torch.bfloat16 |   none   |  compiled max-autotune   |  0.8628  | 4.119310992 |
    # |  cpu   | torch.bfloat16 |   none   |         OpenVINO         |  0.8628  | 2.537127951 |
    # +--------+----------------+----------+--------------------------+----------+-------------+
    #
    # --------+---------------+----------+--------------------------+----------+-------------+
    # | Device |     Dtype     | TTA Type |        Model Type        | Accuracy |   Elapsed   |
    # +--------+---------------+----------+--------------------------+----------+-------------+
    # |  cpu   | torch.float16 |   none   |        raw model         |  0.8628  | 2.131326793 |
    # |  cpu   | torch.float16 |   none   |      scripted model      |  0.8628  |  4.70224819 |
    # |  cpu   | torch.float16 |   none   |       traced model       |  0.8628  |  2.82985854 |
    # |  cpu   | torch.float16 |   none   |       frozen model       |  0.8628  | 1.978930938 |
    # |  cpu   | torch.float16 |   none   | optimized for inference  |  0.8628  | 2.512271403 |
    # |  cpu   | torch.float16 |   none   |      compiled model      |  0.8628  | 1.962599676 |
    # |  cpu   | torch.float16 |   none   | compiled reduce-overhead |  0.8628  | 2.943062132 |
    # |  cpu   | torch.float16 |   none   |  compiled max-autotune   |  0.8628  |  2.93777492 |
    # |  cpu   | torch.float16 |   none   |         OpenVINO         |  0.8628  | 3.068850921 |
    # +--------+---------------+----------+--------------------------+----------+-------------+
    #
    # --------+---------------+----------+--------------------------+----------+-------------+
    # | Device |     Dtype     | TTA Type |        Model Type        | Accuracy |   Elapsed   |
    # +--------+---------------+----------+--------------------------+----------+-------------+
    # |  cpu   | torch.float32 |   none   |        raw model         |  0.8628  | 3.569214138 |
    # |  cpu   | torch.float32 |   none   |      scripted model      |  0.8628  | 4.810741705 |
    # |  cpu   | torch.float32 |   none   |       traced model       |  0.8628  | 2.347504725 |
    # |  cpu   | torch.float32 |   none   |       frozen model       |  0.8628  | 3.256672198 |
    # |  cpu   | torch.float32 |   none   | optimized for inference  |  0.8628  | 1.870113525 |
    # |  cpu   | torch.float32 |   none   |      compiled model      |  0.8628  | 2.273230307 |
    # |  cpu   | torch.float32 |   none   | compiled reduce-overhead |  0.8628  | 2.200739602 |
    # |  cpu   | torch.float32 |   none   |  compiled max-autotune   |  0.8628  | 2.764132795 |
    # |  cpu   | torch.float32 |   none   |         OpenVINO         |  0.8628  | 2.583271363 |
    # +--------+---------------+----------+--------------------------+----------+-------------+


def do_tta_test(data: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
                model_types: Tuple[str, ...],
                dtypes: Tuple[torch.dtype, ...],
                tta_types: Tuple[str, ...],
                devices: Tuple[torch.device | None, ...],
                model_path: str):
    tta_results = PrettyTable()
    tta_results.field_names = ["Device", "Dtype", "TTA Type", "Model Type", "Accuracy", "Elapsed"]

    device = devices[0] if devices[0] is not None else devices[1]
    model_type = "raw model"

    for dtype, tta_type in tqdm(tuple(product(dtypes, tta_types)), desc="TTA experiments"):
        if device is None:
            continue
        model = create_model(model_path, device, model_type)
        accuracy, elapsed = inference(model, data, device, tta_type, dtype, model_type)
        tta_results.add_row([device, dtype, tta_type, model_type, accuracy, elapsed])

    print(tta_results)

    # +--------+----------------+-------------------------+------------+----------+-----------+
    # | Device |     Dtype      |         TTA Type        | Model Type | Accuracy |  Elapsed  |
    # +--------+----------------+-------------------------+------------+----------+-----------+
    # |  cuda  | torch.bfloat16 |           none          | raw model  |  0.8628  | 0.5250391 |
    # |  cuda  | torch.bfloat16 |        mirroring        | raw model  |  0.8729  | 0.5590067 |
    # |  cuda  | torch.bfloat16 |        translate        | raw model  |  0.8734  | 1.8547866 |
    # |  cuda  | torch.bfloat16 | mirroring_and_translate | raw model  |  0.8783  | 2.9989047 |
    # |  cuda  | torch.float16  |           none          | raw model  |  0.8628  | 0.2334918 |
    # |  cuda  | torch.float16  |        mirroring        | raw model  |  0.8729  | 0.2548314 |
    # |  cuda  | torch.float16  |        translate        | raw model  |  0.8735  | 1.2141909 |
    # |  cuda  | torch.float16  | mirroring_and_translate | raw model  |  0.8785  | 2.2871451 |
    # |  cuda  | torch.float32  |           none          | raw model  |  0.8628  | 0.2817045 |
    # |  cuda  | torch.float32  |        mirroring        | raw model  |  0.8729  | 0.3823375 |
    # |  cuda  | torch.float32  |        translate        | raw model  |  0.8735  | 2.0736249 |
    # |  cuda  | torch.float32  | mirroring_and_translate | raw model  |  0.8785  | 3.9413618 |
    # +--------+----------------+-------------------------+------------+----------+-----------+


def main(model_path: str):
    data = prepare_data("./data")
    model_types = (
        "raw model",
        "scripted model",
        "traced model",
        "frozen model",
        "optimized for inference",
        "compiled model",
        "compiled reduce-overhead",
        "compiled max-autotune",
        "TensorRT",
        "OpenVINO",
    )
    dtypes = (
        torch.bfloat16,
        torch.half,
        torch.float32
    )
    tta_types = (
        "none",
        "mirroring",
        "translate",
        "mirroring_and_translate",
    )
    devices = (
        torch.accelerator.current_accelerator() if torch.accelerator.is_available() else None,
        torch.device("cpu"),
    )

    do_speed_test(data, model_types, dtypes, tta_types, devices, model_path)
    do_tta_test(data, model_types, dtypes, tta_types, devices, model_path)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    print("You need to install torch-tensorrt and openvino")
    if os.name == "nt":
        print("You need to install triton-windows")

    torch.set_float32_matmul_precision('high')
    model_path = os.path.join(str(Path(__file__).parent.resolve()), "checkpoints", "best.pth")
    main(model_path)
