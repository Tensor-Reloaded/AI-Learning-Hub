import os
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
    if model_type == "scripted model":
        return torch.jit.script(model)
    if model_type == "traced model":
        return torch.jit.trace(model, torch.rand((5, 3, 32, 32), device=device))
    if model_type == "frozen model":
        return torch.jit.freeze(torch.jit.script(model))
    if model_type == "optimized for inference":
        return torch.jit.optimize_for_inference(torch.jit.script(model))
    if model_type == "compiled model":
        if os.name == "nt":
            print("torch.compile is not supported on Windows. Try Linux or WSL instead.")
            return model
        return torch.compile(model)
    raise RuntimeError("std::unreachable")


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
            padded = v2.functional.pad(data, [padding_size])
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
            padded = v2.functional.pad(data, [padding_size])
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
    except:
        # Debug only

        # import traceback
        # traceback.print_exc()
        print(f"Model type {model_type} failed on {dtype} on {device.type}")

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
                model = create_model(model_path, device, model_type)
                accuracy, elapsed = inference(model, data, device, tta_type, dtype, model_type)
                speed_results.add_row([device, dtype, tta_type, model_type, accuracy, elapsed])
                tbar.update()

            print(speed_results)

    # +--------+----------------+----------+-------------------------+----------+-------------+
    # | Device |     Dtype      | TTA Type |        Model Type       | Accuracy |   Elapsed   |
    # +--------+----------------+----------+-------------------------+----------+-------------+
    # |  cuda  | torch.bfloat16 |   none   |        raw model        |  0.8627  |  0.74056251 |
    # |  cuda  | torch.bfloat16 |   none   |      scripted model     |  0.8627  | 0.550711881 |
    # |  cuda  | torch.bfloat16 |   none   |      scripted model     |  0.8627  | 0.466999062 |
    # |  cuda  | torch.bfloat16 |   none   |       traced model      |  0.8627  | 0.505114635 |
    # |  cuda  | torch.bfloat16 |   none   |       traced model      |  0.8627  | 0.497691016 |
    # |  cuda  | torch.bfloat16 |   none   |       frozen model      |  0.8618  | 0.630178739 |
    # |  cuda  | torch.bfloat16 |   none   |       frozen model      |  0.8618  | 0.431321397 |
    # |  cuda  | torch.bfloat16 |   none   | optimized for inference |   N/A    |     N/A     |
    # |  cuda  | torch.bfloat16 |   none   | optimized for inference |   N/A    |     N/A     |
    # |  cuda  | torch.bfloat16 |   none   |      compiled model     |  0.863   |  1.37197609 |
    # |  cuda  | torch.bfloat16 |   none   |      compiled model     |  0.863   | 0.439737346 |
    # +--------+----------------+----------+-------------------------+----------+-------------+

    # +--------+----------------+----------+-------------------------+----------+-------------+
    # | Device |     Dtype      | TTA Type |        Model Type       | Accuracy |   Elapsed   |
    # +--------+----------------+----------+-------------------------+----------+-------------+
    # |  cuda  | torch.float16  |   none   |        raw model        |  0.8629  | 0.934939784 |
    # |  cuda  | torch.float16  |   none   |      scripted model     |  0.8629  | 0.776701284 |
    # |  cuda  | torch.float16  |   none   |      scripted model     |  0.8629  |  0.65642132 |
    # |  cuda  | torch.float16  |   none   |       traced model      |  0.8629  | 0.770792187 |
    # |  cuda  | torch.float16  |   none   |       traced model      |  0.8629  | 0.761494488 |
    # |  cuda  | torch.float16  |   none   |       frozen model      |  0.8629  | 0.449910122 |
    # |  cuda  | torch.float16  |   none   |       frozen model      |  0.8629  | 0.428042867 |
    # |  cuda  | torch.float16  |   none   | optimized for inference |   N/A    |     N/A     |
    # |  cuda  | torch.float16  |   none   | optimized for inference |   N/A    |     N/A     |
    # |  cuda  | torch.float16  |   none   |      compiled model     |  0.863   | 1.041609176 |
    # |  cuda  | torch.float16  |   none   |      compiled model     |  0.863   | 0.304629578 |
    # +--------+----------------+----------+-------------------------+----------+-------------+

    # +--------+----------------+----------+-------------------------+----------+-------------+
    # | Device |     Dtype      | TTA Type |        Model Type       | Accuracy |   Elapsed   |
    # +--------+----------------+----------+-------------------------+----------+-------------+
    # |  cuda  | torch.float32  |   none   |        raw model        |  0.8628  | 0.928239116 |
    # |  cuda  | torch.float32  |   none   |      scripted model     |  0.8628  | 0.869112261 |
    # |  cuda  | torch.float32  |   none   |      scripted model     |  0.8628  | 0.818328065 |
    # |  cuda  | torch.float32  |   none   |       traced model      |  0.8628  | 0.831756814 |
    # |  cuda  | torch.float32  |   none   |       traced model      |  0.8628  | 0.835166337 |
    # |  cuda  | torch.float32  |   none   |       frozen model      |  0.8628  | 0.635774185 |
    # |  cuda  | torch.float32  |   none   |       frozen model      |  0.8628  | 0.884842387 |
    # |  cuda  | torch.float32  |   none   | optimized for inference |  0.8628  | 6.401095805 |
    # |  cuda  | torch.float32  |   none   | optimized for inference |  0.8628  | 6.383807842 |
    # |  cuda  | torch.float32  |   none   |      compiled model     |  0.8628  | 0.979224372 |
    # |  cuda  | torch.float32  |   none   |      compiled model     |  0.8628  | 0.510377062 |
    # +--------+----------------+----------+-------------------------+----------+-------------+

    # +--------+----------------+----------+-------------------------+----------+-------------+
    # | Device |     Dtype      | TTA Type |        Model Type       | Accuracy |   Elapsed   |
    # +--------+----------------+----------+-------------------------+----------+-------------+
    # |  cpu   | torch.bfloat16 |   none   |        raw model        |  0.8628  | 2.859445197 |
    # |  cpu   | torch.bfloat16 |   none   |      scripted model     |  0.8628  | 2.635952067 |
    # |  cpu   | torch.bfloat16 |   none   |      scripted model     |  0.8628  | 2.604736663 |
    # |  cpu   | torch.bfloat16 |   none   |       traced model      |  0.8628  | 2.631448843 |
    # |  cpu   | torch.bfloat16 |   none   |       traced model      |  0.8628  | 2.576900248 |
    # |  cpu   | torch.bfloat16 |   none   |       frozen model      |  0.8628  | 2.546161701 |
    # |  cpu   | torch.bfloat16 |   none   |       frozen model      |  0.8628  | 2.502300936 |
    # |  cpu   | torch.bfloat16 |   none   | optimized for inference |  0.8628  | 2.281604414 |
    # |  cpu   | torch.bfloat16 |   none   | optimized for inference |  0.8628  | 2.225087941 |
    # |  cpu   | torch.bfloat16 |   none   |      compiled model     |  0.8628  |  3.58207681 |
    # |  cpu   | torch.bfloat16 |   none   |      compiled model     |  0.8628  | 1.722796112 |
    # +--------+----------------+----------+-------------------------+----------+-------------+

    # +--------+----------------+----------+-------------------------+----------+-------------+
    # | Device |     Dtype      | TTA Type |        Model Type       | Accuracy |   Elapsed   |
    # +--------+----------------+----------+-------------------------+----------+-------------+
    # |  cpu   | torch.float16  |   none   |        raw model        |  0.8628  | 2.737279273 |
    # |  cpu   | torch.float16  |   none   |      scripted model     |  0.8628  | 2.562341959 |
    # |  cpu   | torch.float16  |   none   |      scripted model     |  0.8628  | 2.652842815 |
    # |  cpu   | torch.float16  |   none   |       traced model      |  0.8628  | 2.639518142 |
    # |  cpu   | torch.float16  |   none   |       traced model      |  0.8628  | 2.735255652 |
    # |  cpu   | torch.float16  |   none   |       frozen model      |  0.8628  | 2.903561699 |
    # |  cpu   | torch.float16  |   none   |       frozen model      |  0.8628  | 2.962546338 |
    # |  cpu   | torch.float16  |   none   | optimized for inference |  0.8628  | 2.344554807 |
    # |  cpu   | torch.float16  |   none   | optimized for inference |  0.8628  | 2.360003218 |
    # |  cpu   | torch.float16  |   none   |      compiled model     |  0.8628  | 1.730791658 |
    # |  cpu   | torch.float16  |   none   |      compiled model     |  0.8628  | 1.754020479 |
    # +--------+----------------+----------+-------------------------+----------+-------------+

    # +--------+----------------+----------+-------------------------+----------+-------------+
    # | Device |     Dtype      | TTA Type |        Model Type       | Accuracy |   Elapsed   |
    # +--------+----------------+----------+-------------------------+----------+-------------+
    # |  cpu   | torch.float32  |   none   |        raw model        |  0.8628  |  2.65187362 |
    # |  cpu   | torch.float32  |   none   |      scripted model     |  0.8628  | 2.620745015 |
    # |  cpu   | torch.float32  |   none   |      scripted model     |  0.8628  | 2.583938025 |
    # |  cpu   | torch.float32  |   none   |       traced model      |  0.8628  |  2.68518527 |
    # |  cpu   | torch.float32  |   none   |       traced model      |  0.8628  | 2.670001929 |
    # |  cpu   | torch.float32  |   none   |       frozen model      |  0.8628  | 2.853723278 |
    # |  cpu   | torch.float32  |   none   |       frozen model      |  0.8628  | 2.903551512 |
    # |  cpu   | torch.float32  |   none   | optimized for inference |  0.8628  |  2.47234354 |
    # |  cpu   | torch.float32  |   none   | optimized for inference |  0.8628  | 2.308440241 |
    # |  cpu   | torch.float32  |   none   |      compiled model     |  0.8628  |  1.78701703 |
    # |  cpu   | torch.float32  |   none   |      compiled model     |  0.8628  | 1.771141438 |
    # +--------+----------------+----------+-------------------------+----------+-------------+


def do_tta_test(data: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
                model_types: Tuple[str, ...],
                dtypes: Tuple[torch.dtype, ...],
                tta_types: Tuple[str, ...],
                devices: Tuple[torch.device | None, ...],
                model_path: str):
    tta_results = PrettyTable()
    tta_results.field_names = ["Device", "Dtype", "TTA Type", "Model Type", "Accuracy", "Elapsed"]

    device = devices[0] if devices[0] is not None else devices[1]
    model_type = "scripted model"

    for dtype, tta_type in tqdm(tuple(product(dtypes, tta_types)), desc="TTA experiments"):
        if device is None:
            continue
        model = create_model(model_path, device, model_type)
        accuracy, elapsed = inference(model, data, device, tta_type, dtype, model_type)
        tta_results.add_row([device, dtype, tta_type, model_type, accuracy, elapsed])

    print(tta_results)

    # +--------+----------------+-------------------------+----------------+----------+--------------+
    # | Device |     Dtype      |         TTA Type        |   Model Type   | Accuracy |   Elapsed    |
    # +--------+----------------+-------------------------+----------------+----------+--------------+
    # |  cuda  | torch.bfloat16 |           none          | scripted model |  0.8627  | 0.686984949  |
    # |  cuda  | torch.bfloat16 |        mirroring        | scripted model |  0.8729  | 0.912047684  |
    # |  cuda  | torch.bfloat16 |        translate        | scripted model |  0.8729  |  4.07882746  |
    # |  cuda  | torch.bfloat16 | mirroring_and_translate | scripted model |  0.8795  | 7.338382004  |
    # |  cuda  | torch.float16  |           none          | scripted model |  0.8629  | 0.897940889  |
    # |  cuda  | torch.float16  |        mirroring        | scripted model |  0.8729  | 1.272187126  |
    # |  cuda  | torch.float16  |        translate        | scripted model |  0.8727  |  5.75703762  |
    # |  cuda  | torch.float16  | mirroring_and_translate | scripted model |  0.8795  | 10.648630153 |
    # |  cuda  | torch.float32  |           none          | scripted model |  0.8628  | 0.817683462  |
    # |  cuda  | torch.float32  |        mirroring        | scripted model |  0.8728  | 1.600388468  |
    # |  cuda  | torch.float32  |        translate        | scripted model |  0.8726  | 6.986309829  |
    # |  cuda  | torch.float32  | mirroring_and_translate | scripted model |  0.8795  | 13.140209483 |
    # +--------+----------------+-------------------------+----------------+----------+--------------+


def main(model_path: str):
    data = prepare_data("./data")
    model_types = (
        "raw model",
        "scripted model",
        "scripted model",
        "traced model",
        "traced model",
        "frozen model",
        "frozen model",
        "optimized for inference",
        "optimized for inference",
        "compiled model",
        "compiled model",
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
    torch.set_float32_matmul_precision('high')
    model_path = os.path.join(str(Path(__file__).parent.resolve()), "checkpoints", "best.pth")
    main(model_path)

