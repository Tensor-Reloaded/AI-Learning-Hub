import os

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import v2
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
import timm
from tqdm import tqdm

disable_compile = False
compile_is_slower = False


class CachedDataset(Dataset):
    def __init__(self, dataset: Dataset, runtime_transforms: v2.Transform | None):
        # This operation caches all transformations from the wrapped dataset. Stores the results as a Tuple
        # instead of list, decreasing memory usage. Tuples also have faster indexing, even though it is negligible.
        self.dataset = tuple([x for x in dataset])

        # These are the runtime transformations that can't be cached. Usually, they involve randomness which is a
        # form of regularization for the network, and caching the randomness usually results in overfitting.
        self.runtime_transforms = runtime_transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image, label = self.dataset[i]
        if self.runtime_transforms is None:
            return image, label
        # We clone the data here, otherwise the runtime transforms might corrupt our data. They really do! 
        # You should never trust your users, even if they are yourself.
        return self.runtime_transforms(image.clone()), label


class ClassificationModel(nn.Module):
    def __init__(self, backbone_name: str = 'resnet18', num_classes: int = 10):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(backbone_name, pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.weight.size(1), num_classes)

    def forward(self, x):
        return self.backbone(x)


def get_dataset(data_path: str, is_train: bool):
    # These transformations are cached.
    initial_transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
    normalize = v2.Normalize(mean=(0.491, 0.482, 0.446), std=(0.247, 0.243, 0.261), inplace=True)
    # We use the inplace flag because we can safely change the tensors inplace when normalize is used.
    # For is_train=False, we can safely change the tensors inplace because we do it only once, when caching.
    # For is_train=True, we can safely change the tensors inplace because we clone the cached tensors first.

    if is_train:
        # We could have used RandomCrop with padding. But we are smart, and we know we cache the initial_transforms so
        # we don't compute them during runtime. Therefore, we do the padding beforehand, and apply cropping only at
        # runtime
        initial_transforms.append(
            v2.Pad(padding=4, fill=0.5)
            # Why do we fill with 0.5? That's a good question, you should experiment with the fill value.
        )
        runtime_transforms = v2.Compose([
            # For curious people: check whether RandomCrops returns a copy of its input, or a view
            v2.RandomCrop(size=32),
            v2.RandomHorizontalFlip(),
            v2.RandomErasing(scale=(0.01, 0.15), value=0.5, inplace=True),
            # If we use inplace here, it might modify the cached image. That's why we clone it.
            # Why do we fill with 0.5? See above.
            normalize,
        ])
    else:
        initial_transforms.append(normalize)
        runtime_transforms = None

    # Q: How to make this faster?
    # A: Use batched runtime transformations.

    cifar10 = CIFAR10(root=data_path, train=is_train, transform=v2.Compose(initial_transforms), download=True)
    return CachedDataset(cifar10, runtime_transforms)


def get_cutmix_or_mixup(num_classes: int = 10):
    return v2.RandomChoice([
        v2.CutMix(num_classes=num_classes),  # See the CutMix paper
        v2.MixUp(num_classes=num_classes),  # See the MixUp paper
        v2.Identity(),  # A third of all times, don't use neither CutMix nor MixUp.
    ])


class Trainer:
    def __init__(self, model: ClassificationModel, optimizer: Optimizer, criterion: nn.Module, batch_size: int = 32,
                 val_batch_size: int = 500, disable_tqdm: bool = False, save_path: str = "best.pth"):
        self.device = torch.accelerator.current_accelerator()
        print(f"Using device {self.device}")

        # Efficiency stuff
        if self.device.type == "cuda":
            # This flag tells pytorch to use the cudnn auto-tuner to find the most efficient convolution algorithm for
            # This training.
            torch.backends.cudnn.benchmark = True
            # Check this: https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
            torch.set_float32_matmul_precision('high')

        self.model = model.to(self.device)
        if disable_compile or compile_is_slower:
            # torch.jit.script is still a very good option, often faster than torch.compile, especially on windows
            self.model = torch.jit.script(model)
        else:
            # This compiles the model. See https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
            self.model.compile()
            # This compiles the step function
            self.step = torch.compile(self.step)

        self.optimizer = optimizer
        self.criterion = criterion.to(self.device)
        self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.5)

        train_dataset = get_dataset("./data", True)
        val_dataset = get_dataset("./data", False)

        # We need to shuffle the data during training. We also need to drop last, otherwise it will hurt the performance
        # of torch.compile.
        # Q: What if I am not using torch.compile? Can I set drop_last to False?
        # A: Think of the last batch, with fewer elements than the batch size you selected during training.
        #    If you were to calculate the gradient for it, how would the gradient differ from the gradient of a batch
        #    with all elements?
        # The answer is left as an exercise to the reader.
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

        self.disable_tqdm = disable_tqdm
        self.save_path = save_path
        self.best_va_acc = 0.0

        self.cutmix_mixup = get_cutmix_or_mixup(self.model.backbone.fc.weight.shape[0])

    def step(self, data: torch.Tensor, target: torch.Tensor):
        predicted = self.model(data)
        loss = self.criterion(predicted, target)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return predicted, loss

    def train(self):
        self.model.train()

        total = 0
        correct = 0
        total_loss = 0.0

        for data, target in tqdm(self.train_loader, desc="Training", leave=False, disable=self.disable_tqdm):
            # We apply cutmix or mixup. We need to pass both the data and the labels, because this kind of DA changes
            # The targets from hard labels to soft labels. Check the DA notebook for more details.
            data, target = self.cutmix_mixup(data, target)

            # Using non_blocking=True means that the transfer from cpu RAM to device RAM is done asynchronously.
            # Works when using pin_memory=True. For more details, check the references for pinning memory.
            predicted, loss = self.step(
                data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True))

            if target.ndim > 1:
                # We do this when cutmix or mixup was used, transforming the hard labels into soft labels
                target = target.argmax(1)
            # This metric is actually an approximation of an accuracy, we are checking whether the dominant class
            # predicted by the model is also equal to the dominant soft label
            # The reason we are moving the data from device back to CPU is because these calculations are usually
            # faster on CPU for small batch sizes
            # We use detach because we tell the autograd engine to not track the gradients for predicted anymore
            correct += predicted.detach().cpu().argmax(1).eq(target).sum().item()
            total += data.size(0)
            total_loss += float(loss.item())

        return correct / total, total_loss / len(self.train_loader)

    # Here we use the inference_mode. We are telling pytorch we are doing just inference, we don't need to track
    # tensor operations with the Autograd engine for automatic differentiation. This is also what torch.no_grad() does.
    # torch.inference_mode() = torch.no_grad() + promising torch we will never use any tensor created in this scope in
    # autograd tracked operations.
    # This promise allows additional optimizations, such as removing version tracking from tensors. If we violate the
    # promise, and use a tensor created in the inference_mode scope in an operation for which we need to calculate the
    # gradient, we should expect errors.
    # Recapitulating:
    #  * If we will never use Autograd, inference_mode is more optimized.
    #  * If we use Autograd, but just don't want to track some operations using Autograd, use no_grad.
    @torch.inference_mode()
    def val(self):
        self.model.eval()

        total = 0
        correct = 0
        total_loss = 0.0

        for data, target in tqdm(self.val_loader, desc="Validation", leave=False, disable=self.disable_tqdm):
            # We don't need to move the targets to device for validation.
            predicted = self.model(data.to(self.device, non_blocking=True))
            total_loss += float(self.criterion(predicted, target.to(self.device, non_blocking=True)).item())

            # Here we don't need to argmax the target, because we have hard labels. We don't use DA during validation.
            # We don't need to detach, because we are already in inference_mode
            correct += predicted.cpu().argmax(1).eq(target).sum().item()
            total += data.size(0)

        return correct / total, total_loss / len(self.val_loader)

    def run(self, epochs: int):
        with tqdm(range(epochs), desc="Training") as pbar:
            for _ in pbar:
                tr_acc, tr_loss = self.train()
                va_acc, va_loss = self.val()
                self.scheduler.step()
                if va_acc > self.best_va_acc:
                    self.best_va_acc = va_acc
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'model_name': self.model.backbone_name,
                        'num_classes': int(self.model.backbone.fc.weight.shape[0]),
                    }, self.save_path)

                pbar.set_postfix(train_acc=tr_acc, train_loss=round(tr_loss, 3), val_acc=va_acc,
                                 val_loss=round(va_loss, 3), best_val_acc=self.best_va_acc)


def main():
    model = ClassificationModel()
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True, weight_decay=0.01, fused=True)
    # optimizer = torch.optim.AdamW(model.parameters(), fused=True)
    criterion = torch.nn.CrossEntropyLoss()
    Trainer(model, optimizer, criterion, save_path="best_sgd_step_lr_20.pth").run(100)


if __name__ == "__main__":
    # If torch.compile is actually slower on your machine.
    # On my machine, 10 epochs with torch.compile take 5 minutes. With torch.jit.script, they take 4 minutes.
    # Based on my experience, for small models without custom kernels, torch.jit.script is usually faster.
    compile_is_slower = True

    if os.name == "nt":
        print("torch.compile is disabled")
        disable_compile = True
    else:
        print("torch.compile is enabled" + (" BUT not really used" if compile_is_slower else ""))
        torch._dynamo.config.capture_scalar_outputs = True

    main()
