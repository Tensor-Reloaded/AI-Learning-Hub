import torch


def f1_score(x: torch.BoolTensor, y: torch.BoolTensor) -> float:
    x_sum = x.sum()
    y_sum = y.sum()
    if x_sum == 0 or y_sum == 0:
        if x_sum == 0 and y_sum == 0:
            return 1.0
        return 0.0
    return 2 * (x & y).sum() / (x_sum + y_sum)


def f1_macro(x: torch.Tensor, y: torch.Tensor, classes: int) -> float:
    result = 0.0
    for c in range(classes):
        result += f1_score(x == c, y == c)
    return result / classes
