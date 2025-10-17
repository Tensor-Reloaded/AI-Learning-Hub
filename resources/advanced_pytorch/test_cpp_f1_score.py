import torch
from timed_decorator.simple_timed import timed

from optimized_f1_score import f1_macro_py, f1_macro_cpp


@timed(use_seconds=True, show_args=True)
def test_f1_cpp(x: torch.Tensor, y: torch.Tensor, classes: int):
    f1_macro_cpp.f1_macro(x, y, classes)


@timed(use_seconds=True, show_args=True)
def test_f1_py(x: torch.Tensor, y: torch.Tensor, classes: int):
    f1_macro_py.f1_macro(x, y, classes)


def run_test():
    torch.random.manual_seed(3)
    x = torch.randint(0, 1000, (100000,))
    y = torch.randint(0, 1000, (100000,))
    test_f1_cpp(x, y, 1000)
    test_f1_py(x, y, 1000)
    x = x.cuda()
    y = y.cuda()
    test_f1_cpp(x, y, 1000)
    test_f1_py(x, y, 1000)


if __name__ == "__main__":
    run_test()
