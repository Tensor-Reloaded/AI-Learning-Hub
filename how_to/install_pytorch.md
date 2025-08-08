# Environment setup

PyTorch has release binaries for with support for CUDA on Linux and Windows, which require NVIDIA GPUs.
PyTorch also has support for ROCm (AMD GPUs) on Linux and Metal GPUs (M3-M4) on Mac.

The latest versions of PyTorch run on `python<=3.13`. 

1. Create a Python environment using conda. We recommend installing conda from [Miniforge](https://github.com/conda-forge/miniforge).

```bash
# Create the environment
conda create -n 313 -c conda-forge python=3.13
# activate the environment
conda activate 313
# Run this to use conda-forge as your highest priority channel (not needed if you installed conda from Miniforge)
conda config --add channels conda-forge
# Install numpy
conda install numpy
```

2. Install the stable PyTorch (2.7.1+) from [pytorch.org](https://pytorch.org/get-started/locally/) using pip.

* Example CPU: `pip install torch torchvision`
* Example CUDA 12.8: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`
* Example ROCm 6.3: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3`


3. Install `tensorboard`, `wabdb`, `pandas` and `matplotlib` using conda.
---

Tips:
* Linux and WSL-based projects are faster than on Windows. Mac with M3 and M4 is also a good alternative.
* Stay up-to-date with the cuda versions.
* Stay up-to-date with the python and PyTorch versions.

Advanced tips:
* Building from source and using the C++ API for serving models optimizes the memory footprint and startup time.
