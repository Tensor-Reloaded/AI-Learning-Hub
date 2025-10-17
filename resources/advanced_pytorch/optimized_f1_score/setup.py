import os

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension


def get_extra_compile_args():
    if os.name == "nt":
        return {
            "msvc": ["/std:c++20", "/O2", "/DNDEBUG",
                     "/arch:AVX2"
                     ],
        }
    else:
        return {
            "cxx": ["-std=c++20", "-O3", "-DNDEBUG",
                    "-mavx2"
                    ],
        }


if __name__ == "__main__":
    setup(
        name="optimized_f1_score",
        ext_modules=[
            CppExtension(
                "f1_macro_cpp",
                ["f1_macro_cpp.cpp"],
                extra_compile_args=get_extra_compile_args(),
                define_macros=[
                    # ("ARCH_DEFAULT", None)
                    ("AVX2", None)
                ],
            ),
        ],
        cmdclass={"build_ext": BuildExtension},
    )
