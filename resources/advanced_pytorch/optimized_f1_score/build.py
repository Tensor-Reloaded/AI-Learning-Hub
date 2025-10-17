import os
import subprocess
import filelock


def build():
    p_dir = os.path.dirname(__file__)
    lock_path = os.path.join(p_dir, ".lock")

    try:
        from optimized_f1_score.f1_macro_cpp import f1_macro
    except ImportError:
        with filelock.FileLock(lock_path):
            print("Building Optimized F1 Score")
            try_build(p_dir)


def try_build(p_dir):
    try:
        setup_cmd = f'python setup.py build_ext --inplace'
        if os.name == "nt":
            setup_cmd = prepare_windows_build() + setup_cmd
        subprocess.run(setup_cmd, shell=True, cwd=p_dir)
    except Exception as e:
        print(f"Error during build. Please check logs. See {e}")
    print("Done building")


def prepare_windows_build():
    activate_msvc = "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat"
    if "ACTIVATE_MSVC" in os.environ:
        activate_msvc = os.environ["ACTIVATE_MSVC"]
    if not os.path.isfile(activate_msvc):
        raise RuntimeError(f"MSCV must be activated, but {activate_msvc} file was not found. "
                           f"Ensure that Microsoft Visual Studio is installed and set the ACTIVATE_MSVC "
                           f"to point to vcvars64.bat")
    return f"\"{activate_msvc}\" && set DISTUTILS_USE_SDK=1 && "
