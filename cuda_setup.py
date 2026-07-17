"""Preload the CUDA 13 + cuDNN 9 DLLs that onnxruntime-gpu's CUDA EP needs.

This module is WINDOWS-ONLY and a no-op on other platforms (see setup()).

Windows lacks an automatic search path for DLLs that cuDNN loads by name, so
onnxruntime_providers_cuda.dll imports cuBLAS/cuDNN by name and cuDNN in turn
loads its own sub-DLLs (cudnn_graph/ops/cnn/engines_*) programmatically by
name. None of those folders are on PATH (we never touch the system PATH), so
the OS loader can't find them at runtime and ORT silently falls back to CPU
with "Cannot load symbol cudnnCreate".

This module stages the required DLLs next to onnxruntime's own provider DLL
and pre-loads them with ctypes.CDLL so they are resident before piper/ORT
create the CUDA session. Import it once, before importing piper.

Override the toolkit locations via the CUDA_ROOT / CUDNN_ROOT env vars if your
install lives elsewhere.
"""

import ctypes
import glob
import os
import shutil

# --- locate system CUDA 13 toolkit ---------------------------------------------
CUDA_BASE = os.environ.get(
    "CUDA_ROOT",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
)
def _find_cuda_binx64() -> str | None:
    # newest installed v13.x
    candidates = sorted(glob.glob(os.path.join(CUDA_BASE, "v13.*")))
    for c in reversed(candidates):
        p = os.path.join(c, "bin", "x64")
        if os.path.isfile(os.path.join(p, "cublasLt64_13.dll")):
            return p
    return None

# --- locate system cuDNN 9 (CUDA 13 build) --------------------------------------
CUDNN_BASE = os.environ.get(
    "CUDNN_ROOT",
    r"C:\Program Files\NVIDIA\CUDNN",
)
def _find_cudnn_binx64() -> str | None:
    # cuDNN 9.x layout: <base>\v9.*\bin\13.*\x64\cudnn64_9.dll
    for cudnn_ver in sorted(glob.glob(os.path.join(CUDNN_BASE, "v9.*")), reverse=True):
        for cuda_ver in sorted(glob.glob(os.path.join(cudnn_ver, "bin", "13.*")), reverse=True):
            p = os.path.join(cuda_ver, "x64")
            if os.path.isfile(os.path.join(p, "cudnn64_9.dll")):
                return p
    return None

# prefixes of DLLs that must be staged + preloaded
_STAGE_PREFIXES = (
    "cudnn", "cudart", "cublas", "cufft", "cufftw", "cusparse", "cusolver",
    "curand", "npp", "nvrtc", "nvJitLink", "nvblas", "nvfatbin", "nvjpeg",
)


def _ensure_staged(capi: str) -> None:
    """Copy CUDA/cuDNN DLLs into onnxruntime/capi if the key ones are missing."""
    need = {
        "cudart64_13.dll": _find_cuda_binx64(),
        "cublasLt64_13.dll": _find_cuda_binx64(),
        "cudnn64_9.dll": _find_cudnn_binx64(),
    }
    sources = []
    for name, src in need.items():
        if not os.path.isfile(os.path.join(capi, name)):
            if src is None:
                raise FileNotFoundError(
                    f"{name} not staged and not found under "
                    f"CUDA_ROOT={CUDA_BASE!r} / CUDNN_ROOT={CUDNN_BASE!r}. "
                    "Install CUDA 13 + cuDNN 9 (CUDA 13 build) or set CUDA_ROOT/CUDNN_ROOT."
                )
            sources.append(src)
    if not sources:
        return  # already staged
    # stage everything relevant from each source folder
    for src in dict.fromkeys(sources):
        for f in os.listdir(src):
            if f.lower().endswith(".dll") and f.lower().startswith(_STAGE_PREFIXES):
                shutil.copy2(os.path.join(src, f), os.path.join(capi, f))


def _preload(capi: str) -> None:
    for f in sorted(os.listdir(capi)):
        if f.lower().endswith(".dll") and f.lower().startswith(_STAGE_PREFIXES):
            try:
                ctypes.CDLL(os.path.join(capi, f))
            except OSError:
                pass  # ignore optional libs (e.g. nvjpeg) that may have extra deps


def setup() -> str | None:
    # The DLL-staging + preloading dance below is Windows-only. On Linux/macOS
    # the CUDA/cuDNN shared libraries are found by the dynamic linker via
    # LD_LIBRARY_PATH / the default linker paths, so nothing to do here.
    # (Windows lacks an equivalent search mechanism for by-name loads, which
    # is exactly the gap this module closes.)
    if os.name != "nt":
        return None

    import onnxruntime  # local import keeps this importable in isolation
    capi = os.path.join(os.path.dirname(onnxruntime.__file__), "capi")
    if not os.path.isdir(capi):
        raise RuntimeError(f"onnxruntime capi dir not found: {capi}")
    _ensure_staged(capi)
    _preload(capi)
    return capi


setup()
