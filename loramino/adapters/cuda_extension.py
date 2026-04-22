import os
import shutil
import subprocess
from pathlib import Path
import torch

_EXTENSION = None
_EXTENSION_ERROR = None


def _compiler_major_version(compiler: str) -> int | None:
    try:
        result = subprocess.run(
            [compiler, "-dumpfullversion", "-dumpversion"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    version_text = result.stdout.strip().split(".")[0]
    if not version_text.isdigit():
        return None
    return int(version_text)


def _candidate_cxx_compilers() -> list[str]:
    candidates: list[str] = []

    for env_name in ("LORAMINO_CXX", "CXX"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(value)

    candidates.extend(
        [
            "g++-14",
            "g++-13",
            "g++-12",
            "g++-11",
            "g++-10",
            "g++-9",
            "g++",
            "c++",
        ]
    )

    deduped: list[str] = []
    seen = set()
    for candidate in candidates:
        compiler_path = shutil.which(candidate) or candidate
        if compiler_path in seen:
            continue
        seen.add(compiler_path)
        deduped.append(compiler_path)
    return deduped


def _matching_cc(cxx_compiler: str) -> str | None:
    compiler_name = Path(cxx_compiler).name
    sibling_name = compiler_name.replace("g++", "gcc") if "g++" in compiler_name else None
    if sibling_name is not None:
        sibling_path = str(Path(cxx_compiler).with_name(sibling_name))
        if Path(sibling_path).exists():
            return sibling_path
        resolved = shutil.which(sibling_name)
        if resolved:
            return resolved

    for fallback in ("gcc", "cc"):
        resolved = shutil.which(fallback)
        if resolved:
            return resolved
    return None


def _ensure_modern_compiler() -> str | None:
    for compiler in _candidate_cxx_compilers():
        major_version = _compiler_major_version(compiler)
        if major_version is None or major_version < 9:
            continue

        resolved_cxx = shutil.which(compiler) or compiler
        os.environ.setdefault("CXX", resolved_cxx)
        os.environ.setdefault("CUDAHOSTCXX", resolved_cxx)

        cc_compiler = _matching_cc(resolved_cxx)
        if cc_compiler is not None:
            os.environ.setdefault("CC", cc_compiler)
        return resolved_cxx
    return None


def load_grouped_lora_cuda_extension():
    global _EXTENSION, _EXTENSION_ERROR

    if _EXTENSION is not None:
        return _EXTENSION
    if _EXTENSION_ERROR is not None:
        return None
    if not torch.cuda.is_available():
        _EXTENSION_ERROR = "CUDA is not available."
        return None

    cxx_compiler = _ensure_modern_compiler()
    if cxx_compiler is None:
        _EXTENSION_ERROR = (
            "Could not find a GCC/G++ 9+ host compiler. Load a newer GCC module or set "
            "CXX/CC before building the extension."
        )
        return None

    try:
        from torch.utils.cpp_extension import load
    except Exception as exc:  # pragma: no cover - depends on local torch install
        _EXTENSION_ERROR = str(exc)
        return None

    source_dir = Path(__file__).resolve().parent / "csrc"
    build_dir = Path(__file__).resolve().parents[2] / ".torch_extensions" / "loramino_grouped_lora_cuda"
    build_dir.mkdir(parents=True, exist_ok=True)
    try:
        _EXTENSION = load(
            name="loramino_grouped_lora_cuda",
            sources=[
                str(source_dir / "grouped_lora.cpp"),
            ],
            build_directory=str(build_dir),
            extra_cflags=["-O3"],
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - depends on local CUDA toolchain
        _EXTENSION_ERROR = str(exc)
        return None

    return _EXTENSION


def grouped_lora_cuda_error() -> str | None:
    return _EXTENSION_ERROR
