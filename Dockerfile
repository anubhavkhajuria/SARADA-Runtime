# syntax=docker/dockerfile:1.7
FROM python:3.11-slim-bookworm

ARG RICE_REPO=https://github.com/BrainSeek-Lab/torch-mlir-rice.git
ARG RICE_REF=main

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_MLIR_REF=integrates/llvm-20250128 \
    TORCH_MLIR_BIN=/usr/local/bin \
    MLIR_OPT_PATH=/usr/bin/mlir-opt-21 \
    MLIR_TRANSLATE_PATH=/usr/bin/mlir-translate-21 \
    LLC_PATH=/usr/bin/llc-21 \
    OPT_PATH=/usr/bin/opt-21 \
    PATH=/usr/local/bin:${PATH}

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    binutils-riscv64-linux-gnu \
    build-essential \
    ca-certificates \
    clang \
    cmake \
    curl \
    file \
    g++-riscv64-linux-gnu \
    gcc-riscv64-linux-gnu \
    git \
    gnupg \
    libcurl4-openssl-dev \
    libedit-dev \
    libzstd-dev \
    lld \
    ninja-build \
    pybind11-dev \
    pkg-config \
    qemu-user \
    qemu-user-static \
    zlib1g-dev \
 && curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
    | gpg --dearmor -o /usr/share/keyrings/llvm-archive-keyring.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/llvm-archive-keyring.gpg] http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-21 main" \
    > /etc/apt/sources.list.d/llvm.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    libmlir-21-dev \
    llvm-21 \
    llvm-21-dev \
    llvm-21-tools \
    mlir-21-tools \
 && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${RICE_REF}" "${RICE_REPO}" /workspace/torch-mlir-rice

WORKDIR /workspace/torch-mlir-rice

RUN python - <<'PY'
from pathlib import Path

p = Path("/workspace/torch-mlir-rice/torch_rice/backend.py")
text = p.read_text()
if "torch_mlir_opt_full" in text:
    print("backend patch already present")
else:
    old = """                # Step 3c: Keep RICE in the real compilation path, then lower residual torch ops.
                result = subprocess.run(
                    [
                        torch_mlir_opt,
                        "--convert-torch-to-rice",
                        "--convert-rice-to-linalg",
                        "--torch-backend-to-linalg-on-tensors-backend-pipeline",
                        torch_mlir_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=240,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"Torch->RICE->Linalg pipeline failed: {result.stderr}")
"""
    new = """                # Step 3c: Lower full Torch program with full torch-mlir toolchain.
                torch_mlir_opt_full = os.environ.get(
                    "TORCH_MLIR_OPT_FULL_PATH",
                    "/usr/local/lib/python3.11/site-packages/torch_mlir/_mlir_libs/torch-mlir-opt",
                )
                if not os.path.exists(torch_mlir_opt_full):
                    torch_mlir_opt_full = torch_mlir_opt
                result = subprocess.run(
                    [
                        torch_mlir_opt_full,
                        "--torch-backend-to-linalg-on-tensors-backend-pipeline",
                        torch_mlir_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=240,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"Torch backend->Linalg pipeline failed: {result.stderr}")
"""
    if old not in text:
        raise RuntimeError("Could not find expected Step 3c block in torch_rice/backend.py")
    p.write_text(text.replace(old, new, 1))
    print("applied backend Step 3c patch")
PY

RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install --retries 10 --timeout 120 --pre torch torchvision \
      --index-url https://download.pytorch.org/whl/nightly/cpu \
 && success=0; for i in 1 2 3 4 5; do \
      python -m pip install --retries 10 --timeout 120 --pre torch-mlir \
        -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels \
        --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
      && success=1 && break; \
      sleep 10; \
    done; [ "${success}" -eq 1 ] \
 && python -m pip install --retries 10 --timeout 120 numpy timm pytest opencv-python-headless nanobind \
 && python -m pip install -e . --no-deps \
 && chmod +x /workspace/torch-mlir-rice/scripts/build_torch_mlir_with_rice_plugin.sh \
 && /workspace/torch-mlir-rice/scripts/build_torch_mlir_with_rice_plugin.sh

RUN rm -rf /workspace/torch-mlir-rice/examples \
 && command -v torch-mlir-opt \
 && torch-mlir-opt --help | grep -q -- convert-torch-to-rice \
 && torch-mlir-opt --help | grep -q -- convert-rice-to-linalg \
 && (command -v mlir-opt-21 || command -v mlir-opt || command -v torch-mlir-opt) \
 && (command -v mlir-translate-21 || command -v mlir-translate || command -v torch-mlir-translate) \
 && (command -v llc-21 || command -v llc || command -v llc-14) \
 && (command -v opt-21 || command -v opt || command -v opt-14) \
 && command -v qemu-riscv64 \
 && python -c "import torch, torch_mlir, torchvision, timm, numpy, cv2, torch_rice" \
 && python - <<'PY'
import torch
from torch_rice import RICERISCVBackend

class _Smoke(torch.nn.Module):
    def forward(self, x):
        return x + 1

m = _Smoke().eval()
x = torch.randn(2, 4)
b = RICERISCVBackend(
    use_rice=True,
    execution_mode="riscv",
    vectorization_mode="none",
    enable_linalg_fusion=True,
)
c = b.compile(m, x, func_name="smoke_add")
asm = c.get_riscv_assembly() or ""
if not asm:
    raise RuntimeError("RICE smoke compile produced empty assembly")
print("RICE compiler smoke passed")
PY

CMD ["bash"]
