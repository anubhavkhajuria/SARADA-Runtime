# SARADA: Software Architecture for RISCV Driven AI Workloads

SARADA is distributed as a Docker image for using the RICE compiler stack on RISC-V AI workloads.

This repository contains usage files only.

## What You Get

- Prebuilt Docker image with:
  - LLVM/MLIR tools
  - torch-mlir
  - RICE pass plugin
  - `torch_rice` runtime package
  - RISC-V cross compilation and QEMU support
- Docker commands to run lowering and validation workflows

## RICE Capabilities

SARADA runtime includes the main RICE functionality:

- Torch to RICE conversion passes:
  - `convert-torch-to-rice`
  - `convert-rice-to-linalg`
- Automatic lowering pipeline:
  - Torch IR
  - RICE dialect IR
  - Linalg IR
  - LLVM IR
  - RISC-V assembly
- Automatic fusion support:
  - Linalg elementwise fusion path
  - Fusion on/off control with `enable_linalg_fusion`
- Vectorization support:
  - `vectorization_mode=\"aggressive\"` for RVV-oriented codegen
  - `vectorization_mode=\"none\"` for scalar-only codegen
- Built-in QEMU execution flow:
  - Compile to RISC-V object/executable
  - Execute with QEMU user-mode backend
  - Compare outputs against eager PyTorch
- Transformer-oriented operator coverage:
  - attention-like patterns
  - GELU, layernorm-related patterns
  - 2D and higher-rank/batched matmul shapes
- Artifact generation:
  - intermediate MLIR stages
  - generated `model.s` assembly
  - runtime inputs/outputs for reproducibility

## Image Distribution

Release asset name:

- `sarada-linux-arm64.tar.gz`

## Linux and macOS Setup

1. Download image tar from Releases

```bash
curl -L -o sarada-linux-arm64.tar.gz \
  https://github.com/anubhavkhajuria/SARADA/releases/download/v1.0.1/sarada-linux-arm64.tar.gz

curl -L -o sarada-linux-arm64.sha256 \
  https://github.com/anubhavkhajuria/SARADA/releases/download/v1.0.1/sarada-linux-arm64.sha256

shasum -a 256 -c sarada-linux-arm64.sha256
```

2. Load image into Docker

```bash
gunzip -c sarada-linux-arm64.tar.gz | docker load
```

3. Run interactive shell

```bash
docker run --rm -it sarada:latest bash
```

Notes:

- Apple Silicon macOS and ARM Linux run this image natively.
- Intel macOS/Linux can run with emulation:

```bash
docker run --rm -it --platform linux/arm64 sarada:latest bash
```

## Verify RICE Passes Are Available

```bash
docker run --rm sarada:latest \
  bash -lc 'torch-mlir-opt --help | grep -E "convert-torch-to-rice|convert-rice-to-linalg"'
```

## Usage Examples

### 1) Matmul lowering with vectorization

```bash
docker run --rm -it sarada:latest python - <<'PY'
import torch
from torch_rice import RICERISCVBackend

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(32, 24))
    def forward(self, x):
        return torch.matmul(x, self.w)

m = M().eval()
x = torch.randn(16, 32)

for mode in ["aggressive", "none"]:
    b = RICERISCVBackend(
        use_rice=True,
        execution_mode="riscv",
        vectorization_mode=mode,
        enable_linalg_fusion=True,
    )
    c = b.compile(m, x, func_name=f"matmul_{mode}")
    asm = c.get_riscv_assembly() or ""
    print(mode, "asm_path=", c._asm_path, "has_vsetvli=", "vsetvli" in asm)
PY
```

### 2) Conv2D lowering with and without vectorization

```bash
docker run --rm -it sarada:latest python - <<'PY'
import torch
from torch_rice import RICERISCVBackend

m = torch.nn.Sequential(
    torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
    torch.nn.GELU(),
).eval()

x = torch.randn(1, 3, 32, 32)

for mode in ["aggressive", "none"]:
    b = RICERISCVBackend(use_rice=True, execution_mode="riscv", vectorization_mode=mode, enable_linalg_fusion=True)
    c = b.compile(m, x, func_name=f"conv2d_{mode}")
    asm = c.get_riscv_assembly() or ""
    print(mode, "asm_path=", c._asm_path, "has_vsetvli=", "vsetvli" in asm)
PY
```

### 3) 4D batched matmul with and without vectorization

```bash
docker run --rm -it sarada:latest python - <<'PY'
import torch
from torch_rice import RICERISCVBackend

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(1, 4, 8, 16))
    def forward(self, x):
        return torch.matmul(x, self.w)

m = M().eval()
x = torch.randn(1, 4, 16, 8)

for mode in ["aggressive", "none"]:
    b = RICERISCVBackend(use_rice=True, execution_mode="riscv", vectorization_mode=mode, enable_linalg_fusion=True)
    c = b.compile(m, x, func_name=f"matmul4d_{mode}")
    asm = c.get_riscv_assembly() or ""
    print(mode, "asm_path=", c._asm_path, "has_vsetvli=", "vsetvli" in asm)
PY
```

### 4) Transformer operators with and without fusion, and with and without vectorization

```bash
docker run --rm -it sarada:latest python - <<'PY'
import torch
import torch.nn as nn
from torch_rice import RICERISCVBackend

class TinyTransformerOps(nn.Module):
    def __init__(self, d_model=32, nhead=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, d_model))
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x

m = TinyTransformerOps().eval()
x = torch.randn(1, 16, 32)

for fusion in [False, True]:
    for mode in ["aggressive", "none"]:
        b = RICERISCVBackend(
            use_rice=True,
            execution_mode="riscv",
            vectorization_mode=mode,
            enable_linalg_fusion=fusion,
            require_linalg_fusion=False,
        )
        c = b.compile(m, x, func_name=f"xfm_fusion_{fusion}_{mode}")
        asm = c.get_riscv_assembly() or ""
        report = getattr(c, "_lowering_report", {}) or {}
        print(
            "fusion=", fusion,
            "mode=", mode,
            "asm_path=", c._asm_path,
            "has_vsetvli=", "vsetvli" in asm,
            "linalg_fusion_applied=", report.get("linalg_fusion_applied", False),
        )
PY
```

## Docker Compose Helper

You can use `docker-compose.yml` from this repo:

```bash
docker compose run --rm sarada
```
