#!/usr/bin/env python3
"""Quick environment check: torch sees the GPU and a real Triton kernel runs."""
import torch
import triton
import triton.language as tl

print("torch     :", torch.__version__)
print("torch cuda :", torch.version.cuda)
print("triton    :", triton.__version__)
print("cuda avail :", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not available"
print("device    :", torch.cuda.get_device_name(0))
print("capability :", torch.cuda.get_device_capability(0))


@triton.jit
def _add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(o_ptr + off, tl.load(x_ptr + off, mask=m) + tl.load(y_ptr + off, mask=m), mask=m)


n = 4096
x = torch.randn(n, device="cuda")
y = torch.randn(n, device="cuda")
o = torch.empty_like(x)
_add_kernel[(triton.cdiv(n, 1024),)](x, y, o, n, BLOCK=1024)
torch.cuda.synchronize()
ok = torch.allclose(o, x + y)
print("triton kernel correct:", bool(ok))
assert ok, "triton kernel result mismatch"
print("\nENV OK")
