from functools import lru_cache

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=1024)
def _decompose_scale(scale: float):
    mantissa, exp = torch.frexp(torch.tensor(scale, dtype=torch.float64))
    M0 = int(torch.round(mantissa * (1 << 31)).item())
    exp = int(exp.item())
    if M0 == (1 << 31):
        M0 >>= 1
        exp += 1
    return M0, 31 - exp


@triton.jit
def _upward_kernel(x_ptr, y_ptr, n, M0, shift, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    half_bias = tl.cast(1, tl.int64) << (shift - 1)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.int64)
    y = ((x * M0 + half_bias) >> shift).to(tl.int32)
    tl.store(y_ptr + offs, y, mask=mask)


class Requantizer:
    def __init__(self, scale, BLOCK=1024):
        self.M0, self.shift = _decompose_scale(float(scale))
        self.BLOCK = BLOCK

    def __call__(self, x):
        y = torch.empty_like(x)
        n = x.numel()
        _upward_kernel[(triton.cdiv(n, self.BLOCK),)](
            x, y, n, self.M0, self.shift, BLOCK=self.BLOCK
        )
        return y


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"
    QMAX = 127

    M, K, N = 256, 512, 128
    X_fp32 = torch.randn(M, K, device=device) * 0.5
    W_fp32 = torch.randn(K, N, device=device) * 0.1

    input_scale = X_fp32.abs().max().item() / QMAX
    weight_scale = W_fp32.abs().max().item() / QMAX

    X_int8 = torch.round(X_fp32 / input_scale).clamp(-128, 127).to(torch.int8)
    W_int8 = torch.round(W_fp32 / weight_scale).clamp(-128, 127).to(torch.int8)

    Y_fp32 = X_fp32 @ W_fp32
    output_scale = Y_fp32.abs().max().item() / QMAX
    combined = (input_scale * weight_scale) / output_scale

    acc_int32 = (X_int8.float() @ W_int8.float()).to(torch.int32)

    req = Requantizer(combined)
    out_int32 = req(acc_int32)
    out_int8 = torch.clamp(out_int32, -128, 127).to(torch.int8)

    noise = out_int8.float() * output_scale - Y_fp32
    snr_db = 10 * torch.log10((Y_fp32 ** 2).sum() / (noise ** 2).sum())

    print(f"INT8 GEMM [{M}x{K} @ {K}x{N}]  SNR={snr_db:.2f} dB")
