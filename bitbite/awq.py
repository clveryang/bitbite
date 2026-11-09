"""AWQ (version="gemm") 权重格式的纯 PyTorch 参考实现。

这里的函数是整个项目的"标准答案"：后面每个 CUDA kernel 的输出都要和它比对。

AutoAWQ GEMM 格式（以 Linear 的 in_features=K, out_features=N 为例）：
    qweight: int32 [K, N // 8]          沿 N 方向把 8 个 INT4 打包进一个 int32
    qzeros:  int32 [K // g, N // 8]     zero point 也按同样方式打包
    scales:  fp16  [K // g, N]
反量化：w[k, n] = (q[k, n] - zero[k // g, n]) * scale[k // g, n]

打包顺序不是 0..7，而是交错的 AWQ_ORDER：
    int32 的第 i 个 nibble（bit 4i..4i+3）存的是第 AWQ_ORDER[i] 列。
来源：AutoAWQ awq/utils/packing_utils.py
"""

import torch

AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

_PACK = 8  # 32 bit / 4 bit


def unpack_awq_int4(packed: torch.Tensor) -> torch.Tensor:
    """int32 [R, C // 8] -> uint8 [R, C]，已恢复成正常列顺序，取值 0..15。"""
    if packed.dtype != torch.int32:
        raise TypeError(f"packed 必须是 int32，实际是 {packed.dtype}")
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    # 先按 nibble 位置拆开：nibbles[..., i] 是第 i 个 nibble
    nibbles = (packed.unsqueeze(-1) >> shifts) & 0xF
    # 再把交错顺序换回来：第 j 列在第 AWQ_REVERSE_ORDER[j] 个 nibble
    nibbles = nibbles[..., AWQ_REVERSE_ORDER]
    return nibbles.reshape(packed.shape[0], -1).to(torch.uint8)


def pack_awq_int4(values: torch.Tensor) -> torch.Tensor:
    """uint8/int [R, C]（取值 0..15）-> int32 [R, C // 8]，按 AWQ 交错顺序打包。"""
    rows, cols = values.shape
    if cols % _PACK != 0:
        raise ValueError(f"列数 {cols} 必须是 8 的倍数")
    v = values.to(torch.int64)
    if v.min() < 0 or v.max() > 15:
        raise ValueError("INT4 取值必须在 0..15")
    v = v.reshape(rows, cols // _PACK, _PACK)[..., AWQ_ORDER]
    shifts = torch.arange(0, 32, 4, device=values.device, dtype=torch.int64)
    packed = (v << shifts).sum(dim=-1)
    # 大于 2^31 的值按补码折回 int32
    packed = torch.where(packed >= 2**31, packed - 2**32, packed)
    return packed.to(torch.int32)


def dequantize_awq(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """返回 [K, N] 的浮点权重（dtype 同 scales）。注意是 [in, out]，不是 F.linear 的 [out, in]。"""
    q = unpack_awq_int4(qweight).to(torch.float32)  # [K, N]
    z = unpack_awq_int4(qzeros).to(torch.float32)  # [K // g, N]
    s = scales.to(torch.float32)  # [K // g, N]
    k = q.shape[0]
    if k % group_size != 0 or z.shape[0] != k // group_size or tuple(s.shape) != (k // group_size, q.shape[1]):
        raise ValueError(
            f"形状不匹配：qweight->{tuple(q.shape)} qzeros->{tuple(z.shape)} "
            f"scales->{tuple(s.shape)} group_size={group_size}"
        )
    z = z.repeat_interleave(group_size, dim=0)
    s = s.repeat_interleave(group_size, dim=0)
    return ((q - z) * s).to(scales.dtype)


def awq_linear_reference(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """y = x @ W + bias，x 形状 [..., K]，返回 [..., N]。慢，但保证正确。"""
    w = dequantize_awq(qweight, qzeros, scales, group_size).to(x.dtype)
    y = x @ w
    if bias is not None:
        y = y + bias
    return y
