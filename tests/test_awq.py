import torch

from bitbite.awq import (
    AWQ_ORDER,
    AWQ_REVERSE_ORDER,
    awq_linear_reference,
    dequantize_awq,
    pack_awq_int4,
    unpack_awq_int4,
)


def test_orders_are_inverse():
    assert [AWQ_ORDER[AWQ_REVERSE_ORDER[j]] for j in range(8)] == list(range(8))


def test_pack_known_value():
    # 第 j 列放数字 j：nibble i 存第 AWQ_ORDER[i] 列，从高到低读出来就是 7531 6420
    values = torch.arange(8, dtype=torch.uint8).reshape(1, 8)
    assert pack_awq_int4(values).item() == 0x75316420


def test_pack_unpack_roundtrip():
    torch.manual_seed(0)
    values = torch.randint(0, 16, (64, 256), dtype=torch.uint8)
    packed = pack_awq_int4(values)
    assert packed.shape == (64, 32) and packed.dtype == torch.int32
    assert torch.equal(unpack_awq_int4(packed), values)


def test_dequantize_and_linear():
    torch.manual_seed(0)
    k, n, g = 256, 64, 128
    q = torch.randint(0, 16, (k, n), dtype=torch.uint8)
    z = torch.randint(0, 16, (k // g, n), dtype=torch.uint8)
    s = torch.rand(k // g, n, dtype=torch.float32) * 0.1

    w = dequantize_awq(pack_awq_int4(q), pack_awq_int4(z), s, g)
    expected = (q.float() - z.float().repeat_interleave(g, 0)) * s.repeat_interleave(g, 0)
    assert torch.allclose(w, expected)

    x = torch.randn(3, k)
    assert torch.allclose(awq_linear_reference(x, pack_awq_int4(q), pack_awq_int4(z), s, g), x @ expected, atol=1e-4)
