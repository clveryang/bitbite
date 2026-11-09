# bitbite

> 比特一口一口啃，项目一点一点做（bit by bit）。

从零手写 INT4 量化矩阵乘法 kernel，接入 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)，推理 [Qwen/Qwen3-4B-AWQ](https://huggingface.co/Qwen/Qwen3-4B-AWQ)。
参考项目：[NanjinLin/Pewee](https://github.com/NanjinLin/Pewee)。

原则：**正确性先于速度，loader 先于 kernel，GEMV 先于 GEMM**。

## 路线

- [ ] **阶段 0** 环境：原版 nano-vLLM 在目标 GPU 上跑通 Qwen3-1.7B（重点确认 flash-attn 能装）
- [ ] **阶段 1** 纯 PyTorch 参考实现：`AWQLinear` 在 forward 里反量化再算，`enforce_eager`，模型输出正确文本
  - loader 处理 q/k/v → qkv、gate/up → gate_up 的合并（qweight/qzeros 沿 `N/8` 拼，scales 沿 `N` 拼）
- [x] **阶段 2** AWQ 解包 / 反量化参考函数 + 单元测试（`bitbite/awq.py`）
- [ ] **阶段 3** decode：W4A16 GEMV kernel（g128、FP16 scale、16B 向量化读取、lop3 反量化），目标跑满显存带宽
  - prefill 先用"整块反量化 + cuBLAS"
- [ ] **阶段 4** prefill：自己的 W4A16 GEMM（cp.async 多级流水、ldmatrix、mma.sync）
- [ ] **阶段 5** CUDA graph / torch.compile（fake 注册，算子内无 host 同步）
- [ ] **阶段 6** W4A8：激活按 token 动态量化到 INT8，INT8 mma

## Qwen3-4B-AWQ 要点

`bits=4, group_size=128, zero_point=true, version=gemm`，`tie_word_embeddings=true`（没有单独的 lm_head，复用 embedding）。

量化范围（已读 safetensors 头部确认）：每层 7 个 Linear（q/k/v/o + gate/up/down）都是 INT4，约 3.63B 参数，FFN 占约 74%。

**dtype 注意：** config 写的是 `torch_dtype=float16`，scales 也是 FP16，但 `embed_tokens` 和所有 norm（input/post_attention_layernorm、q_norm、k_norm、norm）实际是 **BF16**，loader 要逐张量转成统一 dtype。

| 张量 | 形状 | 说明 |
|---|---|---|
| qweight | int32 `[K, N/8]` | 沿 N 打包，nibble 顺序 `[0,2,4,6,1,3,5,7]` |
| qzeros | int32 `[K/128, N/8]` | 同上 |
| scales | fp16 `[K/128, N]` | |

反量化：`w = (q - zero) * scale`。

kernel 要覆盖的形状（hidden 2560，32 头 × 128，8 个 KV 头）：

| 层 | N | K |
|---|---|---|
| qkv_proj | 6144 | 2560 |
| o_proj | 2560 | 4096 |
| gate_up_proj | 19456 | 2560 |
| down_proj | 2560 | 9728 |

## 对比基准

RTX 5060 8GB 放不下 FP16 的 4B 模型，所以速度对比：
- vLLM 跑同一 AWQ 模型（awq_marlin）——行业水平
- 本项目"反量化 + cuBLAS"——起点

精度对比：与 transformers 输出的 logits 比较，加小规模 wikitext perplexity。

## 测试

```bash
pip install -e ".[dev]"
pytest
```
