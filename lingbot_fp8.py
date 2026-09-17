"""
LingBot-World 2.0 / 1.3B causal-fast —— selective FP8 weight-only 量化

只量化 30 个 transformer block 里的:
    self_attn.q/k/v/o
    cross_attn.q/k/v/o
    ffn.0 / ffn.2

保留 BF16(不量化):
    blocks.*.cam_injector_layer1/2      <- 相机控制，LingBot 的核心差异点
    blocks.*.cam_scale_layer / cam_shift_layer
    time_embedding / time_projection    <- 对误差极敏感
    text_embedding
    patch_embedding (nn.Conv3d，quantize_ 本来就不命中)
    head

依据(2026-09-10 核对官方仓库):
    wan/configs/wan_i2v_1_3B.py : dim=1536 ffn_dim=8960 num_heads=12 num_layers=30
    wan/modules/model_fast.py   : CausalWanAttentionBlock 属性名如上

接入 wan/image2video.py —— 在下面这一行之后:

    self.model = self._configure_model(
        model=self.model,
        use_sp=use_sp,
        dit_fsdp=dit_fsdp,
        shard_fn=shard_fn,
        convert_model_dtype=convert_model_dtype).to(self.device)

插入:

    if os.getenv("LINGBOT_FP8", "0") == "1":
        from lingbot_fp8 import apply_selective_fp8
        apply_selective_fp8(self.model)

(把本文件放到 lingbot-world-v2 仓库根目录即可 import)

用法:
    LINGBOT_FP8=1 python generate.py ...

干跑(不装 torchao 也能看会命中哪些层):
    python -c "from lingbot_fp8 import report; import torch; \
    from transformers import AutoModel; report(AutoModel.from_pretrained(..., trust_remote_code=True))"
"""

import torch.nn as nn

# 命中的 fqn 片段。注意都带前后点，避免误伤名字符合前缀的其它模块。
_TARGETS = (".self_attn.", ".cross_attn.", ".ffn.")


def lingbot_fp8_filter(module, fqn):
    """torchao quantize_ 的 filter_fn: (nn.Module, fqn) -> bool"""
    if not isinstance(module, nn.Linear):
        return False
    if not fqn.startswith("blocks."):
        return False
    return any(t in fqn for t in _TARGETS)


def report(model, verbose=False):
    """统计会命中 / 会跳过的 Linear 参数量。不需要 torchao。"""
    hit_params = 0
    skip_params = 0
    hit_names = []
    skipped_in_block = []
    skipped_outside = []

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        n = sum(p.numel() for p in mod.parameters(recurse=False))
        if lingbot_fp8_filter(mod, name):
            hit_params += n
            hit_names.append(name)
        else:
            skip_params += n
            (skipped_in_block if name.startswith("blocks.")
             else skipped_outside).append(name)

    total = hit_params + skip_params
    gib = 1024.0 ** 3
    print(f"命中(待量化) : {hit_params / 1e6:8.1f} M 参数 "
          f"-> {hit_params * 2 / gib:.2f} GiB(BF16) -> {hit_params / gib:.2f} GiB(FP8)")
    print(f"跳过(留 BF16): {skip_params / 1e6:8.1f} M 参数 -> {skip_params * 2 / gib:.2f} GiB")
    print(f"Linear 合计  : {total / 1e6:8.1f} M 参数")
    print(f"预计净节省   : ~{hit_params / gib:.2f} GiB  (不含 per-channel scale，量级 <0.01 GiB)")
    if verbose:
        print("\n-- 跳过的 block 内层(应全是 cam_* / norm) --")
        for n in skipped_in_block:
            print("   ", n)
        print("\n-- 跳过的 block 外层 --")
        for n in skipped_outside:
            print("   ", n)
    return hit_params, skip_params


def apply_selective_fp8(model, weight_dtype=None, device=None, do_report=True):
    """原地量化。返回 (model, 节省字节数估计)。"""
    import torch
    from torchao.quantization import Float8WeightOnlyConfig, quantize_

    if weight_dtype is None:
        weight_dtype = torch.float8_e4m3fn

    if do_report:
        hit, _ = report(model)
    else:
        hit = sum(
            sum(p.numel() for p in m.parameters(recurse=False))
            for n, m in model.named_modules()
            if lingbot_fp8_filter(m, n)
        )

    quantize_(
        model,
        Float8WeightOnlyConfig(weight_dtype=weight_dtype),
        filter_fn=lingbot_fp8_filter,
        device=device,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[lingbot_fp8] 已量化 {hit / 1e6:.1f} M 参数为 FP8，"
          f"预计释放 ~{hit / 1024**3:.2f} GiB")
    return model, hit


if __name__ == "__main__":
    print(__doc__)
