"""
Pipeline 预热：首次 GPU 推理时 CUDA 会编译内核、分配显存，耗时 30–90 秒。
显式做一次短步数推理，并用 callback 每步打印，避免用户误以为程序卡死。
"""
from __future__ import annotations

import sys
import time
from typing import Any


def warmup_pipeline(pipe: Any, device: str = "cuda", steps: int = 4) -> float:
    """
    对 Stable Diffusion pipeline 做一次极短步数推理，触发 CUDA 预热。
    首次调用较慢（约 30–90s）。若支持 callback，每步会打印 [Warmup] step i/total，避免假死感。

    Returns:
        预热耗时（秒）
    """
    print("[Warmup] First GPU inference (CUDA compile + alloc). You will see step 1/%d ... %d/%d." % (steps, steps, steps), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()

    # 每步打印的 callback：兼容 (step, timestep) 或 (pipe, step, timestep, callback_kwargs)
    _total = [steps]

    def _on_step(*args):
        if len(args) >= 2:
            step = args[1] if len(args) >= 3 else args[0]
        else:
            step = 0
        print("  [Warmup] step %d/%d" % (step + 1, _total[0]), flush=True)
        sys.stdout.flush()
        if len(args) >= 4:
            return args[3]
        return None

    import torch
    t0 = time.time()
    with torch.no_grad():
        try:
            _ = pipe(
                "warmup",
                num_inference_steps=steps,
                guidance_scale=7.5,
                callback_on_step_end=_on_step,
                callback_on_step_end_tensor_inputs=[],
            )
        except TypeError:
            _ = pipe("warmup", num_inference_steps=steps, guidance_scale=7.5)
    dt = time.time() - t0
    print("[Warmup] Done in %.1fs. Next runs will be faster." % dt, flush=True)
    sys.stdout.flush()
    return dt
