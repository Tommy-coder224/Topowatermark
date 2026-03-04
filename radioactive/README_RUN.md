# TopoRad 运行说明

## 为什么「加载完成后程序停住」？

有两类现象，都不是 bug：

### 1. CLIPTextModel LOAD REPORT / UNEXPECTED key

- **现象**：加载本地 `stable-diffusion-v1-4` 时出现  
  `text_model.embeddings.position_ids | UNEXPECTED`
- **原因**：本地 text_encoder 与 diffusers 默认期望的键略有差异（例如不同任务/架构），属正常。
- **结论**：可忽略，不影响训练与推理。Notes 里也写了：*can be ignored when loading from different task/architecture*。

### 2. 加载 100% 后长时间无输出

- **现象**：`Loading pipeline components...: 100%` 和 safety checker 提示之后，程序长时间无新输出。
- **原因**：**第一次**在 GPU 上做扩散推理时，PyTorch 会编译 CUDA kernel、分配显存，约 30–90 秒，看起来像卡死。
- **做法**：  
  - 使用带预热的入口：`python -u run_test_with_warmup.py`（加载后先跑一次 warmup，**每步会打印 [Warmup] step i/N**，不会假死）。  
  - 或执行 `.\kill_and_run_test.ps1`（会先结束已有 Python 进程再运行测试，避免残留进程占用）。

## 结束卡住的进程后再跑

PowerShell 中可先结束再运行：

```powershell
# 结束当前终端里卡住的 Python（或关闭该终端窗口）
# 然后新开终端：
conda activate watermark
cd "C:\Users\tom20\Desktop\科研\moring commiting\watermark"
python -u run_test_with_warmup.py
```

需要先杀掉所有 Python 再跑时（注意会关掉其他 Python 进程）：

```powershell
Get-Process -Name python -ErrorAction SilentlyContinue | Stop-Process -Force
python -u run_test_with_warmup.py
```

或直接运行项目里的脚本（内部会先 kill 再跑）：

```powershell
conda activate watermark
.\kill_and_run_test.ps1
```

## 推荐运行方式（确保不卡住）

```powershell
conda activate watermark
cd "C:\Users\tom20\Desktop\科研\moring commiting\watermark"
python -u run_test_with_warmup.py
```

- `-u`：无缓冲输出，日志立即显示。  
- 模型路径：脚本内为 `model/stable-diffusion-v1-4`（相对项目根）。  
- 若出现 `[Warmup] step 1/4` … `step 4/4`，说明在正常预热，等几十秒即可。
