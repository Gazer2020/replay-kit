# Remote Server Setup

默认远程环境：

- Linux
- bash
- screen
- Python 3.12
- PyTorch 2.8
- CUDA 12.8 兼容 GPU

## 安装

```bash
conda create -n replay-kit-py312 python=3.12 -y
conda activate replay-kit-py312
pip install -e .
pip install -r requirements-torch-cu128.txt
```

## 验证

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
PY
```

PyTorch 2.8 起 CUDA 12.8 wheel 对旧 GPU 架构有额外限制。旧卡优先按 PyTorch 官方安装页切换 CUDA 12.6 或其他兼容 wheel。
