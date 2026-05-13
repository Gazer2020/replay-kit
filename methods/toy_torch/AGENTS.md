# toy_torch Agent Guide

`toy_torch` 只用于验证仓库基础设施。

## 入口

- 训练入口：`methods/toy_torch/train.py`
- 默认配置：`configs/methods/toy_torch/default.yaml`
- 调试配置：`configs/methods/toy_torch/debug.yaml`
- 失败配置：`configs/methods/toy_torch/fail.yaml`

## 成功标准

- `metadata.status` 为 `finished`。
- `train.log` 中有每个 epoch 的 loss。
- `metrics.json` 包含 `device`、`final_loss`、`best_loss`、`reached_target`。
- `summary.md` 正确展示结果和日志尾部。
- 没有 `FEISHU_WEBHOOK` 时，`notification_payload.json` 的 `dry_run` 为 `true`。

## 失败优先检查

1. PyTorch 是否安装。
2. 强制设备是否可用，例如本机无 MPS 时不要设置 `device=mps`。
3. `train.log` 尾部 traceback。
