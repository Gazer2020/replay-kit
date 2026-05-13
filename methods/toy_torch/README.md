# toy_torch

`toy_torch` 是仓库自带的 smoke test 方法，用一个极小的 PyTorch MLP 拟合合成回归数据。它的目的不是复现论文结果，而是验证 runner、配置、metadata、日志、summary 和飞书通知链路。

## 设备选择

配置中的 `device` 支持：

- `auto`：优先 `cuda`，其次 `mps`，最后 `cpu`。
- `cuda`：强制使用 CUDA，不可用时失败。
- `mps`：强制使用 Apple MPS，不可用时失败。
- `cpu`：强制使用 CPU。

## 运行

```bash
python -m replay_kit.runner launch --method toy_torch --experiment debug --wait --timeout 120
```

故意失败路径：

```bash
python -m replay_kit.runner launch --method toy_torch --experiment fail --wait --timeout 120
```

## 产物

训练脚本会在 run 目录写入 `metrics.json`，包含最终 loss、最佳 loss、实际设备和是否达到 target loss。
