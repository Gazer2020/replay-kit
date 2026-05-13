# noise_warmup_da

`noise_warmup_da` 复用 `/root/noise-warmup/` 中的 OfficeHome 随机噪声 warmup 思路，作为 replay-kit 方法接入。核心问题是：在已有 ImageNet/source-domain 预训练表征后，继续用随机噪声图片和随机标签 warmup，是否会破坏可迁移特征。

## 实验 arm

- `source_pretrained`：在 source domain 监督训练后直接评估。
- `pretrained_noise_all`：从 `source_pretrained` 出发，随机噪声 warmup 更新全部参数。
- `pretrained_noise_head`：从 `source_pretrained` 出发，冻结 backbone，只更新分类头。
- `random_init_noise_before_source`：随机初始化先噪声 warmup，再做 source 监督训练。

每个 arm 会记录 source/target 的 accuracy、NLL、ECE、avg confidence，并做 frozen-backbone linear probe。

## 数据

默认读取：

```text
data/OfficeHome/
  Art/<class>/*
  Clipart/<class>/*
  Product/<class>/*
  Real World/<class>/*
```

本机已按此约定把 `data/OfficeHome` 软链接到 `/root/autodl-tmp/noise-warmup-data/OfficeHome`。debug 配置使用 `FakeData`，不需要真实数据。

## 运行

```bash
python -m replay_kit.runner launch --method noise_warmup_da --experiment debug --wait --timeout 120
```

OfficeHome Art -> Clipart：

```bash
python -m replay_kit.runner launch --method noise_warmup_da --experiment default
```

常用覆盖：

```bash
python -m replay_kit.runner launch --method noise_warmup_da --experiment default \
  noise_warmup_da.model=resnet50 \
  noise_warmup_da.source_epochs=5 \
  noise_warmup_da.warmup_epochs=1 \
  noise_warmup_da.linear_probe_epochs=3
```

## 产物

训练脚本只写当前 run 目录：

- `metrics.json`：四个 arm 的指标和关键 delta。
- `history.csv`：训练阶段、epoch、loss。
- `checkpoints/`：仅在 `checkpoint_policy.save=true` 且 `noise_warmup_da.save_checkpoints=true` 时保存。
