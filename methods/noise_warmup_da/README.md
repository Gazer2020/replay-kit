# noise_warmup_da

`noise_warmup_da` 复用 `/root/noise-warmup/` 中的随机噪声 warmup 思路，作为 replay-kit 方法接入。当前协议关注一个更直接的问题：随机初始化模型和 ImageNet 预训练模型，在 OfficeHome 单域训练/测试时，训练前加入随机噪声 warmup 是否会改变最终测试表现。

## 实验 arm

- `random_init_train`：随机初始化 ResNet，在目标域 train split 上训练后测试。
- `random_init_noise_train`：随机初始化 ResNet，先做随机噪声 warmup，再训练和测试。
- `pretrained_train`：ImageNet 预训练 ResNet，直接训练和测试。
- `pretrained_noise_train`：ImageNet 预训练 ResNet，先做随机噪声 warmup，再训练和测试。

默认正式实验会在 OfficeHome 的 `Art`、`Clipart`、`Product`、`Real World` 四个域上分别做固定 seed 的分层 train/test split，跑 3 个 seed，并汇总 accuracy、NLL、ECE、average confidence 和 final train loss。

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

正式实验：OfficeHome 四域、ResNet50、3 seeds，完成 summary 和飞书通知后关机。

```bash
python -m replay_kit.runner launch --method noise_warmup_da --experiment officehome_resnet50_3seed
```

随机初始化收敛补充实验：只跑 `random_init_train` 和 `random_init_noise_train`，使用更高学习率、更长
max epoch，并按 train loss 目标或 plateau 停止。

```bash
python -m replay_kit.runner launch --method noise_warmup_da --experiment random_init_convergence
```

常用覆盖：

```bash
python -m replay_kit.runner launch --method noise_warmup_da --experiment default \
  noise_warmup_da.epochs=5 \
  noise_warmup_da.warmup_epochs=1 \
  noise_warmup_da.batch_size=32
```

## 产物

训练脚本只写当前 run 目录：

- `metrics.json`：逐域、逐 seed、逐 arm 的结果和 aggregate。
- `history.csv`：训练阶段、epoch、loss、best loss 和 stale epoch。
- `checkpoints/`：仅在 `checkpoint_policy.save=true` 且 `noise_warmup_da.save_checkpoints=true` 时保存。
