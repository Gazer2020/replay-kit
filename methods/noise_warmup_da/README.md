# noise_warmup_da

`noise_warmup_da` 复用 `/root/noise-warmup/` 中的随机噪声 warmup 思路，作为 replay-kit 方法接入。当前协议关注一个更直接的问题：随机初始化模型和 ImageNet 预训练模型，在 OfficeHome 单域训练/测试时，训练前加入随机噪声 warmup 是否会改变最终测试表现。

## 实验 arm

- `random_init_train`：随机初始化 ResNet，在目标域 train split 上训练后测试。
- `random_init_noise_train`：随机初始化 ResNet，先做随机噪声 warmup，再训练和测试。
- `pretrained_train`：ImageNet 预训练 ResNet，直接训练和测试。
- `pretrained_noise_train`：ImageNet 预训练 ResNet，先做随机噪声 warmup，再训练和测试。
- `pretrained_dsan_train`：ImageNet 预训练 ResNet，在源域监督训练的同时用目标域无标签样本做 DSAN 风格 label-aware MMD 对齐。
- `pretrained_noise_dsan_train`：先做随机噪声 warmup，再做上述 DSAN 风格训练。

默认正式实验会在 OfficeHome 的 `Art`、`Clipart`、`Product`、`Real World` 四个域上分别做固定 seed 的分层 train/test split，跑 3 个 seed，并汇总 accuracy、NLL、ECE、average confidence 和 final train loss。

## 数据

OfficeHome 默认读取：

```text
data/OfficeHome/
  Art/<class>/*
  Clipart/<class>/*
  Product/<class>/*
  Real World/<class>/*
```

本机已按此约定把 `data/OfficeHome` 软链接到 `/root/autodl-tmp/noise-warmup-data/OfficeHome`。debug 配置使用 `FakeData`，不需要真实数据。

SAMPLE SAR source-only 实验读取：

```text
data/SAMPLE_dataset_public/png_images/decibel/
  synth/<class>/*.png
  real/<class>/*.png
```

其中 `synth` 是模拟 SAR 源域，`real` 是真实 SAR 目标域。该配置使用 ImageNet 预训练
ResNet50，只比较：

- `pretrained_noise_train`：先 noise warmup，再在 `synth` 的 train split 上监督训练。
- `pretrained_train`：不做 warmup，直接在 `synth` 的 train split 上监督训练。

评估同时覆盖 `synth->synth` held-out split 和 `synth->real` source-only 目标域预测。输入图像
保持原始 128x128 内容，不 resize，居中 padding 到 224x224 后送入 ResNet。

## 运行

```bash
python -m replay_kit.runner launch --method noise_warmup_da --experiment debug --wait --timeout 120
```

完整实验：OfficeHome 四域、ResNet50、3 seeds，使用 screen runner 和飞书真实通知。
完成后关机需在启动前单独询问并确认。

```bash
python -m replay_kit.runner launch --method noise_warmup_da --experiment officehome_resnet50_3seed
```

随机初始化收敛补充实验：只跑 `random_init_train` 和 `random_init_noise_train`，使用更高学习率、更长
max epoch，并按 train loss 目标或 plateau 停止。

```bash
python -m replay_kit.runner launch --method noise_warmup_da --experiment random_init_convergence
```

SAMPLE SAR source-only 完整实验：使用 screen runner 和飞书真实通知。
完成后关机需在启动前单独询问并确认。

```bash
python -m replay_kit.runner launch --method noise_warmup_da --experiment sample_sar_resnet50_sourceonly
```

SAMPLE SAR DSAN 5-seed 完整实验：比较 `pretrained_noise_dsan_train` 和
`pretrained_dsan_train`，使用 5 个 seed，关闭随机水平翻转，DataLoader 单进程，并启用
deterministic 设置以尽量减小随机性。该实验使用 `real` 作为无标签目标域参与训练。

```bash
python -m replay_kit.runner launch --method noise_warmup_da --experiment sample_sar_dsan_resnet50_5seed
```

同一设置在 SAMPLE `qpm` 版本上的完整实验：

```bash
python -m replay_kit.runner launch --method noise_warmup_da --experiment sample_sar_qpm_dsan_resnet50_5seed
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
