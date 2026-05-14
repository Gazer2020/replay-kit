# noise_warmup_da Agent Guide

## 入口

- 训练入口：`methods/noise_warmup_da/train.py`
- 默认配置：`configs/methods/noise_warmup_da/default.yaml`
- 调试配置：`configs/methods/noise_warmup_da/debug.yaml`
- 正式配置：`configs/methods/noise_warmup_da/officehome_resnet50_3seed.yaml`
- 随机初始化收敛补充配置：`configs/methods/noise_warmup_da/random_init_convergence.yaml`
- SAMPLE SAR source-only 配置：`configs/methods/noise_warmup_da/sample_sar_resnet50_sourceonly.yaml`
- SAMPLE SAR DSAN 5-seed 配置：`configs/methods/noise_warmup_da/sample_sar_dsan_resnet50_5seed.yaml`
- SAMPLE SAR QPM DSAN 5-seed 配置：`configs/methods/noise_warmup_da/sample_sar_qpm_dsan_resnet50_5seed.yaml`
- 方法级 summary hook：`methods/noise_warmup_da/summary.py`
- 数据说明：`data/README.md`

## 成功标准

- `metadata.status` 为 `finished`。
- 主实验 `officehome_resnet50_3seed` 的 `train.log` 中出现四个域、三个 seed、四个 arm 的训练和评估日志；收敛补充实验 `random_init_convergence` 中出现四个域、三个 seed、两个随机初始化 arm。
- `metrics.json` 包含 `device`、`domains`、`seeds`、`results`、`aggregate`；收敛实验还应包含 `stopped_epoch`、`best_train_loss`、`converged`、`convergence_reason`。
- SAMPLE SAR 实验应只跑 `pretrained_noise_train` 和 `pretrained_train`，训练域为 `synth`，评估域为 `synth->synth` 和 `synth->real`，transform 为 `pad`，不得 resize 原图。
- SAMPLE SAR DSAN 实验只跑 `pretrained_noise_dsan_train` 和 `pretrained_dsan_train`，使用 5 seeds、`deterministic=true`、`random_horizontal_flip=false`、`num_workers=0` 来降低随机性。
- `history.csv` 存在。
- checkpoint 不应保存，除非配置显式同时打开 `checkpoint_policy.save` 和 `noise_warmup_da.save_checkpoints`。

## 排查优先级

1. `data/OfficeHome` 是否为有效软链接。
2. OfficeHome 四个 domain 目录是否齐全。
3. SAMPLE SAR 实验检查 `data/SAMPLE_dataset_public/png_images/{decibel,qpm}/{synth,real}` 是否齐全。
4. `torchvision` 预训练权重是否可用；debug 配置不会下载权重。
5. CUDA 显存不足时先减小 `batch_size`、`image_size` 或 epoch 数。
6. 完整实验默认使用 screen runner 和飞书真实通知；完成后关机只有在用户启动前明确确认时才开启。
