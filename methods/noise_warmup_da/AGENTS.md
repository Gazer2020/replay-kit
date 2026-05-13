# noise_warmup_da Agent Guide

## 入口

- 训练入口：`methods/noise_warmup_da/train.py`
- 默认配置：`configs/methods/noise_warmup_da/default.yaml`
- 调试配置：`configs/methods/noise_warmup_da/debug.yaml`
- 正式配置：`configs/methods/noise_warmup_da/officehome_resnet50_3seed.yaml`
- 随机初始化收敛补充配置：`configs/methods/noise_warmup_da/random_init_convergence.yaml`
- 数据说明：`data/README.md`

## 成功标准

- `metadata.status` 为 `finished`。
- `train.log` 中出现四个域、三个 seed、四个 arm 的训练和评估日志。
- `metrics.json` 包含 `device`、`domains`、`seeds`、`results`、`aggregate`；收敛实验还应包含 `stopped_epoch`、`best_train_loss`、`converged`、`convergence_reason`。
- `history.csv` 存在。
- checkpoint 不应保存，除非配置显式同时打开 `checkpoint_policy.save` 和 `noise_warmup_da.save_checkpoints`。

## 排查优先级

1. `data/OfficeHome` 是否为有效软链接。
2. OfficeHome 四个 domain 目录是否齐全。
3. `torchvision` 预训练权重是否可用；debug 配置不会下载权重。
4. CUDA 显存不足时先减小 `batch_size`、`image_size` 或 epoch 数。
5. 正式配置会在 runner 完成 summary 和飞书通知后执行 `shutdown -h now`。
