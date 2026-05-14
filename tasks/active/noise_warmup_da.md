# noise_warmup_da

## 目标

把 `/root/noise-warmup/` 的随机噪声 warmup 实验重写为 OfficeHome 四域内随机初始化 vs ImageNet 预训练、noise warmup vs no warmup 的对照，统一 runner、metadata、日志、summary 和飞书通知。

## 状态

- [x] 新增 method 目录、训练入口和方法说明。
- [x] 新增 default/debug 配置。
- [x] 更新数据 README，并将真实数据通过软链接接到 `data/OfficeHome`。
- [x] 跑通 debug 配置。
- [x] 检查 run 目录产物。
- [x] 正式 `officehome_resnet50_3seed` 配置跑完。
- [x] 跑随机初始化收敛补充实验 `random_init_convergence`。
- [x] 接入 SAMPLE public SAR `png_images/decibel/{synth,real}`，新增 padding transform 和 source-only 评估。
- [x] SAMPLE SAR source-only 配置 1 epoch dry-run 跑通。
- [x] 正式 `sample_sar_resnet50_sourceonly` 配置跑完。
- [x] 将运行约定更新为：完整实验默认 screen+飞书；关机需启动前主动确认；不完整实验不飞书、不关机。
- [x] 将 summary 自定义逻辑解耦到 `methods/noise_warmup_da/summary.py`，公共 `summary.py` 只保留 hook/兜底。
- [x] 新增 SAMPLE SAR DSAN 5-seed 配置和 DSAN 风格训练 arm。
- [x] SAMPLE SAR DSAN 1 epoch dry-run 跑通。
- [x] 正式 `sample_sar_dsan_resnet50_5seed` 配置跑完。
- [x] 新增 SAMPLE SAR QPM DSAN 5-seed 配置，并跑通 1 epoch dry-run。
- [x] 正式 `sample_sar_qpm_dsan_resnet50_5seed` 配置跑完。

## 备注

历史参考位于 `/root/noise-warmup/`；当前协议覆盖 OfficeHome 四域、四个训练 arm、三个 seed。

debug run：`outputs/runs/noise_warmup_da/debug/20260513_190251_9fae732_3231`，状态 `finished`，通知为 dry-run。

正式 run：`outputs/runs/noise_warmup_da/officehome_resnet50_3seed/20260513_173208_9f25ac5_1bec`，状态 `finished`。

补充实验目标：确认随机初始化较差是否来自 20 epoch 欠拟合；只跑
`random_init_train` / `random_init_noise_train`，使用更高 lr 和更长 max epoch，按 train loss
target 或 plateau 停止。

补充实验 run：`outputs/runs/noise_warmup_da/random_init_convergence/20260513_190428_e00b089_335e`，状态
`finished`。首次飞书发送因 shell 中 `http_proxy/https_proxy=http://127.0.0.1:7897` 且代理拒绝连接失败；
已绕过代理补发成功，并修复 runner，避免未来通知失败阻断后处理。

SAMPLE SAR 数据来自 `https://github.com/benjaminlewis-afrl/SAMPLE_dataset_public/tree/master/png_images`。
本机软链接：`data/SAMPLE_dataset_public -> /root/autodl-tmp/noise-warmup-data/SAMPLE_dataset_public`。
当前正式配置使用 `decibel` 版本，`synth` 为模拟 SAR 源域，`real` 为真实 SAR 目标域；
原图为 128x128，实验不 resize，居中 padding 到 224x224。该配置只跑预训练 ResNet50 的
`pretrained_noise_train` / `pretrained_train`，评估 `synth->synth` held-out split 与
`synth->real` source-only 预测。

SAMPLE SAR source-only dry-run：`outputs/runs/noise_warmup_da/sample_sar_resnet50_sourceonly/20260514_155944_2561c69_df4`，
状态 `finished`，通知为 dry-run，覆盖 3 seeds、2 arms、2 eval domains。

SAMPLE SAR 正式 run 已通过 screen 启动：
`outputs/runs/noise_warmup_da/sample_sar_resnet50_sourceonly/20260514_160241_2561c69_f93`。
状态 `finished`，飞书真实通知已发送。后续完整实验配置保留 `notify.real_send=true`，但不默认关机；
如需 `system.shutdown_on_finish=true`，必须在实验启动前主动询问并获得用户确认。

SAMPLE SAR DSAN 5-seed 配置：
`configs/methods/noise_warmup_da/sample_sar_dsan_resnet50_5seed.yaml`。该实验比较
`pretrained_noise_dsan_train` 与 `pretrained_dsan_train`，使用 real 作为无标签目标域参与
DSAN 风格 label-aware MMD 训练；为减小随机性，使用 seeds `[7, 13, 21, 42, 84]`、
`deterministic=true`、`random_horizontal_flip=false`、`num_workers=0`。

SAMPLE SAR DSAN dry-run：
`outputs/runs/noise_warmup_da/sample_sar_dsan_resnet50_5seed/20260514_162915_2561c69_d21`，
状态 `finished`，通知为 dry-run。

SAMPLE SAR DSAN 正式 run 已通过 screen 启动：
`outputs/runs/noise_warmup_da/sample_sar_dsan_resnet50_5seed/20260514_162947_2561c69_dc5`。
状态 `finished`，飞书真实通知已发送，配置保留 `notify.real_send=true`，不关机。5-seed 结果：
`pretrained_noise_dsan_train` 在 `synth->real` 上 accuracy `0.4776 +/- 0.0903`；
`pretrained_dsan_train` 为 `0.5178 +/- 0.0616`。DSAN 相比 source-only 明显提高真实域表现，
但在该 DSAN 设置下 noise warmup 平均低于 no-warmup。

SAMPLE SAR QPM DSAN 配置：
`configs/methods/noise_warmup_da/sample_sar_qpm_dsan_resnet50_5seed.yaml`。沿用 decibel
DSAN 5-seed 设置，但 `sample_variant=qpm`，不关机。

SAMPLE SAR QPM DSAN dry-run：
`outputs/runs/noise_warmup_da/sample_sar_qpm_dsan_resnet50_5seed/20260514_164853_2561c69_1421`，
状态 `finished`，通知为 dry-run。

SAMPLE SAR QPM DSAN 正式 run 已通过 screen 启动：
`outputs/runs/noise_warmup_da/sample_sar_qpm_dsan_resnet50_5seed/20260514_164927_2561c69_14b4`。
状态 `finished`，飞书真实通知已发送，配置保留 `notify.real_send=true`，
`system.shutdown_on_finish=false`。5-seed 结果：`pretrained_noise_dsan_train` 在
`synth->real` 上 accuracy `0.6848 +/- 0.0811`；`pretrained_dsan_train` 为
`0.6565 +/- 0.0350`。QPM 版本上 noise warmup 在 DSAN 设置下平均高于 no-warmup，
真实域提升约 +0.0283。
