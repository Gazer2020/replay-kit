# noise_warmup_da

## 目标

把 `/root/noise-warmup/` 的随机噪声 warmup 实验重写为 OfficeHome 四域内随机初始化 vs ImageNet 预训练、noise warmup vs no warmup 的对照，统一 runner、metadata、日志、summary、飞书通知和完成后关机。

## 状态

- [x] 新增 method 目录、训练入口和方法说明。
- [x] 新增 default/debug 配置。
- [x] 更新数据 README，并将真实数据通过软链接接到 `data/OfficeHome`。
- [x] 跑通 debug 配置。
- [x] 检查 run 目录产物。
- [x] 正式 `officehome_resnet50_3seed` 配置跑完。
- [x] 跑随机初始化收敛补充实验 `random_init_convergence`。

## 备注

历史参考位于 `/root/noise-warmup/`；当前协议覆盖 OfficeHome 四域、四个训练 arm、三个 seed。

latest debug run：`outputs/runs/noise_warmup_da/debug/20260513_190251_9fae732_3231`，状态 `finished`，通知为 dry-run。

正式 run：`outputs/runs/noise_warmup_da/officehome_resnet50_3seed/20260513_173208_9f25ac5_1bec`，状态 `finished`。

补充实验目标：确认随机初始化较差是否来自 20 epoch 欠拟合；只跑
`random_init_train` / `random_init_noise_train`，使用更高 lr 和更长 max epoch，按 train loss
target 或 plateau 停止。

补充实验 run：`outputs/runs/noise_warmup_da/random_init_convergence/20260513_190428_e00b089_335e`，状态
`finished`。首次飞书发送因 shell 中 `http_proxy/https_proxy=http://127.0.0.1:7897` 且代理拒绝连接失败；
已绕过代理补发成功，并修复 runner，避免未来通知失败阻断自动关机请求。
