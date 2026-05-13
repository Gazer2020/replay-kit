# noise_warmup_da

## 目标

把 `/root/noise-warmup/` 的随机噪声 warmup 实验重写为 OfficeHome 四域内随机初始化 vs ImageNet 预训练、noise warmup vs no warmup 的对照，统一 runner、metadata、日志、summary、飞书通知和完成后关机。

## 状态

- [x] 新增 method 目录、训练入口和方法说明。
- [x] 新增 default/debug 配置。
- [x] 更新数据 README，并将真实数据通过软链接接到 `data/OfficeHome`。
- [x] 跑通 debug 配置。
- [x] 检查 run 目录产物。
- [ ] 跑正式 `officehome_resnet50_3seed` 配置。

## 备注

历史参考位于 `/root/noise-warmup/`；当前协议覆盖 OfficeHome 四域、四个训练 arm、三个 seed。

latest debug run：`outputs/runs/noise_warmup_da/debug/20260513_172837_9127709_1a15`，状态 `finished`。
