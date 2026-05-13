# noise_warmup_da

## 目标

把 `/root/noise-warmup/` 的随机噪声 warmup 域适应实验接入 replay-kit，统一 runner、metadata、日志、summary 和 dry-run 通知链路。

## 状态

- [x] 新增 method 目录、训练入口和方法说明。
- [x] 新增 default/debug 配置。
- [x] 更新数据 README，并将真实数据通过软链接接到 `data/OfficeHome`。
- [x] 跑通 debug 配置。
- [x] 检查 run 目录产物。

## 备注

历史参考位于 `/root/noise-warmup/`；当前接入优先覆盖 OfficeHome A->C 四 arm 对照。

debug run：`outputs/runs/noise_warmup_da/debug/20260513_164749_053c9ce_d1c`，状态 `finished`。
