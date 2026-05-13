# noise_warmup_da summary rules

summary 至少包含：

- dataset、source domain、target domain、model、seed；
- requested device 和 actual device；
- 四个 arm 的 source/target accuracy、NLL、ECE；
- `noise_all_target_delta`、`noise_head_target_delta`、linear probe target delta；
- 是否支持“随机噪声 warmup 破坏 pretrained transferable features”的假设；
- 失败时的 traceback 摘要。
