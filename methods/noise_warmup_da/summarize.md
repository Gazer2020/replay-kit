# noise_warmup_da summary rules

summary 至少包含：

- OfficeHome 四个域、三个 seed、四个 arm 的覆盖情况；
- dataset、domains、seeds、model；
- requested device 和 actual device；
- 每个 domain/arm 的 accuracy、NLL、ECE、final train loss 的 mean/std；
- 随机初始化与预训练、noise warmup 与 no warmup 的对比；
- 失败时的 traceback 摘要。
