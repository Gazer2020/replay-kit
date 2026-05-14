# noise_warmup_da summary rules

summary 至少包含：

- OfficeHome 四个域、三个 seed、四个 arm 的覆盖情况；
- dataset、domains、seeds、model；
- requested device 和 actual device；
- 每个 domain/arm 的 accuracy、NLL、ECE、final train loss、best train loss、stopped epoch 的 mean/std；
- 随机初始化与预训练、noise warmup 与 no warmup 的对比；
- 收敛补充实验应说明有多少 seed/domain/arm 达到 target train loss 或 plateau。
- SAMPLE SAR source-only 实验应说明 `sample_variant`、`source_domain`、`target_domain`、
  `transform_mode=pad`，以及 `synth->synth` 和 `synth->real` 两个评估域的覆盖情况。
- SAMPLE SAR DSAN 实验应说明 `adaptation=['dsan']`、`dsan_lambda`、5 seeds、
  `deterministic=true`，并报告 `da_loss`。
- 失败时的 traceback 摘要。
