# Experiment Rules

## 命名

实验输出目录为：

```text
outputs/runs/{method_name}/{experiment_name}/{run_id}/
```

`run_id` 由日期时间、短 commit 和进程后缀组成。

## 必须产物

- `metadata.json`
- `config.yaml`
- `train.log`
- `summary.md`
- `notification_payload.json`

方法训练脚本应尽量输出 `metrics.json`。

如果方法需要保存 checkpoint，只能写入当前 run 目录的 `checkpoints/` 子目录，不使用仓库根目录的独立 checkpoint 文件夹。

`metadata.json` 会记录 `checkpoint_dir`；只有 `checkpoint_policy.save=true` 时，`checkpoint_path` 才指向该目录。
summary、通知、关机等后处理失败时，runner 会追加 `postprocess_errors`，不覆盖训练本身的 `status`。

## 运行

完整实验只通过 `screen` runner 启动，并使用飞书真实通知：

```bash
python -m replay_kit.runner launch --method noise_warmup_da --experiment sample_sar_resnet50_sourceonly
```

本地测试、debug、smoke 和方法可跑性测试可追加 `--wait --timeout 120`，但应保持
`notify.real_send=false` 且不设置关机。

## 通知

所有实验都会写 `notification_payload.json`。默认 `notify.real_send=false`，只做 dry-run；
完整实验必须在实验配置中显式设置 `notify.real_send=true`，专门飞书联通测试也可开启真实发送。
debug、smoke、方法可跑性测试和其他不完整实验不得开启飞书真实通知。

## 关机

关机不是完整实验的默认动作。完整实验如需完成后关机，启动前必须主动询问用户并获得明确确认；
确认后才可设置 `system.shutdown_on_finish=true`。debug、smoke、方法可跑性测试和其他不完整实验不关机。
