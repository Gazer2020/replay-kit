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

## 运行

正式实验只通过 `screen` runner 启动：

```bash
python -m replay_kit.runner launch --method toy_torch --experiment debug
```

本地测试可追加 `--wait --timeout 120`。

## 通知

所有实验都会写 `notification_payload.json`。默认 `notify.real_send=false`，只做 dry-run；
正式实验如需飞书通知，必须在实验配置中显式设置 `notify.real_send=true`。
