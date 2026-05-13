# outputs

实验输出入口。`outputs/runs/` 默认不进 Git。

每次运行应生成：

```text
outputs/runs/{method_name}/{experiment_name}/{run_id}/
  metadata.json
  config.yaml
  train.log
  metrics.json
  summary.md
  notification_payload.json
  checkpoints/        # 可选，只在方法需要保存 checkpoint 时创建
```

checkpoint 属于具体实验的产物，应保存在对应 run 目录下，不再使用仓库根目录的独立 `checkpoints/`。
