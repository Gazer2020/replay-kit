# Data And Artifact Rules

Git 只保存目录结构和 README。

## 数据

真实数据通过软链接或本地路径映射：

```bash
ln -s /mnt/datasets/my_dataset data/my_method
```

## Checkpoint

默认不保存 checkpoint。如果方法必须保存，应放在当前实验 run 目录下：

```text
outputs/runs/{method_name}/{experiment_name}/{run_id}/checkpoints/
```

建议只保留 `latest` 或 `best`。`outputs/runs/` 和常见权重后缀已被 `.gitignore` 忽略。
