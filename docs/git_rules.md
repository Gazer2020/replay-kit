# Git Rules

## Branch

推荐分支：

- `main`
- `repro/{method-name}`
- `debug/{method-name}`

## 禁止提交

- `.env`
- `outputs/runs/`
- 真实数据文件
- checkpoint、权重、`.pt`、`.pth`、`.ckpt`、`.safetensors`

checkpoint 应作为实验产物保存在 `outputs/runs/.../{run_id}/checkpoints/`，随整个 run 目录被 Git 忽略。

## 提交前

- 必须先调用 `neat-freak`，检查 `AGENTS.md`、`README.md`、`docs/`、任务文件和方法文档是否需要同步。
- `neat-freak` 检查完成后，再执行 `git status`、测试或静态检查，并提交代码。

## 合并标准

- 方法目录存在。
- 方法 README 和 AGENTS 文档存在。
- 配置文件存在。
- 至少有一次实验 summary。
- 数据说明清楚。
- 不包含大文件。
