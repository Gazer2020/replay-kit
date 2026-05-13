# Agent Guide

本仓库服务于轻量级深度学习复现实验。Agent 接手任务时，先读本文件，再读 `docs/` 和目标方法目录下的 `README.md`、`AGENTS.md`。

## 工作原则

- 不把数据、checkpoint、模型权重或 `outputs/runs/` 提交进 Git。
- checkpoint 属于具体实验产物，只能写入当前 run 目录下的 `checkpoints/`。
- 方法代码可以保持各自风格，但实验入口、metadata、日志和 summary 必须遵守统一约定。
- 正式实验只通过 `screen` runner 启动。
- 修改已有方法前，先确认对应任务文件和配置文件。
- 失败实验也要保留日志、metadata、summary 和通知 payload。
- 飞书 webhook 只能通过 `FEISHU_WEBHOOK` 提供；默认 `notify.real_send=false`，debug、smoke 和方法可跑性测试只写 dry-run payload。
- 只有正式实验或专门飞书联通测试可以设置 `notify.real_send=true` 真实发送。
- 每次执行 `git commit` 前必须先调用 `neat-freak`，完成文档/任务状态同步检查后再提交。

## 新增方法流程

1. 在 `methods/{method_name}/` 新建方法代码、`README.md`、`AGENTS.md`、`summarize.md`。
2. 在 `configs/methods/{method_name}/` 新建 `default.yaml` 和至少一个 `debug.yaml`。
3. 在 `data/README.md` 说明数据来源、目录结构和软链接方式。
4. 用 debug 配置跑一次：

   ```bash
   python -m replay_kit.runner launch --method {method_name} --experiment debug --wait --timeout 120
   ```

5. 确认 run 目录包含 `metadata.json`、`config.yaml`、`train.log`、`metrics.json`、`summary.md`。
6. 更新任务文件和方法 README。

## 排查优先级

1. 查看 `metadata.json` 的 `status`、`error_message` 和环境信息。
2. 查看 `train.log` 末尾 traceback。
3. 查看 `config.yaml` 是否为预期最终配置。
4. 确认数据软链接和设备选择。
