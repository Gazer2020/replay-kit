# Agent Workflow

## 接手任务

1. 阅读根目录 `AGENTS.md`。
2. 阅读任务文件。
3. 阅读方法目录的 `README.md` 和 `AGENTS.md`。
4. 检查配置和数据说明。
5. 先运行 debug 配置。
6. 根据日志和 metadata 修复问题。
7. 运行完整实验配置：使用 `screen` runner，并开启飞书真实通知。
8. 生成并检查 summary。
9. 更新任务状态。
10. 如需 `git commit`，先调用 `neat-freak` 完成同步检查。

## 运行边界

- 完整实验默认使用 `screen` runner 和飞书真实通知。
- 完整实验如需完成后关机，启动前必须主动询问用户并获得明确确认。
- debug、smoke、方法可跑性测试和其他不完整实验不真实发送飞书，也不关机。

## 失败排查

1. `metadata.json`
2. `train.log`
3. `config.yaml`
4. 数据软链接
5. 设备和依赖版本
