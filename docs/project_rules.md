# Project Rules

本仓库是轻量级深度学习复现工作仓库，不是通用训练框架，也不是完整 MLOps 系统。

## 原则

- 只面向单机单卡 PyTorch 复现实验。
- 统一运行入口、实验记录、summary 和通知流程。
- 不强制统一每个方法内部代码结构。
- 不提交数据、checkpoint、模型权重和实验输出。
- 不引入 Docker、Slurm、WandB、MLflow、DVC、多卡 DDP 或任务队列。
