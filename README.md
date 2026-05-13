# replay-kit

`replay-kit` 是一个面向 PyTorch 单机单卡复现实验的轻量工作仓库。它不尝试统一所有模型代码，也不引入 Slurm、Docker、WandB、MLflow 或 DVC；它统一的是实验入口、配置快照、日志、metadata、summary、飞书通知和 Agent 协作规则。

Agent 入口和完整协作指南统一使用 `AGENTS.md`。

## 快速开始

本地 macOS / MPS 测试：

```bash
conda create -n replay-kit-py312 python=3.12 -y
conda activate replay-kit-py312
pip install -e .
pip install -r requirements-torch-macos.txt
python -m replay_kit.runner launch --method toy_torch --experiment debug --wait --timeout 120
```

远程 Linux / CUDA 12.8：

```bash
conda create -n replay-kit-py312 python=3.12 -y
conda activate replay-kit-py312
pip install -e .
pip install -r requirements-torch-cu128.txt
python -m replay_kit.runner launch --method toy_torch --experiment debug --wait --timeout 120
```

若远程 GPU 较旧，不适合 CUDA 12.8 wheel，请参考 PyTorch 官方安装页选择 CUDA 12.6 或其他兼容 wheel：<https://pytorch.org/get-started/locally/>。

## 常用命令

```bash
python -m replay_kit.runner launch --method toy_torch --experiment debug
python -m replay_kit.runner launch --method toy_torch --experiment debug --wait --timeout 120
python -m replay_kit.runner status --run-dir outputs/runs/toy_torch/debug/<run_id>
python -m replay_kit.runner summarize --run-dir outputs/runs/toy_torch/debug/<run_id>
```

所有正式运行都通过 `screen` 启动。`--wait` 只是让当前终端轮询 metadata，方便本地 smoke test 和 CI 风格验证。

## 输出约定

每次实验输出到：

```text
outputs/runs/{method_name}/{experiment_name}/{run_id}/
```

最小产物：

```text
metadata.json
config.yaml
train.log
metrics.json
summary.md
notification_payload.json
```

失败实验也会写入 `metadata.json`、`train.log`、`summary.md` 和通知 payload。

## 仓库结构

```text
configs/              # 全局与方法级配置
methods/              # 各复现方法代码
src/replay_kit/       # 公共 runner、metadata、summary、notifier
tools/                # 公共工具说明
tasks/                # 小规模协作任务
data/                 # 数据入口和 README，真实数据不进 Git
outputs/              # 实验输出入口，runs 不进 Git
docs/                 # 项目规则和 Agent 工作流
```

checkpoint 不使用仓库根目录的独立文件夹；如需保存，放在对应 run 目录的 `checkpoints/` 子目录下。

## 飞书通知

远程服务器上设置 `FEISHU_WEBHOOK` 后，还需要实验配置显式设置 `notify.real_send=true` 才会真实发送通知。默认配置只写 dry-run `notification_payload.json`，用于 debug、smoke test 和方法可跑性测试。
