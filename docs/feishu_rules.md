# Feishu Rules

## 配置

飞书 webhook 只能通过环境变量提供：

```bash
export FEISHU_WEBHOOK="https://open.feishu.cn/open-apis/bot/v2/hook/..."
```

不要把 webhook 写进 Git。

## Dry-run

未设置 `FEISHU_WEBHOOK` 时，runner 自动 dry-run，并写入：

```text
notification_payload.json
```

这使本地测试不依赖真实通知。

## 通知时机

- 实验完成。
- 实验失败。
