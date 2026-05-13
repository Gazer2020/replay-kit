# Feishu Rules

## 配置

飞书 webhook 只能通过环境变量提供：

```bash
export FEISHU_WEBHOOK="https://open.feishu.cn/open-apis/bot/v2/hook/..."
```

也可以写在仓库根目录 `.env`，runner 在发送通知前会自动加载：

```bash
FEISHU_WEBHOOK="https://open.feishu.cn/open-apis/bot/v2/hook/..."
```

不要把 webhook 写进 Git。

默认配置 `notify.real_send=false`，即使存在 webhook 也只会写
`notification_payload.json`，不会真实发送。方法 debug、smoke test、可跑性测试应保持这个默认值。

只有正式实验或专门做飞书联通测试时，才设置：

```yaml
notify:
  enabled: true
  real_send: true
```

如果设置了 `REPLAY_KIT_NOTIFY_DRY_RUN=true`，即使 `notify.real_send=true` 且存在 webhook，
也只会写 `notification_payload.json`，不会真实发送。

## Dry-run

未设置 `FEISHU_WEBHOOK` 时，runner 自动 dry-run，并写入：

```text
notification_payload.json
```

这使本地测试不依赖真实通知。

## 通知时机

- 实验完成。
- 实验失败。
