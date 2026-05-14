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

基础配置 `notify.real_send=false`，即使存在 webhook 也只会写
`notification_payload.json`，不会真实发送。方法 debug、smoke test、可跑性测试和其他不完整实验应保持这个默认值。

完整实验默认使用飞书真实通知，实验配置应显式设置：

```yaml
notify:
  enabled: true
  real_send: true
  max_text_chars: 2000
  use_env_proxy: false
```

专门做飞书联通测试时也可以临时设置 `notify.real_send=true`。除此之外，不完整实验不真实发送飞书。

如果设置了 `REPLAY_KIT_NOTIFY_DRY_RUN=true`，即使 `notify.real_send=true` 且存在 webhook，
也只会写 `notification_payload.json`，不会真实发送。

飞书发送默认不读取 `http_proxy` / `https_proxy` 等环境代理，避免服务器上的本地代理端口失效时阻断通知。
只有明确需要让飞书也走环境代理时，才设置 `notify.use_env_proxy=true`。

## 长消息

飞书正文发送紧凑结果摘要、结论和 `summary.md` / `train.log` 路径，不发送完整 summary 表格。
`notify.max_text_chars` 控制正文长度预算；超过预算时会截短并提示查看 summary/log。

## Dry-run

`notify.real_send=false`、未设置 `FEISHU_WEBHOOK`、或设置了 `REPLAY_KIT_NOTIFY_DRY_RUN=true` 时，
runner 自动 dry-run，并写入：

```text
notification_payload.json
```

这使 debug、smoke test 和方法可跑性测试不依赖真实通知，也不会误发群通知。

## 通知时机

- 实验完成。
- 实验失败。
