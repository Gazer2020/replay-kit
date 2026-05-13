# methods

每个复现方法单独放在 `methods/{method_name}/`。

方法目录至少应包含：

- `README.md`
- `AGENTS.md`
- 训练入口
- `summarize.md`
- 可选 `reports/report_template.md`

方法内部代码可以保持各自风格，但必须能通过 runner 配置中的 `command.entrypoint` 启动。
