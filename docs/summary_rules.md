# Summary Rules

每次实验都必须生成 `summary.md`。

## 必须包含

- 基本信息：方法、实验名、run id、branch、commit。
- 复现目标。
- 实验设置：数据、模型、关键参数、设备。
- 结果：核心指标。
- 结论：是否达到预期。
- 问题与备注。

失败实验必须说明错误摘要，并保留日志尾部。

## 方法级扩展

公共 `src/replay_kit/summary.py` 只负责读取 run 产物、拼装统一章节和兜底格式。
新增方法需要自定义指标表格或结论时，在方法目录新增 `summary.py`，实现：

```python
def format_metrics(metrics: dict) -> str: ...
def conclusion_lines(metrics: dict, status: str) -> list[str]: ...
```

不要为了某个方法的指标或 arm 名称在公共 summary 里写特例。
