# configs

配置按三层组织：

- `base.yaml`：全局默认值。
- `methods/{method}/default.yaml`：方法默认值。
- `methods/{method}/{experiment}.yaml`：具体实验配置。

runner 会按以上顺序合并配置，并在每个 run 目录保存最终 `config.yaml`。
