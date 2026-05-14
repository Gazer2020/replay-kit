# data

Git 中只保留数据入口和说明文件，真实数据通过软链接或本地绝对路径提供。

推荐做法是让 `data/` 顶层直接对应各个数据集：

```bash
ln -s /mnt/datasets/OfficeHome data/OfficeHome
ln -s /mnt/datasets/CUB200-2011 data/CUB200-2011
ln -s /mnt/datasets/SAMPLE_dataset_public data/SAMPLE_dataset_public
```

不要提交数据文件、压缩包或预处理后的大文件。

## SAMPLE SAR

SAMPLE public 数据集来自 `https://github.com/benjaminlewis-afrl/SAMPLE_dataset_public`，
当前 SAR source-only 实验读取：

```text
data/SAMPLE_dataset_public/png_images/{decibel,qpm}/
  synth/<class>/*.png
  real/<class>/*.png
```

本机数据软链接为：

```bash
ln -s /root/autodl-tmp/noise-warmup-data/SAMPLE_dataset_public data/SAMPLE_dataset_public
```

SAMPLE SAR source-only 配置使用 `decibel/`；DSAN 5-seed 配置分别覆盖 `decibel/` 和 `qpm/`。
