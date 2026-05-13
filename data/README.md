# data

Git 中只保留数据入口和说明文件，真实数据通过软链接或本地绝对路径提供。

推荐做法是让 `data/` 顶层直接对应各个数据集：

```bash
ln -s /mnt/datasets/OfficeHome data/OfficeHome
ln -s /mnt/datasets/CUB200-2011 data/CUB200-2011
```

不要提交数据文件、压缩包或预处理后的大文件。
