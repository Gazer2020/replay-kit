# data

Git 中只保留数据入口和说明文件，真实数据通过软链接或本地绝对路径提供。

推荐做法：

```bash
ln -s /mnt/datasets/example data/example
```

不要提交数据文件、压缩包或预处理后的大文件。
