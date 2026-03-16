# Contributing

## 开发前准备

推荐使用标准环境：

```bash
cd /Users/charles/Downloads/pdf2md
conda env create -f environment.yml
conda activate conda_rag
```

如果环境已存在：

```bash
conda env update -n conda_rag -f environment.yml --prune
conda activate conda_rag
```

## 本地开发常用命令

```bash
make lint
make test
make debug-ingest
```

## 安装提交钩子

```bash
conda activate conda_rag
pre-commit install
pre-commit run --all-files
```

## 提交前建议

至少确保下面两项通过：

```bash
ruff check .
pytest -q
```

## 分支与发布建议

- 日常开发走 feature branch
- 合并到 `main` 前确保 CI 通过
- 版本发布使用 `v*` tag，触发构建流程

