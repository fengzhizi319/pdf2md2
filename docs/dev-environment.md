# 开发环境与 base 清理策略

本文档的目标不是“把 base 清得一干二净”，而是把风险最小化：

1. `pdf2md` 项目以后统一跑在 `conda_rag`
2. `base` 不再承担项目依赖
3. 只有确认不会影响其他项目时，才对 `base` 做卸载

## 当前机器上已观察到的问题

在本机 `base` 环境里，已经出现过以下情况：

- `pdf2md-rag`
- `marker-pdf`
- `chromadb`
- `sentence-transformers`
- `torch`
- `transformers`
- `numpy`
- `scipy`

同时还存在：

- `mlx-lm`
- `mlx-vlm`

其中 `mlx-lm` / `mlx-vlm` 对 `transformers` 的版本要求，与 `marker-pdf` 拉下来的版本并不完全一致。也就是说，`base` 已经不适合作为这个项目的长期运行环境。

## 推荐原则

### 最稳妥的做法

- 保留 `base` 作为 conda 自身管理环境
- 项目统一使用 `conda_rag`
- 以后不要再在 `base` 里执行 `pip install -e .` 或安装项目依赖

这已经能解决 90% 的漂移问题。

### 只有在确认安全时，才清理 `base`

如果 `base` 里这些包只服务于 `pdf2md`，可以按“从最专用到最通用”的顺序清理。

## 清理前先做快照

```bash
conda activate base
python -m pip list --format=freeze > ~/Desktop/base-pip-freeze-before-pdf2md-clean.txt
conda list -n base > ~/Desktop/base-conda-list-before-pdf2md-clean.txt
```

## 第 1 层：先卸载明确属于 pdf2md 的包

这一步通常风险最小：

```bash
conda activate base
python -m pip uninstall -y \
  pdf2md-rag \
  marker-pdf \
  chromadb \
  sentence-transformers \
  langchain-text-splitters \
  pymupdf
```

如果你确定这些工具也只用于本项目，还可以继续：

```bash
conda activate base
python -m pip uninstall -y typer pytest pre-commit ruff
```

## 第 2 层：谨慎处理底层数值与模型栈

下面这些包不要轻易动，因为它们往往被其他项目复用：

- `transformers`
- `torch`
- `numpy`
- `scipy`

尤其是当前机器里还装了：

- `mlx-lm`
- `mlx-vlm`

如果这些项目仍要继续在 `base` 里运行，就**不要**在 `base` 中卸载 `transformers` / `torch`。

## 第 3 层：确认是否还残留 pdf2md 相关包

```bash
conda activate base
python -m pip list --format=freeze | egrep '^(pdf2md-rag|marker-pdf|chromadb|sentence-transformers|langchain-text-splitters|pymupdf|transformers|torch|numpy|scipy)='
```

## 推荐的最终状态

推荐把 `base` 控制在“尽量少装业务依赖”的状态，并把项目固定到：

```bash
conda activate conda_rag
cd /Users/charles/Downloads/pdf2md
python examples/debug_ingest.py
pytest -q
```

## 如果担心误删

最保守方案其实是：

- 不卸载 `base` 里的任何底层库
- 以后只用 `conda_rag`
- 把 IDE、终端、Run/Debug 配置全部指向 `conda_rag`

这通常已经足够稳定，而且不会影响别的项目。

