# PyCharm / IntelliJ IDEA Run & Debug 配置

这份配置以“最稳”为目标：

- 统一解释器：`conda_rag`
- 统一工作目录：仓库根目录
- 统一从项目根启动脚本 / 模块
- 避免把 `.venv`、`base`、IDE 内建终端混在一起

## 1. 解释器

在 IDE 中选择项目解释器：

```text
/Users/charles/miniconda3/envs/conda_rag/bin/python
```

如果你的 Miniconda 不在这个路径，请替换成你机器上的实际路径，但环境名应保持为：

```text
conda_rag
```

## 2. 通用配置原则

无论是普通脚本还是 CLI，统一使用下面这些原则：

- Working directory：`/Users/charles/Downloads/pdf2md`
- Python interpreter：`conda_rag`
- 勾选 `Add content roots to PYTHONPATH`
- 勾选 `Add source roots to PYTHONPATH`
- 不要复用旧 `.venv` 的 Run Configuration

虽然项目已经做了 `src/` 自举兼容，但 IDE 里把 source root 加上，仍然是最稳的配置。

## 3. 推荐 Run/Debug 配置

### A. 调试完整 ingest

这是最推荐的入口。

- Configuration type: `Python`
- Name: `debug_ingest`
- Script path: `examples/debug_ingest.py`
- Working directory: `/Users/charles/Downloads/pdf2md`
- Python interpreter: `conda_rag`
- Parameters: 留空

运行效果：

- 真实执行 `PDF -> Markdown -> Chunk -> Embedding -> Chroma`
- 适合沿着 `pipeline.py` 全链路打断点

### B. 调试 CLI ingest

如果你想调命令行层，而不是示例脚本：

- Configuration type: `Python`
- Name: `cli_ingest_hash`
- Module name: `pdf2md_rag.cli`
- Working directory: `/Users/charles/Downloads/pdf2md`
- Python interpreter: `conda_rag`
- Parameters:

```text
ingest "pdf/Understanding Lasso – A Novel Lookup Argument Protocol.pdf" --collection understanding-lasso-hash-debug --embedder hash --embedding-model unused
```

适合观察：

- `typer` 参数解析
- `PipelineConfig` 组装
- CLI 输出校验逻辑

### C. 调试离线 search

- Configuration type: `Python`
- Name: `debug_search`
- Script path: `examples/debug_search.py`
- Working directory: `/Users/charles/Downloads/pdf2md`
- Python interpreter: `conda_rag`

这个入口不依赖真实 PDF 提取，启动更快。

### D. 调试 QA

- Configuration type: `Python`
- Name: `debug_qa`
- Script path: `examples/debug_qa.py`
- Working directory: `/Users/charles/Downloads/pdf2md`
- Python interpreter: `conda_rag`

这是最适合看 RAG 最终 prompt 组装的位置。

## 4. 终端设置建议

如果你使用 IDE 内置 Terminal，建议确认：

- Shell path 使用 `zsh`
- Terminal 启动后先执行：

```bash
conda activate conda_rag
```

如果你希望每次自动进入该环境，也可以把 Terminal 的启动命令改成带 conda 激活的形式；但通常手动激活更容易排查问题。

## 5. 常见错误与对应修复

### 错误 1：`ModuleNotFoundError: pdf2md_rag`

通常说明发生了以下情况之一：

- 解释器仍然指向旧 `.venv`
- 当前解释器没有执行过 `pip install -e '.[dev]'`
- Working directory 不在项目根目录

修复：

```bash
conda activate conda_rag
cd /Users/charles/Downloads/pdf2md
python -m pip install -e '.[dev]'
```

### 错误 2：Marker / OpenMP 直接 abort

项目代码已经在 macOS + conda 下自动设置兼容环境变量；如果仍然异常，优先确认：

- 运行配置解释器是不是 `conda_rag`
- 是否误用了 `base`
- 是否误用了旧 `.venv`

### 错误 3：命令行能跑，IDE 里不能跑

一般是 Run Configuration 的 Working directory 配错了。必须确认是：

```text
/Users/charles/Downloads/pdf2md
```

## 6. 建议保留的配置集合

实际开发时，保留下面 4 个 Run/Debug 配置就够用了：

1. `debug_ingest`
2. `debug_search`
3. `debug_qa`
4. `cli_ingest_hash`

这 4 个分别覆盖：

- 完整 ingest 主流程
- 检索结构层
- QA 层
- CLI 封装层

