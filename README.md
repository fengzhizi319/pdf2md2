# pdf2md-rag

本项目提供一套本地可运行的 RAG 知识库构建流程：

`PDF -> Markdown -> Chunks -> Embedding -> Chroma`

适合在 macOS / Apple Silicon 上把单个 PDF 或一批 PDF 建成可检索的本地知识库。

## 功能

- PDF 提取为 Markdown：默认使用 `Marker`，更适合学术论文、数学公式和 LaTeX 场景
- Markdown 感知式切块，保留章节上下文
- 本地 Embedding：默认 `sentence-transformers`
- 离线测试 Embedding：`hash` embedder
- 本地 Chroma 持久化
- 结构化检索层：`search.py` 返回适合 RAG 拼接的结果结构
- 简易问答脚本：`simple_qa.py` 直接把检索结果喂给本地/远程 LLM
- CLI 一键导入和检索

## 推荐环境

- macOS Apple Silicon（M 系列）
- Python `3.12`
- 标准 conda 环境名：`conda_rag`

> 当前机器默认 `python3` 是 3.13，但本地 embedding / OCR / Marker 相关依赖通常在 3.12 更稳，建议显式使用 `python3.12` 创建虚拟环境。

## 标准 conda 环境（推荐）

仓库根目录已提供 [`environment.yml`](environment.yml)，固定标准环境名为 `conda_rag`。

### 首次创建

```bash
cd ~/Documents/AI/pdf2md
conda env create -f environment.yml
conda activate conda_rag
```

### 已存在时更新

```bash
cd ~/Documents/AI/pdf2md
conda env update -n conda_rag -f environment.yml --prune
conda activate conda_rag
```

这个环境文件会统一安装：

- `python=3.12`
- `llvm-openmp`
- 已验证可工作的 `numpy=2.2.4`
- 已验证可工作的 `scipy=1.15.3`
- 以及当前仓库的 editable 包安装：`-e .[dev]`

> 不建议继续把项目跑在 `base` 里。`base` 往往已经混入其他 AI/ML 包，容易和 `marker-pdf` / `transformers` / `torch` 产生版本漂移。

### IDE 解释器选择

在 PyCharm / IntelliJ / VS Code 中，把项目解释器切到：

```text
<miniconda-root>/envs/conda_rag/bin/python
```

例如当前机器通常会是：

```text
/Users/charles/miniconda3/envs/conda_rag/bin/python
```

## venv 安装（可选）

```bash
cd ~/Documents/AI/pdf2md
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

如果你已经手动创建了自己的 conda 环境，也至少要在该环境里执行：

```bash
conda activate <your-conda-env>
cd ~/Documents/AI/pdf2md
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

> 项目是标准 `src` 布局。切换解释器后，如果没有把项目重新安装到当前环境里，普通脚本运行会出现 `ModuleNotFoundError: pdf2md_rag`。仓库里的 `examples/*.py` 现在会自动补上本地 `src/` 路径，便于在 IDE 中直接调试；但 CLI 和测试仍建议先执行上面的 editable install。

## 标准环境验证

建议在 `conda_rag` 激活后，先做一次真实链路验证，再跑测试：

```bash
cd ~/Documents/AI/pdf2md
conda activate conda_rag
python examples/debug_ingest.py
pytest -q
```

如果只想先做离线快速检查，也可以运行：

```bash
cd ~/Documents/AI/pdf2md
conda activate conda_rag
python examples/debug_embeddings.py
python examples/debug_vectorstore.py
python examples/debug_search.py
python examples/debug_qa.py
python examples/debug_langgraph.py
python examples/debug_langgraph_local_chroma.py
python examples/debug_langgraph_rag.py
```

## 工程化配置

仓库里额外补充了这些工程化文件：

- [`.gitignore`](.gitignore)：忽略本地环境、Chroma 持久化数据、生成的 Markdown/manifest 等产物
- [`.editorconfig`](.editorconfig)：统一缩进、换行和编码
- [`.pre-commit-config.yaml`](.pre-commit-config.yaml)：提交前自动做基础检查与 Ruff 格式化/静态检查
- [`Makefile`](Makefile)：统一常用开发命令
- [`CONTRIBUTING.md`](CONTRIBUTING.md)：贡献与提交流程
- [`docs/dev-environment.md`](docs/dev-environment.md)：`base` 环境清理与隔离策略
- [`docs/ide-run-debug.md`](docs/ide-run-debug.md)：PyCharm / IDEA 最稳 Run/Debug 配置
- [`.github/workflows/ci.yml`](.github/workflows/ci.yml)：CI
- [`.github/workflows/release.yml`](.github/workflows/release.yml)：tag 构建发布产物
- [`.github/dependabot.yml`](.github/dependabot.yml)：自动依赖更新

推荐先安装一次本地 git hook：

```bash
cd ~/Documents/AI/pdf2md
conda activate conda_rag
pre-commit install
pre-commit run --all-files
```

常用工程命令：

```bash
cd ~/Documents/AI/pdf2md
conda activate conda_rag
make lint
make test
make debug-ingest
```

## 默认 PDF 提取器

整个项目现在默认使用 `Marker` 进行 PDF -> Markdown 提取，包括：

- `pdf2md_rag.pdf_to_markdown.extract_markdown`
- `pdf2md_rag.pipeline.ingest_pdf`
- `src/mac_rag_pipeline.py`

这意味着对包含大量数学公式、学术版式、LaTeX 表达式的论文，提取质量通常会明显好于普通纯文本抽取。

## 导入示例 PDF

```bash
cd ~/Documents/AI/pdf2md
source .venv/bin/activate
pdf2md-rag ingest \
  "pdf/Understanding Lasso – A Novel Lookup Argument Protocol.pdf" \
  --collection understanding-lasso \
  --embedding-model BAAI/bge-small-en-v1.5
```

CLI 在完成后会：

- 打印 Markdown 和 Manifest 的输出路径
- 自动校验这两个输出文件确实存在
- 如果文件不存在，会直接退出并返回非零状态码

默认输出：

- Markdown: `pdf/`
- Chroma: `data/chroma/`
- Manifest: `data/manifests/`

## 纯代码调试示例

如果你想在 PyCharm / VS Code 里断点调试，不想用 CLI，可以直接运行仓库里的示例脚本：

```bash
cd ~/Documents/AI/pdf2md
source .venv/bin/activate
python examples/debug_ingest.py
```

如果你在 macOS 的 conda 环境里运行，项目会在调用 Marker 前自动设置一层 OpenMP 兼容开关，避免部分 `libomp` 重复加载导致进程直接 abort。

如果你想单独学习工作流/agent 编排，而不是直接进入 RAG 主链路，可以运行：

```bash
cd ~/Documents/AI/pdf2md
source .venv/bin/activate
python examples/debug_langgraph.py
```

这个脚本不依赖真实 LLM，会用一个内置的小型知识库演示 `StateGraph`、条件分支、`invoke()` 和 `stream()`，适合作为 LangGraph 入门样例。

如果你想看一个更贴近真实 Agent / RAG 工作流的版本，可以运行 `examples/debug_langgraph_rag.py`。它会把流程拆成“分析问题 -> 检索上下文 -> 调用 LLM 生成答案”三个节点，并支持：

- OpenAI 官方接口
- 本地 OpenAI-compatible 服务（如 LM Studio / vLLM / llama.cpp server）
- Ollama

例如：

```bash
cd ~/Documents/AI/pdf2md
source .venv/bin/activate

# OpenAI
OPENAI_API_KEY=<your-key> \
PDF2MD_RAG_LLM_PROVIDER=openai-compatible \
PDF2MD_RAG_LLM_BASE_URL=https://api.openai.com \
PDF2MD_RAG_LLM_MODEL=gpt-4o-mini \
python examples/debug_langgraph_rag.py

# 本地 OpenAI-compatible
PDF2MD_RAG_LLM_PROVIDER=openai-compatible \
PDF2MD_RAG_LLM_BASE_URL=http://localhost:1234 \
PDF2MD_RAG_LLM_MODEL=qwen2.5-7b-instruct \
python examples/debug_langgraph_rag.py

# Ollama
PDF2MD_RAG_LLM_PROVIDER=ollama \
PDF2MD_RAG_LLM_BASE_URL=http://localhost:11434 \
PDF2MD_RAG_LLM_MODEL=qwen2.5:3b \
python examples/debug_langgraph_rag.py
```

如果你想让 LangGraph 直接连接**已经落盘的真实本地 Chroma 知识库**，可以运行 `examples/debug_langgraph_local_chroma.py`。这个脚本不会创建临时 demo 向量库，而是直接读取 `data/chroma/`，并把流程拆成：

- 分析问题
- 校验本地知识库/collection
- 从真实 collection 检索上下文
- 生成答案（默认 fake LLM，也可切到 OpenAI-compatible / Ollama）

例如：

```bash
cd ~/Documents/AI/pdf2md
source .venv/bin/activate

# 先离线学习图结构（默认 fake LLM）
python examples/debug_langgraph_local_chroma.py

# 指定真实 collection
PDF2MD_RAG_COLLECTION=understanding-lasso \
python examples/debug_langgraph_local_chroma.py

# 如果 collection 是 hash embedding
PDF2MD_RAG_COLLECTION=understanding-lasso-hash-debug \
PDF2MD_RAG_EMBEDDER=hash \
PDF2MD_RAG_EMBEDDING_MODEL=unused \
python examples/debug_langgraph_local_chroma.py

# 切到 OpenAI-compatible
OPENAI_API_KEY=<your-key> \
PDF2MD_RAG_LLM_PROVIDER=openai-compatible \
PDF2MD_RAG_LLM_BASE_URL=https://api.openai.com \
PDF2MD_RAG_LLM_MODEL=gpt-4o-mini \
python examples/debug_langgraph_local_chroma.py
```

这个示例特别适合学习“为什么图编排在真实知识库场景下仍然有价值”：因为你能把 collection 校验、检索失败处理、无命中兜底和生成阶段分成清晰的节点，而不是把所有逻辑塞进一个函数。

它内部直接调用：

- `PipelineConfig`
- `ingest_pdf(...)`

并会打印：

- 输入 PDF 路径
- Markdown 输出路径与是否存在
- Manifest 输出路径与是否存在
- 页数 / chunk 数 / 向量数

如果你想临时自己写一段最小调试代码，也可以直接用下面这个例子：

```python
from pathlib import Path

from pdf2md_rag.config import DEFAULT_CHROMA_DIR, DEFAULT_MANIFEST_DIR, DEFAULT_MARKDOWN_DIR, PipelineConfig
from pdf2md_rag.pipeline import ingest_pdf

pdf_path = Path("pdf/Understanding Lasso – A Novel Lookup Argument Protocol.pdf").resolve()

config = PipelineConfig(
    markdown_dir=DEFAULT_MARKDOWN_DIR,
    chroma_dir=DEFAULT_CHROMA_DIR,
    manifest_dir=DEFAULT_MANIFEST_DIR,
    collection_name="understanding-lasso-hash-debug",
    embedder_type="hash",
    embedding_model="unused",
)

result = ingest_pdf(pdf_path, config)
print(result)
print(Path(result.markdown_path).exists(), result.markdown_path)
print(Path(result.manifest_path).exists(), result.manifest_path)
```

## 检索测试

```bash
cd ~/Documents/AI/pdf2md
source .venv/bin/activate
pdf2md-rag query \
  --collection understanding-lasso \
  --question "What is the main idea of the Lasso lookup protocol?" \
  --top-k 5 \
  --embedding-model BAAI/bge-small-en-v1.5
```

## 在 Python 里调用结构化检索

`src/pdf2md_rag/search.py` 会返回一个更适合 RAG 拼接的 `SearchResult`：

- `hits`：每个 chunk 的结构化命中信息
- `context_text`：已经拼好的上下文文本，可直接喂给 LLM
- `sources`：便于引用的来源列表
- `retrieval_meta`：检索配置与统计信息

```bash
cd ~/Documents/AI/pdf2md
source .venv/bin/activate
python - <<'PY'
from pdf2md_rag.search import search_chunks

result = search_chunks(
    question="What is the main idea of the Lasso lookup protocol?",
    collection_name="understanding-lasso-hash",
    persist_directory="data/chroma",
    top_k=3,
    embedder_type="hash",
    embedding_model="unused",
)
print(result.context_text)
print(result.sources)
PY
```

## 用本地 / 远程 LLM 做问答

### 1) Ollama 本地问答

```bash
cd ~/Documents/AI/pdf2md
source .venv/bin/activate
pdf2md-rag-qa \
  "What is the main idea of the Lasso lookup protocol?" \
  --collection understanding-lasso-hash \
  --embedder hash \
  --embedding-model unused \
  --llm-provider ollama \
  --llm-model llama3.1 \
  --llm-base-url http://localhost:11434
```

### 2) OpenAI 兼容接口问答

这个接口可对接 OpenAI、vLLM、LM Studio、llama.cpp server 等兼容 `/v1/chat/completions` 的服务。

```bash
cd ~/Documents/AI/pdf2md
source .venv/bin/activate
export OPENAI_API_KEY="your-api-key"
pdf2md-rag-qa \
  "Summarize the Lasso lookup protocol in 5 bullet points." \
  --collection understanding-lasso \
  --llm-provider openai-compatible \
  --llm-model gpt-4o-mini \
  --llm-base-url https://api.openai.com
```

## 离线快速测试

如果你只是不想先下载 embedding 模型，可以先用 `hash` embedder 验证向量化与入库链路：

```bash
cd ~/Documents/AI/pdf2md
source .venv/bin/activate
pdf2md-rag ingest \
  "pdf/Understanding Lasso – A Novel Lookup Argument Protocol.pdf" \
  --collection understanding-lasso-hash \
  --embedder hash
```

> 注意：这里的“离线快速测试”只是不下载 embedding 模型；PDF 提取阶段仍会使用 `Marker`。

## Marker / Apple Silicon 说明

- 默认 PDF 提取器会优先尝试 `mps`，不可用时回退到 CPU。
- 为了避免 `surya` 的 `TableRecEncoderDecoderModel` 在 `mps` 上强制 fallback 到 CPU，项目在 `mps` 模式下会自动禁用表格识别相关处理器。
- 这意味着正文、标题、数学公式等主路径仍然使用 `Marker + mps` 加速，但复杂表格结构提取能力会有所下降。
- `Marker` 对学术论文、公式和 LaTeX 保留更友好，但首次运行可能需要初始化模型，耗时会比纯文本提取更长。
- 默认 embedding 代码也会优先尝试 `mps`，不可用时回退到 CPU。
- 首次运行 `sentence-transformers` 会下载 embedding 模型，需要联网。
- 若使用本地 LLM，Ollama 是最省事的方式；若使用远程或自托管 OpenAI 兼容接口，直接指向对应 `base_url` 即可。

## 目录结构

- `src/pdf2md_rag/`：主代码
- `tests/`：单元测试与 smoke test

## 源码阅读入口

如果你想系统读懂这个项目，推荐按下面的顺序阅读：

### 第一步：先看全局架构

- [`docs/architecture.md`](docs/architecture.md)
  - 看模块分层
  - 看主数据流
  - 先建立 `PDF -> Markdown -> Chunks -> Embeddings -> Chroma -> Search -> QA` 的整体印象

### 第二步：再看调用关系

- [`docs/call-graph.md`](docs/call-graph.md)
  - 看“谁调用了谁”
  - 适合在 IDE 里配合“跳转到定义”一起看

### 第三步：从可运行代码入口开始读

- [`examples/debug_ingest.py`](examples/debug_ingest.py)
  - 最适合一边运行、一边断点调试
- [`src/pdf2md_rag/cli.py`](src/pdf2md_rag/cli.py)
  - 看命令行入口如何组装参数
- [`src/pdf2md_rag/pipeline.py`](src/pdf2md_rag/pipeline.py)
  - 看主流程编排

### 第四步：按主链路继续往下看

1. [`src/pdf2md_rag/models.py`](src/pdf2md_rag/models.py)
2. [`src/pdf2md_rag/config.py`](src/pdf2md_rag/config.py)
3. [`src/pdf2md_rag/pipeline.py`](src/pdf2md_rag/pipeline.py)
4. [`src/pdf2md_rag/pdf_to_markdown.py`](src/pdf2md_rag/pdf_to_markdown.py)
5. [`src/pdf2md_rag/chunking.py`](src/pdf2md_rag/chunking.py)
6. [`src/pdf2md_rag/embeddings.py`](src/pdf2md_rag/embeddings.py)
7. [`src/pdf2md_rag/vectorstore.py`](src/pdf2md_rag/vectorstore.py)
8. [`src/pdf2md_rag/search.py`](src/pdf2md_rag/search.py)
9. [`src/pdf2md_rag/simple_qa.py`](src/pdf2md_rag/simple_qa.py)
10. [`src/mac_rag_pipeline.py`](src/mac_rag_pipeline.py)

这套顺序的思路是：

- 先理解全局
- 再理解调用关系
- 最后深入每个模块细节

## examples 调试脚本地图

`examples/` 目录现在按“阶段”提供了一组 debug 脚本，适合学习和断点调试：

- [`examples/debug_extract_markdown.py`](examples/debug_extract_markdown.py)
  - 对应：`pdf_to_markdown.extract_markdown`
  - 作用：查看 PDF 提取后的页数、字符数、Markdown 预览
- [`examples/debug_chunking.py`](examples/debug_chunking.py)
  - 对应：`chunking.chunk_markdown`
  - 作用：查看 chunk 数量、`chunk_id`、heading、page、文本预览
- [`examples/debug_embeddings.py`](examples/debug_embeddings.py)
  - 对应：`embeddings.build_embedder`
  - 作用：查看 embedding 维度和向量内容示例
  - 特点：默认使用真实 `sentence-transformers` 嵌入（首次运行会下载模型）
- [`examples/debug_vectorstore.py`](examples/debug_vectorstore.py)
  - 对应：`vectorstore.upsert_chunks` / `vectorstore.query_collection`
  - 作用：查看向量入库与 Top-K 查询结果
  - 特点：默认使用真实 `sentence-transformers` 嵌入 + 临时 Chroma
- [`examples/debug_search.py`](examples/debug_search.py)
  - 对应：`search.search_chunks`
  - 作用：查看 `SearchHit`、`sources` 和 `context_text`
  - 特点：默认使用真实 `sentence-transformers` 嵌入 + 临时 Chroma
- [`examples/debug_qa.py`](examples/debug_qa.py)
  - 对应：`simple_qa.ask_question`
  - 作用：查看 QA 层如何把检索结果拼成 prompt 并得到答案
  - 特点：embedding 使用真实 `sentence-transformers`；LLM 仍用假的 OpenAI-compatible 响应
- [`examples/debug_ingest.py`](examples/debug_ingest.py)
  - 对应：`pipeline.ingest_pdf`
  - 作用：查看完整 ingest 执行结果
  - 特点：默认使用真实 `sentence-transformers` 嵌入，可通过环境变量 `PDF2MD_RAG_EMBEDDING_MODEL` 覆盖模型名

推荐从这个顺序开始：

1. `debug_extract_markdown.py`
2. `debug_chunking.py`
3. `debug_embeddings.py`
4. `debug_vectorstore.py`
5. `debug_search.py`
6. `debug_qa.py`
7. `debug_ingest.py`
