# Architecture Overview

`pdf2md-rag` 是一个本地 RAG 项目，主目标是把 PDF 文档转成可检索、可问答的知识库。

核心链路可以概括为：

`PDF -> Markdown -> Chunks -> Embeddings -> Chroma -> Search -> QA`

---

## 1. 模块地图

按职责分层后，项目主干可以分成这几层：

### 入口层
- `src/pdf2md_rag/cli.py`
  - 命令行入口，负责参数解析、结果打印、输出文件校验。
- `src/mac_rag_pipeline.py`
  - 更适合本地演示和学习的脚本版入口。
- `examples/debug_ingest.py`
  - 纯代码调试示例，方便在 IDE 中断点运行。

### 配置与数据契约层
- `src/pdf2md_rag/config.py`
  - 定义默认目录与 `PipelineConfig`。
- `src/pdf2md_rag/models.py`
  - 定义 `MarkdownDocument` 和 `Chunk`。

### 处理层
- `src/pdf2md_rag/pdf_to_markdown.py`
  - 使用 `Marker` 把 PDF 提取为 Markdown。
- `src/pdf2md_rag/chunking.py`
  - 把 Markdown 切成适合检索的语义片段。
- `src/pdf2md_rag/embeddings.py`
  - 构造 embedding 后端（`sentence-transformers` 或 `hash`）。

### 存储与检索层
- `src/pdf2md_rag/vectorstore.py`
  - 对 Chroma 的薄封装。
- `src/pdf2md_rag/search.py`
  - 把向量检索结果整理成 RAG 友好的 `SearchResult`。

### 应用层
- `src/pdf2md_rag/pipeline.py`
  - 主编排层，串起 ingest 全流程。
- `src/pdf2md_rag/simple_qa.py`
  - 在检索结果之上调用 LLM 做问答。

---

## 2. 核心数据对象

### `PipelineConfig`
描述一次 ingest 的主要参数：
- Markdown 输出目录
- Chroma 持久化目录
- manifest 输出目录
- chunk 大小 / overlap
- embedding 后端与模型

### `MarkdownDocument`
表示 PDF 提取阶段的输出：
- `source_path`
- `text`
- `page_count`

### `Chunk`
表示切块阶段的输出：
- `chunk_id`
- `text`
- `metadata`

### `IngestResult`
表示 ingest 完成后的摘要结果：
- PDF 路径
- Markdown 路径
- manifest 路径
- collection 名称
- 页数 / chunk 数 / 向量数

### `SearchHit` / `SearchResult`
表示结构化检索结果：
- 原始命中片段
- 引用信息
- 拼装好的 `context_text`
- 供 QA 层直接使用的 sources

### `QAResult`
表示问答执行结果：
- 问题
- 最终答案
- 所使用的检索结果
- LLM provider / model
- 原始响应体

---

## 3. Ingest 主链路（文字版流程图）

```text
[PDF file]
   |
   v
[pdf_to_markdown.extract_markdown]
   - 选择 Marker 设备（优先 MPS）
   - 处理 MPS 下的表格识别兼容问题
   - 调用 Marker 输出 Markdown
   |
   v
[MarkdownDocument]
   |
   +--> 写入 markdown_dir/{pdf_stem}.md
   |
   v
[chunking.chunk_markdown]
   - 按段落切分
   - 记录 heading / page / overlap
   - 生成稳定 chunk_id
   |
   v
[list[Chunk]]
   |
   v
[embeddings.build_embedder]
   - sentence-transformers: 真正语义向量
   - hash: 离线调试向量
   |
   v
[list[embedding vector]]
   |
   v
[vectorstore.upsert_chunks]
   - 写入本地 Chroma collection
   |
   v
[Chroma persistent store]
   |
   +--> data/chroma/
   |
   +--> data/manifests/{pdf_stem}.json
   |
   +--> 返回 IngestResult
```

---

## 4. Search 主链路（文字版流程图）

```text
[user question]
   |
   v
[embeddings.build_embedder]
   |
   v
[query embedding]
   |
   v
[vectorstore.query_collection]
   - 向 Chroma 发起向量检索
   - 取回 documents / metadata / distance
   |
   v
[search.search_chunks]
   - 转成 SearchHit
   - 生成 citation
   - 拼成 context_text
   |
   v
[SearchResult]
```

---

## 5. QA 主链路（文字版流程图）

```text
[user question]
   |
   v
[simple_qa.ask_question]
   |
   +--> search.search_chunks(...)
   |       |
   |       v
   |    [SearchResult]
   |
   +--> _build_user_prompt(...)
   |       |
   |       v
   |    [RAG prompt]
   |
   +--> 调用 LLM 后端
           - openai-compatible
           - ollama
   |
   v
[QAResult]
   - answer
   - search_result
   - provider / model
   - raw_response
```

---

## 6. 关键落盘物

### `pdf/`
保存提取出来的 Markdown 文件。

典型输出：
- `pdf/Understanding Lasso – A Novel Lookup Argument Protocol.md`
- `pdf/math-demo.marker.md`（脚本演示版）

### `data/chroma/`
保存 Chroma 本地向量库数据。

### `data/manifests/`
保存 ingest 执行摘要与 chunk 预览，方便排查：
- 输出路径是否正确
- chunk 是否切得合理
- metadata 是否完整

---

## 7. 平台与依赖约束

### Marker 与 MPS
项目默认使用 `Marker` 做 PDF 提取，并优先尝试 `mps`。

但 `surya` 的表格识别模型 `TableRecEncoderDecoderModel` 在 MPS 上会强制退回 CPU。为了保证主路径仍然使用 MPS：
- 项目在 MPS 模式下会禁用表格识别相关处理器
- 正文、公式、标题的提取仍保留 `Marker + MPS`
- 代价是复杂表格结构提取能力下降

### Hash Embedder
`hash` 模式不是语义检索方案，而是：
- 离线调试
- 管线自测
- 避免下载模型

### Sentence Transformers
默认语义向量方案，适合真实检索与 QA。

---

## 8. 扩展点

如果你要扩展这个项目，通常从这些位置下手：

### 替换 PDF 提取器
修改：
- `src/pdf2md_rag/pdf_to_markdown.py`

### 替换切块策略
修改：
- `src/pdf2md_rag/chunking.py`

### 新增 embedding 后端
修改：
- `src/pdf2md_rag/embeddings.py`

### 替换向量数据库
修改：
- `src/pdf2md_rag/vectorstore.py`

### 新增检索重排 / rerank
优先扩展：
- `src/pdf2md_rag/search.py`

### 新增 LLM provider
优先扩展：
- `src/pdf2md_rag/simple_qa.py`

---

## 9. 推荐阅读顺序

如果你是第一次读源码，推荐按这个顺序：

1. `src/pdf2md_rag/models.py`
2. `src/pdf2md_rag/config.py`
3. `src/pdf2md_rag/pipeline.py`
4. `src/pdf2md_rag/pdf_to_markdown.py`
5. `src/pdf2md_rag/chunking.py`
6. `src/pdf2md_rag/embeddings.py`
7. `src/pdf2md_rag/vectorstore.py`
8. `src/pdf2md_rag/search.py`
9. `src/pdf2md_rag/simple_qa.py`
10. `src/pdf2md_rag/cli.py`
11. `src/mac_rag_pipeline.py`

这样最容易先建立全局，再回头看每个细节。
