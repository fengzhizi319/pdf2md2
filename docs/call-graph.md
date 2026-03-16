# Call Graph Guide

这份文档专门回答一个问题：**谁调用了谁？**

和 `docs/architecture.md` 的区别是：
- `architecture.md` 偏全局职责和数据流
- `call-graph.md` 偏源码阅读时的“函数调用路径”

建议把它当作源码导览地图，而不是严格意义上的静态编译调用图。

---

## 1. 阅读说明

这里主要关注项目内部最重要的显式调用关系：

- CLI ingest 链路
- 纯代码调试脚本链路
- 检索链路
- QA 链路
- 演示脚本链路

不重点展开：

- 第三方库内部调用
  - `Marker`
  - `Chroma`
  - `sentence-transformers`
  - `Ollama / OpenAI-compatible server`
- Python 标准库内部细节

---

## 2. CLI ingest 调用链

入口文件：`src/pdf2md_rag/cli.py`

```text
cli.ingest(...)
  -> PipelineConfig(...)
  -> pipeline.ingest_pdf(pdf_path, config)
       -> config.markdown_dir.mkdir(...)
       -> config.chroma_dir.mkdir(...)
       -> config.manifest_dir.mkdir(...)
       -> pdf_to_markdown.extract_markdown(pdf_path)
            -> pdf_to_markdown.get_marker_device(...)
            -> pdf_to_markdown.should_disable_table_rec(...)
            -> pdf_to_markdown.load_marker_models(...)
            -> pdf_to_markdown.get_marker_processor_list(...)
            -> Marker PdfConverter(...)
       -> markdown_path.write_text(...)
       -> chunking.chunk_markdown(document, ...)
            -> chunking._split_blocks(...)
            -> chunking._split_large_block(...)   [仅超长 block 时]
            -> chunking._carry_overlap(...)       [需要 overlap 时]
            -> chunking._build_chunk(...)
       -> embeddings.build_embedder(...)
            -> embeddings.HashingEmbedder(...) or embeddings.SentenceTransformerEmbedder(...)
       -> embedder.embed_texts(...)
       -> vectorstore.upsert_chunks(...)
       -> manifest_path.write_text(...)
       -> return IngestResult(...)
  -> cli._verify_output_file("Markdown", ...)
  -> cli._verify_output_file("Manifest", ...)
  -> typer.echo(...)
```

### 这条链路最关键的阅读点

如果你只想搞懂 ingest 的主线，建议按这个顺序跳转：

1. `cli.ingest`
2. `pipeline.ingest_pdf`
3. `pdf_to_markdown.extract_markdown`
4. `chunking.chunk_markdown`
5. `embeddings.build_embedder`
6. `vectorstore.upsert_chunks`

---

## 3. 纯代码调试脚本调用链

入口文件：`examples/debug_ingest.py`

```text
debug_ingest.main()
  -> PipelineConfig(...)
  -> pipeline.ingest_pdf(PDF_PATH, config)
       -> [后续完全复用 ingest 主链路]
  -> Path(result.markdown_path).resolve()
  -> Path(result.manifest_path).resolve()
  -> print(...)
  -> exists() 校验
```

### 作用

这条链路比 CLI 更适合：

- 在 IDE 中打断点
- 查看 `result` 对象
- 查看中间变量
- 手动修改 `PipelineConfig`

如果你的目标是“源码学习 + debug”，这是最推荐入口。

---

## 4. 检索调用链

入口文件：`src/pdf2md_rag/search.py`

```text
search.search_chunks(question, ...)
  -> embeddings.build_embedder(...)
       -> HashingEmbedder(...) or SentenceTransformerEmbedder(...)
  -> embedder.embed_query(question)
  -> vectorstore.query_collection(...)
       -> Chroma collection.query(...)
  -> search._row_to_hit(...)          [对每一条结果执行]
  -> search._build_context_text(...)
  -> return SearchResult(...)
```

### 关键理解点

`search_chunks` 做了两件很重要的事：

1. 把底层向量库结果转成项目自己的 `SearchHit`
2. 拼出 `context_text`，方便直接喂给 LLM

所以它是：

```text
向量检索层 -> RAG 上下文层
```

之间的适配器。

---

## 5. QA 调用链

入口文件：`src/pdf2md_rag/simple_qa.py`

```text
simple_qa.ask_question(question, ...)
  -> search.search_chunks(question, ...)
       -> [复用检索链路]
  -> simple_qa._build_user_prompt(question, search_result)
  -> 根据 llm_provider 分支：
       -> simple_qa._call_openai_compatible(...)
            -> simple_qa._post_json(...)
            -> simple_qa._extract_openai_answer(...)
       or
       -> simple_qa._call_ollama(...)
            -> simple_qa._post_json(...)
            -> simple_qa._extract_ollama_answer(...)
  -> return QAResult(...)
```

### 关键理解点

`ask_question` 并不自己做检索算法，它是“应用层拼装”：

```text
SearchResult -> Prompt -> LLM -> QAResult
```

所以如果你以后要：

- 改 prompt 模板
- 加新 LLM provider
- 做答案后处理

优先看这个文件。

---

## 6. CLI query 调用链

入口文件：`src/pdf2md_rag/cli.py`

```text
cli.query(question, ...)
  -> embeddings.build_embedder(...)
  -> query_embedder.embed_query(question)
  -> vectorstore.query_collection(...)
  -> typer.echo(...)  [直接打印结果]
```

### 和 `search.search_chunks` 的区别

`cli.query` 是一个更薄的命令行查询入口：

- 直接查向量库
- 直接打印结果
- 不做 `SearchResult` 结构化封装

而 `search.search_chunks` 更适合 Python 调用和 QA 复用。

---

## 7. Mac 演示脚本调用链

入口文件：`src/mac_rag_pipeline.py`

```text
setup_device()
  -> pdf_to_markdown.get_marker_device()

parse_pdf_to_md(pdf_path, output_md_path, ...)
  -> pdf_to_markdown.extract_markdown(...)
  -> output_md_path.write_text(...)

chunk_markdown_with_math_protection(md_text)
  -> RecursiveCharacterTextSplitter(...)
  -> split_text(...)

__main__
  -> setup_device()
  -> parse_pdf_to_md(...)
  -> chunk_markdown_with_math_protection(...)
  -> print 前 2 个 chunk
```

### 它和主项目的关系

这个脚本不是另一套核心实现。
它只是：

- 复用共享提取逻辑
- 以更直观的方式演示本地跑通 `PDF -> Markdown -> chunk`

真正的主编排仍然是：

- `pipeline.ingest_pdf`

---

## 8. 数据对象在调用链中的流动

```text
PDF path
  -> extract_markdown(...)
  -> MarkdownDocument
  -> chunk_markdown(...)
  -> list[Chunk]
  -> build_embedder(...)
  -> list[list[float]] embeddings
  -> upsert_chunks(...)
  -> IngestResult

question
  -> build_embedder(...)
  -> query embedding
  -> query_collection(...)
  -> list[row]
  -> SearchHit / SearchResult
  -> ask_question(...)
  -> QAResult
```

如果你在阅读源码时迷路，可以优先问自己：

> 当前这个函数输入的是什么对象？输出的又是什么对象？

多数情况下，顺着这些数据对象走，就能自然看懂整个项目。

---

## 9. 最推荐的跳转顺序

如果你正在 IDE 里按“跳转到定义”读源码，推荐用这个顺序：

### 想看 ingest 主线

1. `src/pdf2md_rag/cli.py` -> `ingest`
2. `src/pdf2md_rag/pipeline.py` -> `ingest_pdf`
3. `src/pdf2md_rag/pdf_to_markdown.py` -> `extract_markdown`
4. `src/pdf2md_rag/chunking.py` -> `chunk_markdown`
5. `src/pdf2md_rag/embeddings.py` -> `build_embedder`
6. `src/pdf2md_rag/vectorstore.py` -> `upsert_chunks`

### 想看 search / QA 主线

1. `src/pdf2md_rag/simple_qa.py` -> `ask_question`
2. `src/pdf2md_rag/search.py` -> `search_chunks`
3. `src/pdf2md_rag/vectorstore.py` -> `query_collection`
4. 回到 `simple_qa.py` 看 `_build_user_prompt` 和 provider 分支

### 想边跑边看

1. `examples/debug_ingest.py`
2. `src/pdf2md_rag/pipeline.py`
3. 对关键点打断点

---

## 10. 和 `architecture.md` 搭配阅读建议

推荐这样搭配：

1. 先看 `docs/architecture.md`
   - 建立整体模块分层和数据流印象
2. 再看 `docs/call-graph.md`
   - 确认真实调用路径
3. 然后打开 `examples/debug_ingest.py`
   - 从可运行入口开始跳转源码

这三份结合起来，最适合第一次系统读懂这个项目。
