---
title: "【解密源码】Cognee 四大核心功能实现剖析"
date: 2025-12-15T11:33:10+08:00
draft: false
tags: ["源码","技术","Agent"]
categories: ["Cognee"]
---

# 引言

## Cognee 的核心价值主张

Cognee 是一个 **AI Memory Layer**，将原始数据转化为可查询的知识网络。与传统 RAG（检索增强生成）系统仅提供文本检索不同，Cognee 构建的是**结构化知识图谱**。

四个关键操作定义了完整的工作流：
- **.add** - 准备数据：清洁、规范化、元数据提取
- **.cognify** - 构建知识图谱：LLM 驱动的实体、关系、嵌入提取
- **.search** - 上下文查询：向量相似度 + 图遍历的融合
- **.memify** - 语义增强：进一步的知识优化（进行中）

## 四大功能

| 功能 | 职责 | 关键特性 |
|------|------|---------|
| **.add** | 数据摄取与准备 | 多源适配、去重、元数据 |
| **.cognify** | 知识图谱构建 |  LLM 驱动、七步骤管道 |
| **.search** | 上下文感知查询 | 六大搜索模式 |
| **.memify** | 语义增强 | 灵活任务编排 |

---

# 一、.add 功能实现详解

## 1.1 功能简述

**.add** 是 Cognee 工作流的第一步，负责将多源异构数据统一为结构化的 Data 对象，存储到数据库。

## 1.2 核心操作

**.add** 的三个关键操作**：

```
输入数据 → [路径解析] → [文本提取] → [去重检查] → [数据库存储] → Data 对象
           ↓
        多源适配        文件分类       内容哈希        元数据记录
        (本地/S3/URL)   (PDF/DOCX/    (SHA-256)       (loader/size/
         HTML/纯文本)   extension)                     timestamp)
```

**Operation 1：路径解析与多源适配**
- 支持输入：本地文件路径、S3 路径、URL、纯文本字符串、二进制流
- 输出：统一的文件路径或临时存储位置
- 核心函数：`resolve_data_directories()`、`save_data_item_to_storage()`

**Operation 2：文本提取与加载器适配**
- 通过加载器（Loader）解析不同文件格式
- 支持：PDF、DOCX、CSV、Code、WebPage 等
- 输出：规范化的文本文件与元数据

**Operation 3：去重与数据库存储**
- 计算文件内容哈希（SHA-256）
- 生成确定性 UUID（哈希 + 用户 ID）
- 检查数据集中是否已存在
- 新增或更新 Data 对象到数据库

## 1.3 实现架构概览

```python
async def add(
    data: Union[BinaryIO, list[BinaryIO], str, list[str]],
    dataset_name: str = "main_dataset",
    user: User = None,
    node_set: Optional[List[str]] = None,
    vector_db_config: dict = None,
    graph_db_config: dict = None,
    dataset_id: Optional[UUID] = None,
    preferred_loaders: Optional[List[...]] = None,
    incremental_loading: bool = True,
    data_per_batch: Optional[int] = 20,
) -> PipelineRunInfo:
```

**关键参数的设计意义**：
- `data`：支持多种输入格式的多态设计
- `dataset_name`：逻辑隔离与权限管理
- `preferred_loaders`：灵活的文件格式扩展
- `incremental_loading`：防止重复与幂等性
- `data_per_batch`：内存管理与性能调优

## 1.4 加载器与文档格式适配

Cognee 支持多种文档格式，通过不同的加载器（Loader）进行解析。系统会按 `preferred_loaders` 参数指定的顺序尝试：

**主要加载器类型与处理特性**：

| 加载器 | 文件类型 | 关键处理能力 | 依赖库 | 可选性 |
|--------|---------|-----------|--------|--------|
| **pypdf_loader** | .pdf | 按页文本提取、页码标记 | `pypdf` | 核心（必备） |
| **advanced_pdf_loader** | .pdf | 布局感知、表格识别、图像位置 | `unstructured` + fallback to `pypdf` | 可选（推荐） |
| **unstructured_loader** | .docx, .doc, .xlsx, .xls, .pptx, .ppt, .odt, .html, .htm, .eml, .epub 等 | 多格式统一解析、结构化分区 | `unstructured` | 可选 |
| **beautiful_soup_loader** | .html, URL | 网页解析、噪声移除、CSS 选择器/XPath 提取 | `beautifulsoup4`, 可选 `lxml`（XPath） | 可选 |
| **text_loader** | .txt, .md, .json, .xml | 纯文本解析、基础格式支持 | `os`（标准库） | 核心（必备） |
| **csv_loader** | .csv | CSV 表格解析、列识别 | `csv`（标准库） | 核心（必备） |
| **image_loader** | .png, .jpg, .jpeg, .gif, .webp, .bmp 等 | 图像转文本（通过 LLM 视觉识别） | `os` + 内部 LLMGateway | 核心（必备） |
| **audio_loader** | .mp3, .wav, .m4a, .ogg, .flac 等 | 音频转写（通过 LLM 语音识别） | `os` + 内部 LLMGateway | 核心（必备） |

**LLMGateway 简介**：

`LLMGateway` 是 Cognee 内置的 **LLM 通用适配层**，为多媒体文件提供统一的转文本接口。它支持：
- `transcribe_image()` - 调用配置的 LLM（如 GPT-4）进行图像视觉识别
- `create_transcript()` - 调用配置的 LLM 进行音频语音转写

这样 image/audio 加载器无需依赖专门的 OCR 或语音识别库（如 Tesseract、Whisper 等），而是直接使用项目配置的 LLM provider。若未配置 LLM，这两个加载器在使用时会抛出相应错误。

**加载器注册机制**：

系统通过条件导入注册加载器。核心加载器（text、csv、image）始终可用。可选加载器（unstructured、advanced_pdf、beautiful_soup）在对应库未安装时会被跳过，不影响其他加载器的使用。

**自定义加载器扩展**：
```python
await cognee.add(
    data=[file1, file2, file3],
    preferred_loaders=[CustomJSONLoader, advanced_pdf_loader, pypdf_loader, TextLoader],
)
```

系统会按优先级顺序尝试每个加载器，直到成功解析文件。若指定的加载器未安装，会自动跳过到下一个。

## 1.6 去重机制实现

**.add** 的去重采用**三层策略**，确保同一数据的幂等性：

**Layer 1：内容哈希识别**
```python
# 计算文件内容的 SHA-256 哈希
content_hash = hashlib.sha256(file_content).hexdigest()

# 生成确定性 UUID：哈希值 + 用户 ID
data_id = uuid5(NAMESPACE_OID, f"{content_hash}:{user.id}")
```

**Layer 2：数据集级检查**
```python
# 查询数据集中是否已存在该数据
existing_data = await session.execute(
    select(Data).where(
        (Data.id == data_id) &
        (Data.dataset_id == dataset.id)
    )
)

if existing_data.scalars().first():
    # 已存在：跳过或更新元数据
    continue
else:
    # 新数据：插入
    session.add(new_data_object)
```

**Layer 3：增量加载**
```python
if incremental_loading:
    # 仅处理新数据（内容哈希与现有数据不重复）
    new_data_list = filter_existing_hashes(data_list, user.id)
else:
    # 强制重新处理全部（用于算法迭代或修复）
    new_data_list = data_list
```

**Layer 3 的实现机制**：

- **增量模式**（incremental_loading=True）：
  - 调用 `filter_existing_hashes()` 过滤已存在的数据
  - 只有内容哈希在用户的数据库中不存在的文件才会被加入处理队列
  - 适用于日常数据导入，避免重复处理

- **全量重新处理模式**（incremental_loading=False）：
  - 绕过哈希检查，所有数据都进入处理队列
  - 即使数据已存在，也会被重新摄取、分块、向量化
  - 适用于 LLM 模型升级或去重规则变更等场景

**工作流程**：
```
用户提交数据列表
  ↓
Layer 1: 计算每个文件的 SHA-256 哈希 + 生成 UUID
  ↓
Layer 2: 逐个查询数据库检查是否已存在
  ↓
Layer 3: 根据 incremental_loading 参数决定：
    ├─ True → 过滤现有数据，只保留新数据
    └─ False → 保留全部数据（后续覆盖旧记录）
  ↓
处理新数据列表（存储到 data 表）
```

## 1.6 数据库表结构

**.add** 将数据存储到 `data` 表，关键字段如下：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `id` | UUID | 数据的唯一标识（由哈希 + 用户 ID 确定） |
| `dataset_id` | UUID | 关联的数据集 ID |
| `name` | STRING | 原始文件名 |
| `file_path` | STRING | 文件的存储路径（本地或 S3） |
| `file_extension` | STRING | 文件扩展名（pdf, docx, csv 等） |
| `mime_type` | STRING | MIME 类型（application/pdf 等） |
| `file_size` | INTEGER | 原始文件大小（字节） |
| `content_hash` | STRING | SHA-256 哈希，用于去重 |
| `raw_metadata` | JSON | 从原始文件提取的元数据（作者、日期等） |
| `processed_metadata` | JSON | 处理后的元数据（分块数、向量维度等） |
| `loader_name` | STRING | 使用的加载器类名（PDFLoader, DOCXLoader） |
| `node_sets` | JSON | 关联的节点集标签列表 |
| `created_at` | TIMESTAMP | 数据创建时间 |
| `updated_at` | TIMESTAMP | 数据更新时间 |
| `user_id` | UUID | 数据所属用户 |


# 二、.cognify 功能实现详解

## 2.1 功能简述

**.cognify** 是 Cognee 的核心，将加入的原始数据转化为**结构化的知识图谱**。这是最复杂的处理阶段，涉及文档分类、分块、实体抽取、关系发现、图构建、摘要生成等七个环节。

## 2.2 核心处理管道（七步骤）

```python
tasks = [
    Task(classify_documents),
    Task(check_permissions_on_dataset, user=user, permissions=["write"]),
    Task(extract_chunks_from_documents, max_chunk_size=chunk_size or get_max_chunk_tokens(), chunker=chunker),
    Task(extract_graph_from_data, graph_model=graph_model, config=config, custom_prompt=custom_prompt, task_config={"batch_size": chunks_per_batch}),
    Task(summarize_text, task_config={"batch_size": chunks_per_batch}),
    Task(add_data_points, task_config={"batch_size": chunks_per_batch}),
]
```

**Step 1：classify_documents** - 文档类型识别
- 根据文件扩展名（.pdf, .docx, .csv 等）自动判断文档类型
- 为每个文档创建相应的类型标签（PdfDocument、TextDocument、ImageDocument、AudioDocument 等）
- 输出：标记了类型的文档对象，供后续分块和提取步骤使用

**Step 2：check_permissions_on_dataset** - 权限检查
- 验证用户对数据集的 write 权限
- 多租户场景下的数据隔离保障

**Step 3：extract_chunks_from_documents** - 文本分块
- 将文档分割为语义相关的文本块
- 使用 TextChunker（段落级）或 LangchainChunker（字符级）
- chunk_size 自动计算，权衡粒度与覆盖范围

### 2.2.1 文本分块的实现

文本分块通过文档对象的 `read()` 方法异步生成文本块（DocumentChunk）。每个块包含：
- 文本内容 (`text`)
- 块大小 (`chunk_size`) - token 数量
- 所属文档信息 (`document_id`)
- 关联的节点集 (`belongs_to_set`)

```python
async def extract_chunks_from_documents(
    documents: list[Document],
    max_chunk_size: int,
    chunker: Chunker = TextChunker,
) -> AsyncGenerator:
    """根据最大块大小和分块器对文档进行分块"""
    for document in documents:
        document_token_count = 0
        # 使用文档的 read 方法异步生成分块
        async for document_chunk in document.read(
            max_chunk_size=max_chunk_size, chunker_cls=chunker
        ):
            document_token_count += document_chunk.chunk_size
            document_chunk.belongs_to_set = document.belongs_to_set
            yield document_chunk
        
        # 更新文档的总 token 数
        await update_document_token_count(document.id, document_token_count)
```

**Step 4：extract_graph_from_data** - 知识图谱提取
- LLM 调用，逐块提取实体、关系、属性
- 可自定义 graph_model（数据结构）和 custom_prompt（提示词）
- 批处理以控制 LLM 调用成本

### 2.2.2 知识图谱提取的实现

知识图谱提取通过 LLM（调用 `extract_content_graph` 函数）从每个文本块并行提取结构化知识，然后与规范化解析器集成验证和集成。

```python
async def extract_graph_from_data(
    data_chunks: List[DocumentChunk],
    graph_model: Type[BaseModel],
    config: Config = None,
    custom_prompt: Optional[str] = None,
    **kwargs,
) -> List[DocumentChunk]:
    """从文本块中提取和整合知识图谱"""
    
    # 并行调用 LLM 提取每个块的图谱（使用 asyncio.gather）
    chunk_graphs = await asyncio.gather(
        *[
            extract_content_graph(chunk.text, graph_model, custom_prompt=custom_prompt, **kwargs)
            for chunk in data_chunks
        ]
    )

    # 若使用默认 KnowledgeGraph 模型，过滤无效的边（源/目标节点不存在）
    if graph_model == KnowledgeGraph:
        for graph in chunk_graphs:
            valid_node_ids = {node.id for node in graph.nodes}
            graph.edges = [
                edge for edge in graph.edges
                if edge.source_node_id in valid_node_ids and edge.target_node_id in valid_node_ids
            ]

    # 获取规范化解析器（从配置或使用默认）
    if config is None:
        config = {
            "ontology_config": {"ontology_resolver": get_default_ontology_resolver()}
        }
    
    ontology_resolver = config["ontology_config"]["ontology_resolver"]

    # 与规范化解析器集成并存储到图数据库
    return await integrate_chunk_graphs(data_chunks, chunk_graphs, graph_model, ontology_resolver)
```

**关键流程**：
1. **并行 LLM 提取** - 对所有块同时调用 LLM，提高效率
2. **边过滤** - 移除指向不存在节点的边，保证图的完整性
3. **规范化验证** - 根据预定义规则验证和扩展提取的实体和关系
4. **数据存储** - 将节点存储到图数据库，将边存储到关系数据库

**Step 5：summarize_text** - 摘要生成
- 为每个文档或章节生成层级摘要
- 支持快速浏览与精确查询

### 2.2.3 摘要生成的实现 

摘要生成通过 LLM 为每个文本块生成简洁的摘要，支持自定义摘要模型。

```python
async def summarize_text(
    data_chunks: list[DocumentChunk], summarization_model: Type[BaseModel] = None
):
    """为每个文本块生成摘要"""
    
    if len(data_chunks) == 0:
        return data_chunks

    # 若未指定摘要模型，使用配置中的默认模型
    if summarization_model is None:
        cognee_config = get_cognify_config()
        summarization_model = cognee_config.summarization_model

    # 并行调用 LLM 为所有块生成摘要
    chunk_summaries = await asyncio.gather(
        *[extract_summary(chunk.text, summarization_model) for chunk in data_chunks]
    )

    # 创建 TextSummary 对象，关联到原块
    summaries = [
        TextSummary(
            id=uuid5(chunk.id, "TextSummary"),
            made_from=chunk,  # 引用原文本块
            text=chunk_summaries[chunk_index].summary,
        )
        for (chunk_index, chunk) in enumerate(data_chunks)
    ]

    return summaries
```

**关键特性**：
- **uuid5 生成** - 使用块 ID 生成确定性摘要 ID（保证幂等性）
- **异步并行处理** - 对所有块同时生成摘要，提高速度
- **与原块关联** - 每个摘要都保持对原文本块的引用，便于溯源

**Step 6：add_data_points** - 数据存储
- 将提取的图数据（节点、边）写入图数据库
- 将向量嵌入写入向量数据库

## 2.3 任务编排与流水线

```python
async def cognify(
    datasets: Union[str, list[str], list[UUID]] = None,
    user: User = None,
    graph_model: BaseModel = KnowledgeGraph,
    chunker=TextChunker,
    chunk_size: int = None,
    chunks_per_batch: int = None,
    config: Config = None,
    run_in_background: bool = False,
    temporal_cognify: bool = False,
    ...
) -> Union[dict, list[PipelineRunInfo]]:
```

**两种处理流程**：
1. **标准流程**（temporal_cognify=False）：默认的七步骤
2. **时序流程**（temporal_cognify=True）：专门提取时间事件的变体

**后台处理与非阻塞执行**：
```python
if run_in_background:
    # 异步启动，立即返回
    pipeline_executor_func = get_pipeline_executor(run_in_background=True)
    return await pipeline_executor_func(...)
else:
    # 阻塞等待完成
    pipeline_executor_func = get_pipeline_executor(run_in_background=False)
    return await pipeline_executor_func(...)
```

## 2.4 Step 3：文本分块实现详解

文本分块是 `.cognify` 的第一个处理环节，将文档转化为向量化与 LLM 处理的基本单位。

**核心机制**：
- 使用异步生成器模式逐块产生 DocumentChunk 对象
- 支持 TextChunker（段落级，语义完整）和 LangchainChunker（字符级，精细控制）
- 自动计算 chunk_size（token 数量）并在数据库中记录文档的总 token 数

**实现代码**：

```python
async def extract_chunks_from_documents(
    documents: list[Document],
    max_chunk_size: int,
    chunker: Chunker = TextChunker,
) -> AsyncGenerator:
    """根据最大块大小和分块器对文档进行分块"""
    for document in documents:
        document_token_count = 0
        # 使用文档的 read 方法异步生成分块
        async for document_chunk in document.read(
            max_chunk_size=max_chunk_size, chunker_cls=chunker
        ):
            document_token_count += document_chunk.chunk_size
            document_chunk.belongs_to_set = document.belongs_to_set
            yield document_chunk
        
        # 更新文档的总 token 数
        await update_document_token_count(document.id, document_token_count)
```

**设计亮点**：
- **异步生成器**：避免一次性加载全部块到内存，流式处理大文档
- **token 计数**：记录每个块与整个文档的 token 数，用于后续成本估算和参数优化
- **元数据传递**：块继承文档的 `belongs_to_set` 关联，保持溯源链

**两种切分策略**：

| 策略 | 实现类 | 切分粒度 | 适用场景 | 优点 | 缺点 |
|------|--------|---------|---------|------|------|
| **段落级** | TextChunker | 按段落/章节边界 | 书籍、论文、新闻 | 语义连贯，上下文完整 | 可能超出 max_chunk_size |
| **字符级** | LangchainChunker | 按字符数精确分割 | 代码、日志、结构化数据 | 严格控制大小，可重叠 | 可能在语义中点断裂 |

**参数指南**：
- `max_chunk_size`：通常 512-8192 tokens，取决于 embedding 模型和 LLM 的 context window
  - 512-1024 tokens：适合轻量级应用、实时响应
  - 2048-4096 tokens：平衡精度与成本（推荐范围）
  - 8192+ tokens：深度理解，适合复杂领域知识抽取
- `chunker`：
  - TextChunker 适合语义完整的内容（长文本、书籍），保留段落/章节结构
  - LangchainChunker 适合精细控制（代码、日志），支持 overlap 参数防止信息丢失

## 2.5 Step 4：知识图谱提取实现详解

知识图谱提取是 `.cognify` 的核心，通过 LLM 从文本块中抽取结构化的实体、关系、属性，构建知识网络。

**核心机制**：
- 使用 `asyncio.gather()` 并行调用 LLM 处理所有文本块（提高吞吐）
- 自动过滤无效的边（源/目标节点不存在）
- 与规范化解析器集成，对提取的知识进行验证和规范化
- 最终通过 `add_data_points()` 将节点和边分别存储到图数据库和关系数据库

**实现代码**：

```python
async def extract_graph_from_data(
    data_chunks: List[DocumentChunk],
    graph_model: Type[BaseModel],
    config: Config = None,
    custom_prompt: Optional[str] = None,
    **kwargs,
) -> List[DocumentChunk]:
    """从文本块中提取和整合知识图谱"""
    
    # 并行调用 LLM 提取每个块的图谱（使用 asyncio.gather）
    chunk_graphs = await asyncio.gather(
        *[
            extract_content_graph(chunk.text, graph_model, custom_prompt=custom_prompt, **kwargs)
            for chunk in data_chunks
        ]
    )

    # 若使用默认 KnowledgeGraph 模型，过滤无效的边（源/目标节点不存在）
    if graph_model == KnowledgeGraph:
        for graph in chunk_graphs:
            valid_node_ids = {node.id for node in graph.nodes}
            graph.edges = [
                edge for edge in graph.edges
                if edge.source_node_id in valid_node_ids and edge.target_node_id in valid_node_ids
            ]

    # 获取规范化引擎（从配置或使用默认）
    if config is None:
        config = {
            "ontology_config": {"ontology_resolver": get_default_ontology_resolver()}
        }
    
    ontology_resolver = config["ontology_config"]["ontology_resolver"]

    # 与规范化引擎集成并存储到图数据库
    return await integrate_chunk_graphs(data_chunks, chunk_graphs, graph_model, ontology_resolver)
```

**设计亮点**：
- **asyncio.gather() 并行处理**：将 N 个块的 LLM 调用从顺序(N×耗时) 优化为并行(耗时)
- **边过滤**：确保图的完整性，避免孤立的边或指向不存在节点的错误边
- **知识规范化**：根据预定义的规则对提取的实体和关系进行规范化，增强一致性
- **下游存储**：`add_data_points()` 自动处理数据持久化，用户不需要手动存储

### 2.5.1 边过滤策略详解

LLM 在提取图时可能产生以下问题：
1. **孤立边**：边的源或目标节点在提取过程中未被识别为实体
2. **跨块引用**：边指向在当前块中不存在、但在其他块中出现的节点
3. **幻觉问题**：LLM 虚构的不存在的实体关联

Cognee 采用**图级边验证机制**：

```python
# 当使用默认 KnowledgeGraph 模型时，执行严格的边过滤
if graph_model == KnowledgeGraph:
    for graph in chunk_graphs:
        # Step 1: 收集当前块中所有有效的节点 ID
        valid_node_ids = {node.id for node in graph.nodes}
        
        # Step 2: 过滤边，保留源和目标都在有效节点集中的边
        graph.edges = [
            edge for edge in graph.edges
            if edge.source_node_id in valid_node_ids 
            and edge.target_node_id in valid_node_ids
        ]
```

**过滤流程**：

```
LLM 提取的原始图
  ├─ 节点：[实体A, 实体B, 实体C, ...]
  ├─ 边：[
  │    Edge(A→B, "关联"),      ✅ 保留（两个端点都在节点集）
  │    Edge(A→D, "引用"),      ❌ 删除（D 不在当前块节点集）
  │    Edge(E→B, "依赖"),      ❌ 删除（E 不在当前块节点集）
  │    Edge(B→C, "继承"),      ✅ 保留（两个端点都在节点集）
  │  ]

验证后的清洁图
  ├─ 节点：[实体A, 实体B, 实体C]
  └─ 边：[Edge(A→B), Edge(B→C)]  ← 只保留两端都有效的边
```

**自定义图模型的处理**：

如果使用自定义图模型（非 KnowledgeGraph），用户需自行决定是否进行边过滤：

```python
# 示例：自定义的论文知识图
class ResearchGraph(DataPoint):
    papers: List[Paper]
    citations: List[Citation]  # 可能包含指向论文库外的引用
    methodologies: List[str]

# 对于自定义模型，不进行自动过滤，保留原始 LLM 输出
# 用户可在后续处理中自行验证或使用规范化规则进行过滤
await cognee.cognify(
    datasets=["research"],
    graph_model=ResearchGraph,
    custom_prompt="Extract research citations and methodologies..."
)
```

**过滤的副作用与权衡**：

| 场景 | 自动过滤 | 影响 |
|------|---------|------|
| **单块处理** | 有效过滤孤立边 | 图局部完整，缺失跨块关联 |
| **规范化前** | 必须执行 | 防止规范化验证失败 |
| **自定义模型** | 不执行 | 保留完整输出，用户决策 |
| **大规模图** | 高效（集合查询） | O(n) 复杂度，可接受 |

**最佳实践**：

1. **段落级文本**：使用 TextChunker 时，单块内提取的图通常完整，边过滤有效
2. **长文档**：对于超过 4096 tokens 的大文档，分块后可能出现跨块引用，需在 `integrate_chunk_graphs()` 中合并去重
3. **规范化集成**：过滤后的图更易通过规范化验证，减少冲突

### 2.5.2 知识规范化与验证

LLM 提取的知识可能存在重复或不规范的问题。Cognee 提供了一个**知识规范化引擎**，通过预定义的规则来清理和统一知识。

规范化包括三个步骤：
1. **实体名称统一**：将相同概念的不同表述统一为标准名称（如"CEO"、"Chief Executive Officer"、"首席执行官"都规范化为"CEO"）
2. **关系类型验证**：检查提取的关系是否合理（如是否允许"CEO 驾驶 汽车"这种关系）
3. **关系名称统一**：将同一个关系的不同表述统一为标准名称（如"manages"和"管理"都规范化为统一形式）

```python
# 知识规范化的核心流程 - 完全基于预定义的规则表
resolver = config["ontology_config"]["ontology_resolver"]

# 对每个提取的块级图进行规范化
validated_chunk_graphs = []
for chunk_idx, chunk_graph in enumerate(chunk_graphs):
    # Step 1: 实体名称统一 - 查找别名映射表
    for node in chunk_graph.nodes:
        # 在预定义的别名映射表中查找标准名称
        canonical_entity = resolver.resolve_entity(
            node.name, 
            node.type if hasattr(node, 'type') else None
        )
        if canonical_entity:
            node.name = canonical_entity.name
            node.type = canonical_entity.type
    
    # Step 2: 关系合理性验证 - 查找规则限制表
    valid_edges = []
    for edge in chunk_graph.edges:
        # 查询规则表：这两个实体类型间是否允许此关系
        is_valid = resolver.validate_relationship(
            source_type=chunk_graph.nodes_by_id.get(edge.source_node_id).type,
            target_type=chunk_graph.nodes_by_id.get(edge.target_node_id).type,
            relationship_type=edge.type
        )
        if is_valid:
            valid_edges.append(edge)
        else:
            # 不合理的关系删除
            logger.warning(
                f"关系 {edge.type} 在 {edge.source_node_id} 和 "
                f"{edge.target_node_id} 之间不被允许"
            )
    
    chunk_graph.edges = valid_edges
    
    # Step 3: 关系名称统一 - 查找关系别名映射表
    for edge in chunk_graph.edges:
        # 在预定义的关系名称映射表中查找标准名称
        canonical_rel_type = resolver.resolve_relationship_type(edge.type)
        if canonical_rel_type:
            edge.type = canonical_rel_type
    
    validated_chunk_graphs.append(chunk_graph)
```

**规范化效果示例**：

```
原始提取结果（杂乱）：
  ├─ 实体：[CEO, 首席执行官, CEO_PERSON, 中文CEO]  ← 同一个概念，多种写法
  ├─ 关系：[
  │    (CEO, manages, Company)        
  │    (CEO, drives_car, Vehicle)     ← 不合理的关系
  │    (首席执行官, leads, Department)  
  │  ]

规范化后（统一、清洁）：
  ├─ 实体：[CEO, CEO, CEO, CEO]  ← 全部统一为标准名称
  ├─ 去重后：[CEO]               ← 只保留一份
  └─ 关系：[
       (CEO, 管理, Company),      ← 只保留合理的关系
       (CEO, 领导, Department)    ← 关系名称也统一了
     ]
```

**规范化规则从哪里来**：

Cognee 使用的规则有三种来源，按推荐优先级排列：

| 规则来源 | 说明 | 适用场景 | 推荐度 |
|---------|------|---------|--------|
| **用户上传规则文件** | 用户提供 RDF/OWL 格式的规则文件（标准行业格式） | 生产环境、需要标准化管理 | ⭐⭐⭐⭐⭐ |
| **程序定义规则** | 在代码中直接定义规则字典 | 开发测试、快速迭代 | ⭐⭐⭐ |
| **不使用规则** | Cognee 默认行为，只做基础清洁，不规范化 | 不需要实体/关系统一 | ⭐⭐ |

```python
# 方式 1：使用规则文件（推荐用于生产环境）
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver

await cognee.cognify(
    datasets=["companies"],
    config={
        "ontology_config": {
            "ontology_resolver": RDFLibOntologyResolver(
                ontology_file="company_rules.owl"  # 用户提供的规则文件
            )
        }
    }
)

# 方式 2：在代码中定义规则（用于开发测试）
rules = {
    # 实体名称统一规则
    "entities": [
        {
            "standard_name": "CEO",
            "aliases": ["Chief Executive Officer", "首席执行官", "C级高管"],
            "type": "Person",
        },
        {
            "standard_name": "Company",
            "aliases": ["Organization", "企业", "公司"],
            "type": "Organization",
        },
    ],
    # 关系合理性规则
    "relationships": [
        {
            "name": "manages",           # 关系类型
            "allowed_from": "Person",    # 只能从 Person 发出
            "allowed_to": ["Organization", "Department"],  # 只能指向这些实体类型
            "standard_name": "管理",     # 标准名称（中文）
        },
        {
            "name": "leads",
            "allowed_from": "Person",
            "allowed_to": ["Team", "Department"],
            "standard_name": "领导",
        },
        # 注意：不在这个列表里的关系（如 drives_car）会被删除
    ]
}

await cognee.cognify(
    datasets=["companies"],
    config={
        "ontology_config": {
            "ontology_resolver": OntologyResolver.from_dict(rules)
        }
    }
)

# 方式 3：不使用规则（仅做基础图清洁）
from cognee.modules.ontology.get_default_ontology_resolver import get_default_ontology_resolver

# 使用 Cognee 默认的空规则
resolver = get_default_ontology_resolver()
# 此时只能做基础的图完整性检查（删除孤立边）
# 不能进行实体统一、关系验证等规范化操作
```

**何时需要规范化规则**：

```
使用场景选择
├─ 简单应用（不需要规范化）
│  └─ 不使用规则
│     优点：快速、简单，无需维护规则
│     缺点：知识会有重复和不一致
│
├─ 中等应用（快速开发测试）
│  └─ 在代码中定义规则（方式 2）
│     优点：灵活快速，改规则不需要重启
│     缺点：难以在团队间共享和复用
│
└─ 生产应用（需要标准化）
   └─ 上传规则文件（方式 1）
      优点：标准格式、易于维护、可版本控制
      缺点：需要预先定义规则
```

**最佳实践**：

1. **开发阶段**：用方式 2（代码定义）快速迭代
2. **测试通过**：将规则转换为 RDF/OWL 文件（方式 1）
3. **生产上线**：从文件加载规则，确保可维护性

**高级定制**：
```python
# 自定义图模型
class ScientificPaper(DataPoint):
    title: str
    authors: List[str]
    methodology: str
    findings: List[str]

await cognee.cognify(
    datasets=["papers"],
    graph_model=ScientificPaper,  # 领域特定的结构
    custom_prompt="Extract entities specific to scientific research..."
)

# 规范化解析器集成
config = {
    "ontology_config": {
        "ontology_resolver": get_ontology_resolver_from_env(...)
    }
}
```

## 2.6 Step 5：摘要生成实现详解

摘要生成为每个文本块创建简洁的摘要，支持快速浏览和高层次的上下文理解。

**核心机制**：
- 使用 `asyncio.gather()` 并行调用 LLM 为所有文本块生成摘要
- 生成确定性的 TextSummary ID（基于块 ID 的 uuid5），保证幂等性
- 保持摘要与原块的关联（`made_from` 字段），便于溯源和精确查询

**实现代码**：

```python
async def summarize_text(
    data_chunks: list[DocumentChunk], summarization_model: Type[BaseModel] = None
):
    """为每个文本块生成摘要"""
    
    if len(data_chunks) == 0:
        return data_chunks

    # 若未指定摘要模型，使用配置中的默认模型
    if summarization_model is None:
        cognee_config = get_cognify_config()
        summarization_model = cognee_config.summarization_model

    # 并行调用 LLM 为所有块生成摘要
    chunk_summaries = await asyncio.gather(
        *[extract_summary(chunk.text, summarization_model) for chunk in data_chunks]
    )

    # 创建 TextSummary 对象，关联到原块
    summaries = [
        TextSummary(
            id=uuid5(chunk.id, "TextSummary"),
            made_from=chunk,  # 引用原文本块
            text=chunk_summaries[chunk_index].summary,
        )
        for (chunk_index, chunk) in enumerate(data_chunks)
    ]

    return summaries
```

**设计亮点**：
- **异步并行生成**：同时为多个块生成摘要，避免顺序调用的性能瓶颈
- **确定性 ID 生成**：使用 `uuid5(chunk.id, "TextSummary")` 生成摘要 ID，确保同样的块总是生成相同的 ID（幂等性）
- **双向关联**：摘要通过 `made_from` 保留对原块的引用，支持精确检索和溯源

**输出**：
- 摘要与原块都存储在数据库中，`.search` 可独立返回摘要或快速概览


# 三、.search 功能实现详解

## 3.1 功能简述

**.search** 是知识的查询利用阶段。Cognee 的搜索不同于传统关键词或向量搜索，而是**向量相似度 + 图遍历的融合**，为 LLM 提供精准的上下文。

## 3.2 六大搜索类型

```python
async def search(
    query_text: str,
    query_type: SearchType = SearchType.GRAPH_COMPLETION,
    user: Optional[User] = None,
    datasets: Optional[Union[list[str], str]] = None,
    top_k: int = 10,
    ...
) -> Union[List[SearchResult], CombinedSearchResult]:
```

**GRAPH_COMPLETION**（默认推荐）
- 机制：向量检索 → 图遍历 → LLM 推理
- 返回：自然语言回答
- 最聪慧，但成本最高

**工作流**：
1. **向量检索**：编码查询，在向量库中找到 top-k 相关块
2. **获取节点**：从这些块的文本中提取实体作为图遍历的种子节点（而不是从随机节点开始）
3. **图遍历**：从种子节点出发，BFS 深度 2-3 层获取结构化关系链
4. **融合内容**：融合向量块内容和图结构关系，构成统一上下文。将**文本内容**和**结构知识**输入到 LLM 进行理解融合，若发生冲突，以文本内容为准。
5. **LLM 推理**：基于融合上下文生成自然语言回答

**RAG_COMPLETION**
- 机制：向量检索块 → LLM 组织
- 返回：基于文本块的回答
- 介于图和块之间

**CHUNKS**
- 机制：纯向量相似度
- 返回：相关文本片段列表
- 最快，无 LLM 调用

**SUMMARIES**
- 机制：返回预生成的摘要
- 返回：层级摘要（高层 → 详细）
- 快速概览

**CODE & CYPHER**
- CODE：代码专用搜索
- CYPHER：直接图数据库查询

## 3.3 权限与数据集管理

```python
# 数据集解析：支持按名称或 UUID
if datasets is not None and [all(isinstance(dataset, str) for dataset in datasets)]:
    # 按名称查询，需要验证用户的 read 权限
    datasets = await get_authorized_existing_datasets(datasets, "read", user)
    datasets = [dataset.id for dataset in datasets]
    if not datasets:
        raise DatasetNotFoundError(message="No datasets found.")
```

## 3.4 结果处理

```python
filtered_search_results = await search_function(
    query_text=query_text,
    query_type=query_type,
    dataset_ids=dataset_ids if dataset_ids else datasets,
    user=user,
    top_k=top_k,
    ...
)
return filtered_search_results
```

**返回值**：
- 列表形式：多个 SearchResult 对象
- 或聚合形式：CombinedSearchResult（多种搜索类型的组合）


# 四、.memify 功能实现详解

## 4.1 功能简述

**.memify** 是语义图增强阶段（当前版本未完全实现）。其目标是进一步优化已构建的知识图谱，添加高阶的语义关系与规则关联。

## 4.2 任务的灵活编排

```python
async def memify(
    extraction_tasks: Union[List[Task], List[str]] = None,
    enrichment_tasks: Union[List[Task], List[str]] = None,
    data: Optional[Any] = None,
    dataset: Union[str, UUID] = "main_dataset",
    user: User = None,
    node_type: Optional[Type] = NodeSet,
    node_name: Optional[List[str]] = None,
    ...
):
```

**两层任务设计**：
1. **extraction_tasks**：从图或外部数据提取子图/特征
2. **enrichment_tasks**：对提取结果进行增强与关联

# 尾言：四大功能的协作体系

通过本文的深度剖析，我们已经掌握了 Cognee 四大核心功能的实现机制：

## 功能概览回顾

- **.add**：数据多源适配与智能去重，确保摄入的数据清洁、可溯源
- **.cognify**：七步骤知识图谱构建管道，通过 LLM 驱动将文本转化为结构化知识
- **.search**：融合向量相似度与图遍历的混合查询，为 LLM 提供精准上下文
- **.memify**：灵活的任务编排框架，支持图的进一步语义增强（规划中）

## 内在的统一架构

这四个函数看似独立，实则通过一个**统一的 Pipeline 机制**协调运作：

```
输入 → [Task 组织] → [并行/顺序执行] → [权限验证] → [数据持久化] → 输出
        └─ 每个函数都是任务序列 ┘
```

- 每个函数都由一列 `Task` 对象组成
- 任务支持异步并行处理（如 LLM 调用、向量计算）
- 支持后台执行与流式处理
- 统一的错误处理与监控

## 后续深入方向

本文专注于**各功能内部的实现细节**。下一篇《**Cognee Pipeline 机制与工作流**》将：
- 深入 Pipeline 执行引擎的设计
- 揭示任务间的数据流与协调机制
- 探讨错误恢复、可观测性等深层议题
- 展示如何自定义和扩展 Pipeline

完成这两篇文章的学习，你将获得对 Cognee 架构的**全面而深入的理解**。