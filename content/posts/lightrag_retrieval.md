---
title: "【解密源码】 轻量 GrapghRAG - LightRAG 检索工程实践"
date: 2025-12-08T11:33:10+08:00
draft: false
tags: ["源码","技术","RAG"]
categories: ["LightRAG"]
---

# 引言

LightRAG 是一个轻量级的 RAG（Retrieval-Augmented Generation）框架，通过知识图谱和向量数据库的融合检索，提供更精准的信息检索和生成能力。相比传统 RAG 框架，LightRAG 的创新点在于：

1. **混合检索架构**：同时利用知识图谱（图结构信息）和向量数据库（语义相似度），弥补两者的不足
2. **灵活的查询模式**：提供 local、global、hybrid、naive、mix、bypass 六种查询模式，满足不同检索场景
3. **智能关键词提取**：使用 LLM 自动或手动指定关键词，支持高层（主题级）和低层（细粒度）双层次关键词
4. **精细的 Token 管理**：对实体、关系、文本块分别进行 Token 限制，防止 Prompt 过长
5. **完整的数据可追溯性**：所有检索结果都包含引用信息，支持调试和评估

本文深度解析 LightRAG 的三个检索接口实现原理，帮助开发者更好地理解和使用这个框架。

# 省流版

**LightRAG 核心架构**：

- **三个检索接口** → `/query`（标准问答）、`/query/stream`（流式问答）、`/query/data`（数据检索）
- **六种查询模式** → local（实体查询）、global（关系查询）、hybrid（混合）、naive（向量）、mix（推荐）、bypass（直接LLM）
- **核心检索流程** → 关键词提取 → 图谱/向量检索 → 结果合并 → Token 截断 → 上下文构建 → LLM 生成
- **Key Features**：双层关键词、Round-robin 合并、向量+权重两种 chunk 选择策略、灵活的 Token 控制

**快速对比**：

| 接口 | 返回内容 | 用途 | 延迟 |
|------|--------|------|------|
| `/query` | 答案 + 引用 | 标准问答 | 低 |
| `/query/stream` | 流式答案 + 引用 | 实时交互 | 中 |
| `/query/data` | 实体/关系/chunks | 调试分析 | 最低 |

# 手撕版

LightRAG 提供三种检索接口：
1. query() —— 标准 RAG 问答接口（含 LLM 生成）：用户查询 → 检索 → LLM 生成最终回答，LightRAG 检索最常用的接口。
2. query_stream() —— 流式标准 RAG 问答接口（含 LLM 生成）： 用户查询 → 检索 → 流式 LLM 生成回答，适用于长文本生成场景。
3. query_data() —— 结构化数据检索接口（不含 LLM 生成）：只返回 “检索层的数据”，不进行生成，用于调试和分析。

```python
@router.post("/query")
async def query_text(request: QueryRequest):
    """Comprehensive RAG query endpoint with non-streaming response. Parameter "stream" is ignored."""

@router.post("/query/stream")
async def query_text_stream(request: QueryRequest): 
    """ Advanced RAG query endpoint with flexible streaming response."""

@router.post("/query/data")
async def query_data(request: QueryRequest):
    """ Advanced data retrieval endpoint for structured RAG analysis."""
```
QueryRequest 是检索接口的请求体，包含以下字段：
| 参数名                       | 类型                                                     | 默认值   | 约束           | 说明                         | 使用场景               |
| ------------------------- | ------------------------------------------------------ | ----- | ------------ | -------------------------- | ------------------ |
| **query**                 | str                                                    | 必填    | min_length=3 | 查询内容（搜索问题或文本）              | 用户的核心查询            |
| **mode**                  | enum("local","global","hybrid","naive","mix","bypass") | mix   | –            | 检索方式：向量/图谱/混合              | 控制 RAG 工作模式        |
| **only_need_context**     | bool                                                   | None  | –            | 只返回检索到的上下文，不生成回答           | 调试/评估检索质量          |
| **only_need_prompt**      | bool                                                   | None  | –            | 只返回构造好的 prompt，不调用 LLM     | Prompt 工程、系统调试     |
| **response_type**         | str                                                    | None  | min_length=1 | 定义回答格式（如 多段、单段、要点）         | 强制 LLM 输出指定格式      |
| **top_k**                 | int                                                    | None  | ≥1           | 返回前 K 个实体/关系（根据模式决定）       | 控制知识图谱检索规模         |
| **chunk_top_k**           | int                                                    | None  | ≥1           | 返回前 K 个向量 chunk，并做 rerank  | 控制向量检索规模           |
| **max_entity_tokens**     | int                                                    | None  | ≥1           | 实体部分可使用的最大 tokens          | 统一 token 控制系统      |
| **max_relation_tokens**   | int                                                    | None  | ≥1           | 关系部分可使用的最大 tokens          | 控制图谱关系上下文          |
| **max_total_tokens**      | int                                                    | None  | ≥1           | 整体（实体+关系+chunks）可用总 tokens | 防止 prompt 过长       |
| **hl_keywords**           | list[str]                                              | []    | –            | 高层关键字（主题级）用户可自定义           | 无需 LLM 自动提取关键词     |
| **ll_keywords**           | list[str]                                              | []    | –            | 低层关键字（细粒度）用户可自定义           | 指导精细检索             |
| **conversation_history**  | List[dict]                                             | None  | role 必填      | 多轮对话历史（含角色与内容）             | 长对话场景保持上下文         |
| **user_prompt**           | str                                                    | None  | –            | 用户自定义的 prompt（覆盖模板）        | 高级用户指定 Prompt      |
| **enable_rerank**         | bool                                                   | None  | –            | 是否启用 rerank（默认开启）          | 检索优化，提高精度          |
| **include_references**    | bool                                                   | True  | –            | 返回引用（文档路径、文件名等）            | 用于可追溯性、显示引用        |
| **include_chunk_content** | bool                                                   | False | –            | 在引用中附带 chunk 的原文           | Debug / OCR / 评测需要 |
| **stream**                | bool                                                   | True  | –            | 是否启用流式输出（仅影响 stream 接口）    | ChatGPT 类实时输出      |

**开发者通过配置不同的 model 参数，检索不同的信息。**
- 使用知识图谱 + 向量检索："local", "global", "hybrid", "mix"
- 使用向量检索："naive"
- 不使用检索，只通过 LLM 生成回答："bypass"

## 1. query（知识图谱 + 向量检索）
### 1.1 确认 LLM
确认用哪个 LLM 推理函数, 优先使用 query_param.model_func, 否则使用默认配置 global_config["llm_model_func"]，默认使用 LLM 为 gpt_4o_mini。
```python
if query_param.model_func:
    use_model_func = query_param.model_func
else:
    use_model_func = global_config["llm_model_func"]
    # Apply higher priority (5) to query relation LLM function
    use_model_func = partial(use_model_func, _priority=5)
```
### 1.2 使用 LLM 提取关键词（关键步骤）
从 query 中使用 LLM 提取关键词，用于检索。 prompt 中提供了 few-shot 学习示例，引导 LLM 提取两种类型关键词：high_level_keywords（主题层关键词）和 low_level_keywords（实体层关键词）。
prompt 解析：
| **模块（Module）**                        | **作用（Purpose）**                                                   | **示例片段（Example Snippet）**                                                                                                                                                                                                                      |
| ------------------------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Role（角色定义）**                        | 明确模型身份，不回答问题、不解释、不推理，只做“关键词抽取器”，避免跑偏。                             | `You are an expert keyword extractor...`                                                                                                                                                                                                       |
| **Goal（任务目标）**                        | 定义任务：必须提取两种类型关键词，分别用于 RAG 的主题层召回与实体层召回。                           | `extract two distinct types of keywords: high_level_keywords, low_level_keywords`                                                                                                                                                              |
| **Instructions & Constraints（规则与限制）** | 通过强规则保证 LLM 输出可控、结构化、不产生幻觉，并能稳定被 JSON 解析。包含输出限制、来源限制、短语优先、边界情况处理。 | - `Your output MUST be a valid JSON object and nothing else.`<br>- `All keywords must be explicitly derived from the user query.`<br>- `Keywords should be concise words or meaningful phrases.`<br>- `For simple queries return empty lists.` |
| **Examples（示例对齐）**                    | 提供 few-shot 学习，让模型理解 JSON 结构、词语粒度、高/低层抽取方式；提高稳定性。                 | `{examples}`（外部注入的示例 JSON）                                                                                                                                                                                                                     |
few-shot 案例：
```
问题：“国际贸易如何影响全球经济的稳定性？”
输出：
{“高级关键词”：["国际贸易", "全球经济稳定", "经济影响"]
“低级关键词”：["贸易协定"， "关税"， "货币兑换"， "进口"， "出口"]}


问题： “森林砍伐会对生物多样性造成哪些环境影响？”
输出：
{“高级关键词”：["环境影响", "森林砍伐", "生物多样性丧失"]
“低级关键词”：["物种灭绝"， "栖息地破坏"， "碳排放"， "雨林"， "生态系统"]}


问题：“教育在减少贫困方面发挥着怎样的作用？”
输出：
{“高级关键词”：["教育", "减贫", "社会经济发展"]
“低级关键词”：["学校入学率", "识字率"， "职业培训"， "收入不平等"]}
```
最终会提取出 high_level_keywords 和 low_level_keywords 两组关键词。

### 1.3 关键词边界场景处理
针对不同模式下的边界场景处理：
- 当 low_level_keywords 为空时，且模式为 "local", "hybrid", "mix" 时，记录警告日志。
- 当 high_level_keywords 为空时，且模式为 "global", "hybrid", "mix" 时，记录警告日志。
- 当 high_level_keywords 和 low_level_keywords 均为空时：
  - 若 query 长度小于 50 个字符，强制将 low_level_keywords 设置为 query 本身，记录警告日志。
  - 否则，返回失败响应。
```python
if ll_keywords == [] and query_param.mode in ["local", "hybrid", "mix"]:
    logger.warning("low_level_keywords is empty")
if hl_keywords == [] and query_param.mode in ["global", "hybrid", "mix"]:
    logger.warning("high_level_keywords is empty")
if hl_keywords == [] and ll_keywords == []:
    if len(query) < 50:
        logger.warning(f"Forced low_level_keywords to origin query: {query}")
        ll_keywords = [query]
    else:
        return QueryResult(content=PROMPTS["fail_response"])
```
### 1.4 信息检索（关键步骤）
```python
context_result = await _build_query_context(...)
```
#### 1.4.1 检索基础信息
通过原始 query 和关键词组检索基础信息，基础信息包括实体信息，关系信息和向量信息。

**1）local 模式**

在 local 模式下，仅基于 low_level_keywords 进行检索。
```python
local_entities, local_relations = await _get_node_data(
    ll_keywords,
    knowledge_graph_inst,
    entities_vdb,
    query_param,
)
```
基于 low_level_keywords 在实体向量库里做检索（这里的 query 是入参命名），返回 top_k 个实体。
```python
results = await entities_vdb.query(query, top_k=query_param.top_k)
```
通过实体 id 从知识图谱中获取节点的基本信息和节点度数。节点度数表示实体在知识图谱中的连接数，一定程度上反应了节点的重要性和影响力。
```python
nodes_dict, degrees_dict = await asyncio.gather(
    knowledge_graph_inst.get_nodes_batch(node_ids),
    knowledge_graph_inst.node_degrees_batch(node_ids),
)
```
通过实体信息获取相关边，同时获取边的信息和度数。
```python
batch_edges_dict = await knowledge_graph_inst.get_nodes_edges_batch(node_names)
edge_data_dict, edge_degrees_dict = await asyncio.gather(
    knowledge_graph_inst.get_edges_batch(edge_pairs_dicts),
    knowledge_graph_inst.edge_degrees_batch(edge_pairs_tuples),
)
```
根据度数和权重等信息排序后输出节点信息和相关边信息 node_datas, use_relations。

**2）global 模式**

在 global 模式下，仅基于 high_level_keywords 进行检索。
```python
global_relations, global_entities = await _get_edge_data(
    hl_keywords,
    knowledge_graph_inst,
    relationships_vdb,
    query_param,
)
```
基于 low_level_keywords 在关系向量库里做检索，返回 top_k 个关系（边）。
```python
results = await relationships_vdb.query(keywords, top_k=query_param.top_k)
```
通过关系 id 从知识图谱中获取关系信息。
```python
edge_data_dict = await knowledge_graph_inst.get_edges_batch(edge_pairs_dicts)
```
通过关系信息获取相关实体。
```python
use_entities = await _find_most_related_entities_from_relationships(
    edge_datas,
    query_param,
    knowledge_graph_inst,
)
```
根据度数和权重等信息排序后输出边信息和相关节点信息 edge_datas, use_entities。

**3）hybrid 模式**

在 hybrid 模式下，同时基于 high_level_keywords 和 low_level_keywords 进行检索。相当于 local 模式 + global 模式。

**4）mix 模式**

在 mix 模式下，hybrid 模式检索 + 对原始 query 直接进行向量检索。
```python
vector_chunks = await _get_vector_context(
    query,
    chunks_vdb,
    query_param,
    query_embedding,
)
```

#### 1.4.2 Round-robin 交错合并信息
**1）合并实体**
先提取本地实体信息加入实体列表，然后提取全局实体加入实体列表。
```python
max_len = max(len(local_entities), len(global_entities))
for i in range(max_len):
    # First from local
    if i < len(local_entities):
        entity = local_entities[i]
        entity_name = entity.get("entity_name")
        if entity_name and entity_name not in seen_entities:
            final_entities.append(entity)
            seen_entities.add(entity_name)

    # Then from global
    if i < len(global_entities):
        entity = global_entities[i]
        entity_name = entity.get("entity_name")
        if entity_name and entity_name not in seen_entities:
            final_entities.append(entity)
            seen_entities.add(entity_name)
```

**2）合并关系**
与实体合并策略一致，先处理本地后处理全局。
```python
max_len = max(len(local_relations), len(global_relations))
for i in range(max_len):
    # First from local
    if i < len(local_relations):
        relation = local_relations[i]
        # Build relation unique identifier
        if "src_tgt" in relation:
            rel_key = tuple(sorted(relation["src_tgt"]))
        else:
            rel_key = tuple(
                sorted([relation.get("src_id"), relation.get("tgt_id")])
            )

        if rel_key not in seen_relations:
            final_relations.append(relation)
            seen_relations.add(rel_key)

    # Then from global
    if i < len(global_relations):
        relation = global_relations[i]
        # Build relation unique identifier
        if "src_tgt" in relation:
            rel_key = tuple(sorted(relation["src_tgt"]))
        else:
            rel_key = tuple(
                sorted([relation.get("src_id"), relation.get("tgt_id")])
            )

        if rel_key not in seen_relations:
            final_relations.append(relation)
            seen_relations.add(rel_key)
```
### 1.5 检索信息长度管理
对实体信息进行长度管理，超过长度的信息进行截断，实体信息默认最大长度限制 6000 tokens。
```python
 if entities_context:
    # Remove file_path and created_at for token calculation
    entities_context_for_truncation = []
    for entity in entities_context:
        entity_copy = entity.copy()
        entity_copy.pop("file_path", None)
        entity_copy.pop("created_at", None)
        entities_context_for_truncation.append(entity_copy)

    entities_context = truncate_list_by_token_size(
        entities_context_for_truncation,
        key=lambda x: "\n".join(
            json.dumps(item, ensure_ascii=False) for item in [x]
        ),
        max_token_size=max_entity_tokens,
        tokenizer=tokenizer,
    )
```
对关系信息进行长度管理，超过长度的信息进行截断，关系信息默认最大长度限制 8000 tokens。
```python
if relations_context:
    # Remove file_path and created_at for token calculation
    relations_context_for_truncation = []
    for relation in relations_context:
        relation_copy = relation.copy()
        relation_copy.pop("file_path", None)
        relation_copy.pop("created_at", None)
        relations_context_for_truncation.append(relation_copy)

    relations_context = truncate_list_by_token_size(
        relations_context_for_truncation,
        key=lambda x: "\n".join(
            json.dumps(item, ensure_ascii=False) for item in [x]
        ),
        max_token_size=max_relation_tokens,
        tokenizer=tokenizer,
    )
```
### 1.6 提取文本块
从实体基本信息中 source_id 字段提取可能关联的 chunk id，形成 chunk 列表。
```python
for entity in node_datas:
    if entity.get("source_id"):
        chunks = split_string_by_multi_markers(
            entity["source_id"], [GRAPH_FIELD_SEP]
        )
        if chunks:
            entities_with_chunks.append(
                {
                    "entity_name": entity["entity_name"],
                    "chunks": chunks,
                    "entity_data": entity,
                }
            )
```
遍历每个实体的 chunks，统计每个 chunk 出现次数，如果某个 chunk 已经在前面的实体中出现过，则跳过。
```python
chunk_occurrence_count = {}
for entity_info in entities_with_chunks:
    deduplicated_chunks = []
    for chunk_id in entity_info["chunks"]:
        chunk_occurrence_count[chunk_id] = (
            chunk_occurrence_count.get(chunk_id, 0) + 1
        )
        if chunk_occurrence_count[chunk_id] == 1:
            deduplicated_chunks.append(chunk_id)
    entity_info["chunks"] = deduplicated_chunks
```
对每个实体的 chunk，根据出现次数进行降序排序，出现次数越多的优先级越高。
```python
total_entity_chunks = 0
for entity_info in entities_with_chunks:
    sorted_chunks = sorted(
        entity_info["chunks"],
        key=lambda chunk_id: chunk_occurrence_count.get(chunk_id, 0),
        reverse=True,
    )
    entity_info["sorted_chunks"] = sorted_chunks
    total_entity_chunks += len(sorted_chunks)
```
选择 chunk 策略，根据全局配置选择策略 kg_chunk_pick_method：
- VECTOR：使用向量相似度（cosine similarity）选出最相关的 chunk，需要提供查询文本和 embedding 函数。如果向量选择失败或没有 embedding 函数，会回退到 WEIGHT 方法。
- WEIGHT：按线性权重轮询（weighted polling），根据 chunk 出现次数和实体顺序选择 top N 个 chunk。
```python
if kg_chunk_pick_method == "VECTOR" and query and chunks_vdb:
    actual_embedding_func = text_chunks_db.embedding_func
    selected_chunk_ids = await pick_by_vector_similarity(
        query=query,
        text_chunks_storage=text_chunks_db,
        chunks_vdb=chunks_vdb,
        num_of_chunks=int(max_related_chunks * len(entities_with_chunks) / 2),
        entity_info=entities_with_chunks,
        embedding_func=actual_embedding_func,
        query_embedding=query_embedding,
    )

if kg_chunk_pick_method == "WEIGHT":
    selected_chunk_ids = pick_by_weighted_polling(
        entities_with_chunks, max_related_chunks, min_related_chunks=1
    )
```
批量获取 chunk 数据，构建结果并更新 chunk tracking，返回最终结果 entity_chunks 列表。

chunk_tracking，会记录每个 chunk 的来源、出现次数和顺序，方便后续分析或日志。

对于关系信息进行相同提取流程，提取相关 chunk 信息。与实体 chunk 列表进行合并，计数，去重，返回最终的 relation_chunks 列表。

### 1.7 合并文本块
Round-robin 依次合并三个来源的 chunk，同时进行去重。
```python
max_len = max(len(vector_chunks), len(entity_chunks), len(relation_chunks))
origin_len = len(vector_chunks) + len(entity_chunks) + len(relation_chunks)

for i in range(max_len):
    # Add from vector chunks first (Naive mode)
    if i < len(vector_chunks):
        chunk = vector_chunks[i]
        chunk_id = chunk.get("chunk_id") or chunk.get("id")
        if chunk_id and chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(chunk_id)
            merged_chunks.append(
                {
                    "content": chunk["content"],
                    "file_path": chunk.get("file_path", "unknown_source"),
                    "chunk_id": chunk_id,
                }
            )

    # Add from entity chunks (Local mode)
    if i < len(entity_chunks):
        chunk = entity_chunks[i]
        chunk_id = chunk.get("chunk_id") or chunk.get("id")
        if chunk_id and chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(chunk_id)
            merged_chunks.append(
                {
                    "content": chunk["content"],
                    "file_path": chunk.get("file_path", "unknown_source"),
                    "chunk_id": chunk_id,
                }
            )

    # Add from relation chunks (Global mode)
    if i < len(relation_chunks):
        chunk = relation_chunks[i]
        chunk_id = chunk.get("chunk_id") or chunk.get("id")
        if chunk_id and chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(chunk_id)
            merged_chunks.append(
                {
                    "content": chunk["content"],
                    "file_path": chunk.get("file_path", "unknown_source"),
                    "chunk_id": chunk_id,
                }
            )
```
### 1.8 最终上下文构建

经过上方一系列操作，已经将用户查询相关的 chunk 全部召回，检索过程已经结束。

此步骤是结合以上检索的所有信息，包括实体信息，关系信息，chunk 列表，用户原始查询，来构建回答用户查询的最终 prompt。

```python
sys_prompt_temp = system_prompt if system_prompt else PROMPTS["rag_response"]
sys_prompt = sys_prompt_temp.format(
    response_type=response_type,
    user_prompt=user_prompt,
    context_data=context_result.context,  # 包含实体、关系、chunks的完整上下文
)
```

最终的 prompt 结构为：
```
[System Prompt - 包含 response_type、user_prompt 和完整的 context_data]
---
[User Query - 原始用户问题]
```

### 1.9 LLM 调用与流式/非流式响应

确认流式模式后调用 LLM：
```python
response = await use_model_func(
    sys_prompt,
    user_query,
    stream=param.stream,  # True: 流式返回 AsyncIterator，False: 返回完整字符串
)
```

最后返回统一的 `QueryResult` 对象：
```python
QueryResult(
    content=llm_response_text,              # 非流式时是完整文本
    response_iterator=stream_iterator,     # 流式时是异步迭代器
    raw_data=context_result.raw_data,      # 原始检索数据
    is_streaming=param.stream               # 是否流式
)
```

---

## 2. query/stream（流式 RAG 问答接口）

### 2.1 流式接口的核心设计

`/query/stream` 使用与 `/query` 相同的检索流程，仅在响应方式上有所不同。`/query/stream` 是 LightRAG 提供的**最灵活的查询接口**，它支持两种响应模式：
- **流式模式**（`stream=true`）：实时返回生成的内容，每个 token 单独发送
- **非流式模式**（`stream=false`）：一次性返回完整响应

这种设计让开发者可以用同一个端点满足不同的应用需求：Web UI 使用流式获得实时体验，而批处理系统可以使用非流式模式。

### 2.2 流式响应的实现原理

#### 步骤1：参数解析与转换

```python
async def query_text_stream(request: QueryRequest):
    # 提取流式标志（默认 True）
    stream_mode = request.stream if request.stream is not None else True
    param = request.to_query_params(stream_mode)
```

#### 步骤2：调用核心查询引擎

```python
# 统一使用 aquery_llm 进行查询
result = await rag.aquery_llm(request.query, param=param)
```

`aquery_llm` 内部会根据 `stream_mode` 返回不同的响应结构：
- 如果 `is_streaming=true`：`llm_response.response_iterator` 包含异步迭代器
- 如果 `is_streaming=false`：`llm_response.content` 包含完整的字符串

#### 步骤3：构建异步生成器

```python
async def stream_generator():
    references = result.get("data", {}).get("references", [])
    llm_response = result.get("llm_response", {})
    
    if llm_response.get("is_streaming"):
        # 流式模式：先发送引用，再流式发送响应块
        if request.include_references:
            yield f"{json.dumps({'references': references})}\n"
        
        response_stream = llm_response.get("response_iterator")
        if response_stream:
            try:
                async for chunk in response_stream:
                    if chunk:
                        yield f"{json.dumps({'response': chunk})}\n"
            except Exception as e:
                yield f"{json.dumps({'error': str(e)})}\n"
    else:
        # 非流式模式：一次性发送完整响应
        response_content = llm_response.get("content", "")
        complete_response = {"response": response_content}
        if request.include_references:
            complete_response["references"] = references
        yield f"{json.dumps(complete_response)}\n"
```

#### 步骤4：返回 StreamingResponse

```python
return StreamingResponse(
    stream_generator(),
    media_type="application/x-ndjson",  # NDJSON 格式
    headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
    },
)
```

### 2.3 NDJSON 响应格式详解

LightRAG 使用 **NDJSON**（Newline Delimited JSON）格式，每行是一个独立的 JSON 对象：

**流式模式响应**：
```
{"references": [{"reference_id": "1", "file_path": "/docs/ai.pdf"}]}
{"response": "Artificial Intelligence"}
{"response": " is a branch of"}
{"response": " computer science"}
```

**非流式模式响应**：
```
{"references": [{"reference_id": "1", "file_path": "/docs/ai.pdf"}], "response": "Artificial Intelligence is a branch of computer science"}
```

## 3. query/data（结构化数据检索接口）

### 3.1 数据检索接口的设计目标

`/query/data` 是一个**纯检索接口**，它跳过 LLM 生成阶段，直接返回检索到的原始数据结构。这个接口的价值在于：

- **可视化知识图谱**：理解文档库中的实体和关系
- **评估检索质量**：不被 LLM 生成掩盖，看到真实的检索效果
- **调试和优化**：通过分析检索数据来优化模型参数
- **数据分析**：用于研究知识图谱的覆盖率和质量
- **二次处理**：作为其他系统的数据源

### 3.2 返回数据结构详解

#### 基础响应结构

```json
{
    "status": "success|failure",
    "message": "状态描述信息",
    "data": {
        "entities": [...],
        "relationships": [...],
        "chunks": [...],
        "references": [...]
    },
    "metadata": {
        "query_mode": "local|global|hybrid|naive|mix|bypass",
        "keywords": {
            "high_level": [...],
            "low_level": [...]
        },
        "processing_info": {...}
    }
}
```

#### 实体数据结构

```python
{
    "entity_name": "Neural Networks",           # 实体名称
    "entity_type": "CONCEPT",                   # 实体类型
    "description": "Computational models...",   # 实体描述
    "source_id": "chunk-123|chunk-124",        # 来源 chunk ID（管道分隔）
    "file_path": "/documents/ai_basics.pdf",   # 源文件路径
    "created_at": "2025-01-15T10:30:00",       # 创建时间
    "reference_id": "1"                         # 引用 ID
}
```

#### 关系数据结构

```python
{
    "src_id": "Neural Networks",                # 源实体 ID
    "tgt_id": "Machine Learning",              # 目标实体 ID
    "description": "Neural networks are...",   # 关系描述
    "keywords": "subset,algorithm,learning",   # 关键词
    "weight": 0.85,                            # 关系权重（0-1）
    "source_id": "chunk-123",                  # 来源 chunk
    "file_path": "/documents/ai_basics.pdf",   # 源文件
    "created_at": "2025-01-15T10:30:00",       # 创建时间
    "reference_id": "1"                         # 引用 ID
}
```

#### 文本块数据结构

```python
{
    "content": "Neural networks are computational models...",
    "file_path": "/documents/ai_basics.pdf",
    "chunk_id": "chunk-123",
    "reference_id": "1"
}
```

#### 元数据结构

```python
{
    "query_mode": "hybrid",                    # 使用的查询模式
    "keywords": {
        "high_level": ["AI", "learning"],      # 高层关键词
        "low_level": ["neural", "network"]     # 低层关键词
    },
    "processing_info": {
        "total_entities_found": 15,            # 截断前实体数
        "total_relations_found": 8,            # 截断前关系数
        "entities_after_truncation": 5,        # 截断后实体数
        "relations_after_truncation": 3,       # 截断后关系数
        "merged_chunks_count": 20,             # 合并前 chunk 数
        "final_chunks_count": 10                # 最终 chunk 数
    }
}
```

### 3.3 不同模式下的数据特点

#### Local 模式
```python
# local 模式的特点：
# - 实体数量多（基于低层关键词的直接匹配）
# - 关系数量相对较少（通过实体获取）
# - chunks 数量中等（通过实体的 source_id 关联）

{
    "entities": [  # 丰富
        {"entity_name": "Neural Networks", ...},
        {"entity_name": "Deep Learning", ...},
        ...
    ],
    "relationships": [  # 相对较少
        {"src_id": "Neural Networks", "tgt_id": "Machine Learning", ...},
        ...
    ],
    "chunks": [  # 中等
        {"content": "...", "chunk_id": "chunk-123"},
        ...
    ]
}
```

#### Global 模式
```python
# global 模式的特点：
# - 关系数量多（基于高层关键词的直接匹配）
# - 实体数量相对较少（通过关系获取）
# - chunks 数量中等（通过关系的 source_id 关联）

{
    "entities": [  # 相对较少
        ...
    ],
    "relationships": [  # 丰富
        {"src_id": "AI", "tgt_id": "Machine Learning", "weight": 0.92, ...},
        {"src_id": "ML", "tgt_id": "Deep Learning", "weight": 0.88, ...},
        ...
    ],
    "chunks": [  # 中等
        ...
    ]
}
```

#### Hybrid 模式
```python
# hybrid 模式的特点：
# - 实体和关系都相对丰富（local + global 的并集）
# - chunks 数量最多（来自多个来源）
# - 数据最完整

{
    "entities": [...],      # 多（local + global）
    "relationships": [...], # 多（local + global）
    "chunks": [...]        # 最多（来自多个来源）
}
```

#### Naive 模式
```python
# naive 模式的特点：
# - 没有实体和关系（纯向量检索）
# - 只有 chunks
# - 最轻量

{
    "entities": [],         # 空
    "relationships": [],    # 空
    "chunks": [...]        # 向量搜索结果
}
```

### 3.4 使用 query/data 进行分析

#### 分析1：评估检索覆盖率

```python
def analyze_retrieval_coverage(query: str, mode: str = "mix"):
    response = requests.post(
        "http://localhost:8000/query/data",
        json={"query": query, "mode": mode, "top_k": 20}
    ).json()
    
    if response["status"] != "success":
        return None
    
    data = response["data"]
    metadata = response["metadata"]
    
    # 分析覆盖率
    coverage = {
        "entities_found": len(data["entities"]),
        "entities_total": metadata["processing_info"]["total_entities_found"],
        "relations_found": len(data["relationships"]),
        "relations_total": metadata["processing_info"]["total_relations_found"],
        "chunks_found": len(data["chunks"]),
        "unique_files": len(set(c["file_path"] for c in data["chunks"])),
        "coverage_rate": (
            len(data["entities"]) / max(1, metadata["processing_info"]["total_entities_found"]) * 100
        )
    }
    
    print(f"实体覆盖率: {coverage['coverage_rate']:.1f}%")
    print(f"关系截断: {coverage['relations_total']} → {coverage['relations_found']}")
    print(f"涉及文件数: {coverage['unique_files']}")
    
    return coverage
```

#### 分析2：关键词质量评估

```python
def analyze_keywords_quality(query: str):
    response = requests.post(
        "http://localhost:8000/query/data",
        json={"query": query, "mode": "hybrid"}
    ).json()
    
    metadata = response["metadata"]
    hl_keywords = metadata["keywords"]["high_level"]
    ll_keywords = metadata["keywords"]["low_level"]
    
    print(f"查询: {query}")
    print(f"高层关键词 ({len(hl_keywords)}): {hl_keywords}")
    print(f"低层关键词 ({len(ll_keywords)}): {ll_keywords}")
    
    # 评估
    if not hl_keywords and not ll_keywords:
        print("⚠️ 警告：未提取到任何关键词，可能导致检索失败")
    
    if len(hl_keywords) > 5:
        print("⚠️ 警告：高层关键词过多，可能导致结果发散")
    
    if len(ll_keywords) > 10:
        print("⚠️ 警告：低层关键词过多，可能导致 token 超出")
```

#### 分析3：模式对比

```python
def compare_modes(query: str):
    modes = ["local", "global", "hybrid", "naive", "mix"]
    results = {}
    
    for mode in modes:
        response = requests.post(
            "http://localhost:8000/query/data",
            json={"query": query, "mode": mode}
        ).json()
        
        data = response["data"]
        results[mode] = {
            "entities": len(data["entities"]),
            "relations": len(data["relationships"]),
            "chunks": len(data["chunks"]),
            "files": len(set(c["file_path"] for c in data["chunks"]))
        }
    
    # 对比表
    import pandas as pd
    df = pd.DataFrame(results).T
    print(df)
    print("\n模式建议:")
    print(f"- 最多实体: {df['entities'].idxmax()}")
    print(f"- 最多关系: {df['relations'].idxmax()}")
    print(f"- 最多 chunks: {df['chunks'].idxmax()}")
```

#### 分析4：Token 影响分析

```python
def analyze_token_impact(query: str):
    response = requests.post(
        "http://localhost:8000/query/data",
        json={
            "query": query, 
            "mode": "hybrid",
            "max_total_tokens": 4000
        }
    ).json()
    
    metadata = response["metadata"]
    info = metadata["processing_info"]
    
    truncation_rate = {
        "entities": (1 - info["entities_after_truncation"] / max(1, info["total_entities_found"])) * 100,
        "relations": (1 - info["relations_after_truncation"] / max(1, info["total_relations_found"])) * 100,
        "chunks": (1 - info["final_chunks_count"] / max(1, info["merged_chunks_count"])) * 100,
    }
    
    print("截断率（token 限制导致的数据丢失）:")
    for key, rate in truncation_rate.items():
        print(f"  {key}: {rate:.1f}%")
    
    if any(rate > 30 for rate in truncation_rate.values()):
        print("\n⚠️ 建议: 增加 max_total_tokens 以保留更多检索结果")
```

### 3.5 query/data 的实际应用

#### 应用1：知识图谱可视化

```python
def visualize_knowledge_graph(query: str):
    """使用 pyvis 可视化知识图谱"""
    response = requests.post(
        "http://localhost:8000/query/data",
        json={"query": query, "mode": "hybrid", "top_k": 30}
    ).json()
    
    from pyvis.network import Network
    net = Network(directed=True, physics=True)
    
    # 添加节点
    for entity in response["data"]["entities"]:
        net.add_node(
            entity["entity_name"],
            label=entity["entity_name"],
            title=entity.get("description", ""),
        )
    
    # 添加边
    for rel in response["data"]["relationships"]:
        net.add_edge(
            rel["src_id"],
            rel["tgt_id"],
            label=rel.get("description", ""),
            weight=rel.get("weight", 0.5),
        )
    
    net.show("knowledge_graph.html")
    print("知识图谱已保存到 knowledge_graph.html")
```

#### 应用2：RAG 效果基准测试

```python
def benchmark_rag(test_queries: List[str], modes: List[str]):
    """对比不同模式的检索效果"""
    import pandas as pd
    
    results = []
    for query in test_queries:
        for mode in modes:
            response = requests.post(
                "http://localhost:8000/query/data",
                json={"query": query, "mode": mode}
            ).json()
            
            data = response["data"]
            results.append({
                "query": query[:30],
                "mode": mode,
                "entities": len(data["entities"]),
                "relations": len(data["relationships"]),
                "chunks": len(data["chunks"]),
                "files": len(set(c["file_path"] for c in data["chunks"]))
            })
    
    df = pd.DataFrame(results)
    pivot = df.pivot_table(
        index="query",
        columns="mode",
        values="chunks",
        aggfunc="first"
    )
    print(pivot)
```

#### 应用3：自动化测试

```python
def test_retrieval_quality(query: str, expected_entities: List[str]):
    """检查检索是否包含期望的实体"""
    response = requests.post(
        "http://localhost:8000/query/data",
        json={"query": query, "mode": "mix"}
    ).json()
    
    entities = {e["entity_name"] for e in response["data"]["entities"]}
    expected = set(expected_entities)
    
    found = entities & expected
    missed = expected - found
    
    print(f"查询: {query}")
    print(f"找到的期望实体: {found}")
    print(f"缺失的实体: {missed}")
    
    recall = len(found) / len(expected) if expected else 1.0
    print(f"召回率: {recall * 100:.1f}%")
    
    assert recall >= 0.8, f"召回率过低: {recall}"
```

## 尾言

LightRAG 的检索架构体现了"混合优于单一"的设计哲学。传统 RAG 依赖向量相似度检索，容易受语义漂移影响；纯知识图谱检索则受限于图谱构建质量。LightRAG 通过双层关键词策略和 Round-robin 合并机制，让两种检索方式相互补充，既保证了主题相关性（global 模式），又兼顾了细粒度匹配（local 模式）。

在实际应用中，建议根据场景选择合适的查询模式：对于需要精确匹配的问答场景，优先使用 `mix` 模式；对于需要探索性检索的知识发现场景，`hybrid` 模式能提供更全面的结果；而 `naive` 模式则适合轻量级场景或图谱尚未完善的初期阶段。

值得注意的是，LightRAG 的 Token 管理机制虽然精细，但在实际使用中需要根据文档规模和 LLM 上下文窗口进行调优。过小的 token 限制会导致重要信息被截断，过大的限制则可能引入噪声。通过 `/query/data` 接口定期分析检索质量，结合 `processing_info` 中的截断统计，可以找到最佳的参数平衡点。

最后，LightRAG 的完整可追溯性设计（引用信息、chunk tracking）不仅方便调试，也为 RAG 系统的持续优化提供了数据基础。在构建生产级 RAG 应用时，建议充分利用这些元数据，建立检索质量监控体系，持续迭代优化检索效果。