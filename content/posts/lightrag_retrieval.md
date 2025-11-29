---
title: "【解密源码】 轻量 GrapghRAG - LightRAG 检索工程实践"
date: 2025-11-22T11:33:10+08:00
draft: false
tags: ["源码","技术","RAG"]
categories: ["LightRAG"]
---

# 引言

# 省流版

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

此步骤是结合以上检索的所有信息，包括实体信息，关系信息，chunk 列表，用户原始查询，来构建回答用户查询的最终 prompt。最终调用



## 向量检索

## 只通过 LLM 生成回答
