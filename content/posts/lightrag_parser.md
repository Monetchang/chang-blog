---
title: "【解密源码】 轻量 GrapghRAG - LightRAG 文档解析工程实践"
date: 2025-11-22T11:33:10+08:00
draft: false
tags: ["源码","技术","RAG"]
categories: ["LightRAG"]
---

# 引言
传统 RAG 做法依赖文本相似度，难以处理多跳推理与跨段整合。为突破这一瓶颈，行业开始探索基于知识图谱的 GraphRAG，通过实体、关系与图谱结构让模型“理解”文档。不过传统 GraphRAG 往往过于“重”，需要外部图数据库、复杂本体以及多阶段构建流程，实际落地成本高、门槛也不低。

LightRAG 正是在这种背景下出现的“轻量级 GraphRAG 实现”。

它保留了 GraphRAG 的核心思想（实体 → 关系 → 图谱 → 推理），但通过极简架构把复杂度大幅压缩，使其能够在单机、本地、无图数据库依赖的情况下流畅运行。与普通 RAG 相比，它拥有更好的信息组织能力；与传统 GraphRAG 相比，它几乎没有额外的部署门槛。

总结来看，LightRAG 具备三大关键优势：

- 更智能的检索：不仅按内容查找，还能依据实体关系和全局语义图谱定位答案。

- 超轻量的工程实现：只依赖本地向量库 + KV 存储，不需要 Neo4j、RedisGraph 等复杂系统。

- 支持增量更新与实时构建：文件级异步流水线架构可持续处理新文档，适合动态知识库。

凭借以上特性，LightRAG 成为当前最具“工程可落地性”的 GraphRAG 方案之一，也让“图谱增强检索”首次能够以真正低门槛的方式应用在实际产品场景中。
# 省流版
## 基础流程
下面这张流程图总结了 LightRAG 从 文档入库 → 文档切分 → 实体关系抽取 → 知识图谱合并 的完整处理链路：
```
┌──────────────────────────────────┐
  文档入库任务处理（入口）             
└──────────────────────────────────┘
                │
                ▼
    ┌──────────────────────────┐
      获取待处理/失败的文档       
    └──────────────────────────┘
                │
                ▼
    ┌──────────────────────────┐
      校验文档状态与一致性        
    └──────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────┐
  并发调度每个文档（使用并发与信号量限制）        
└───────────────────────────────────────────┘
                │
                ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【单文档处理流程】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                │
                ▼
    ┌──────────────────────────┐
      标记文档为“处理中”          
    └──────────────────────────┘
                │
                ▼
    ┌──────────────────────────┐
      阶段 1：文档切分为 chunks  
       - 解析文本               
       - 存储分块内容            
       - 建立文档与分块映射     
    └──────────────────────────┘
                │
                ▼
    ┌──────────────────────────┐
      阶段 2：抽取实体与关系    
       - 识别重要实体           
       - 识别实体之间的关系     
       - 生成嵌入向量           
       - 存入向量库             
    └──────────────────────────┘
                │
                ▼
    ┌──────────────────────────┐
      阶段 3：合并知识图谱      
       - 合并实体节点           
       - 合并关系边             
       - 更新节点权重           
    └──────────────────────────┘
                │
                ▼
    ┌──────────────────────────┐
      更新文档状态（成功/失败） 
      记录错误信息（如有）      
    └──────────────────────────┘
                │
                ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【入库任务结束逻辑】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                │
                ▼
    ┌──────────────────────────┐
      等待所有文档处理完成     
    └──────────────────────────┘
                │
                ▼
    ┌──────────────────────────┐
      若仍有待处理任务则继续   
      （例如：新增或重试任务） 
    └──────────────────────────┘

```
## 设计亮点
**1. Gleaning Prompt：在最小负担下最大化模型抽取能力**
LightRAG 引入轻量但高效的“信息提炼” Prompt，既减少模型 Hallucination，又提升结构化抽取质量。

**2. 自建缓存机制减少重复 LLM 调用**
对每个 chunk 抽取做 hash 缓存，让重复任务几乎零成本，特别适合增量更新和批量文档处理。

**3. 图谱两阶段合并策略（节点 → 关系）**
先合并节点，再合并边，避免图谱被错误关系污染，也更便于后续扩展和版本化管理。

**4. 文档级与 chunk 级并发双层优化**
文档之间有 semaphore，chunk 内部也支持快速处理，使整个系统更轻量、更快、更稳定。

# 手撕版
## 1. 上传文件
### 1.1 上传文件前置检查
对上传的文件名以及上传文件夹路径进行校验，防止路径穿越攻击、文件名类型检查、重复上传检查（上传中，已上传）。
#### 1.1.1 防止路径穿越攻击
对文件进行基础校验，文件名，文件夹不能为空，通过一系列策略防止路径穿越攻击，返回安全的文件路径。
```python
def sanitize_filename(filename: str, input_dir: Path) -> str:
    if not filename or not filename.strip():
        aise HTTPException(status_code=400, detail="Filename cannot be empty")
    clean_name = filename.replace("/", "").replace("\\", "")
    clean_name = clean_name.replace("..", "")
    clean_name = "".join(c for c in clean_name if ord(c) >= 32 and c != "\x7f")
    clean_name = clean_name.strip().strip(".")
    ...
```
#### 1.1.2 文件名类型检查
对上传的文件名进行类型检查，判断是否为支持的文件类型。
```python
 supported_extensions: tuple = (
    ".txt",
    ".md",
    ".pdf",
    ".docx",
    ...
)
def is_supported_file(self, filename: str) -> bool:
    return any(filename.lower().endswith(ext) for ext in self.supported_extensions)
```
#### 1.1.3 重复上传检查
检查文件是否已存在，状态为上传中，已上传的文件，防止重复上传。
```python
existing_doc_data = await rag.doc_status.get_doc_by_file_path(safe_filename)
if existing_doc_data:
    # Get document status information for error message
    status = existing_doc_data.get("status", "unknown")
```

## 2. 解析文件
将一个上传后的文件解析成纯文本内容，然后把解析后的文本投递（enqueue）到 LightRAG 的异步处理队列里。
### 2.1 提取文件内容
读取文件内容，并及其详细的记录读取文件错误场景的异常信息，包括读取各类型文件特有错误场景的异常信息。
```python
async with aiofiles.open(file_path, "rb") as f:
    file = await f.read()
```
根据不同文件类型，采用不同的文本提取方案。
```
match ext:
    case .txt / .md / .html / 等文本文件 → UTF-8 解码
    case .pdf → PDF 提取文本
    case .docx → Word 抽取文本
    case .pptx → PPT 抽取文本
    case .xlsx → Excel 抽取文本
    case _ → 不支持的格式错误
```
### 2.2 解析文件内容
对于所有文件优先使用 Docling 框架进行文本解析，若用户未配置 DOCLING 解析器，则根据文件类型，采用不同的文本解析方案。
| 文件类型                                                                                                                                                                                                                                           | 使用引擎 / 库              | 解析方式                                                        |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- | ----------------------------------------------------------- |
| `.txt` `.md` `.html` `.htm` `.tex` `.json` `.xml` `.yaml` `.yml` `.rtf` `.odt` `.epub` `.csv` `.log` `.conf` `.ini` `.properties` `.sql` `.bat` `.sh` `.c` `.cpp` `.py` `.java` `.js` `.ts` `.swift` `.go` `.rb` `.php` `.css` `.scss` `.less` | 无（UTF-8 decode）       | 使用 `file.decode("utf-8")` 获取纯文本内容                           |
| `.pdf`                                                                                                                                                                                                                                         | Docling 或 PyPDF       | Docling：转换为 Markdown；PyPDF：逐页 `extract_text()`              |
| `.docx`                                                                                                                                                                                                                                        | Docling 或 python-docx | Docling：转换为 Markdown；python-docx：遍历 `paragraphs` 拼接文本       |
| `.pptx`                                                                                                                                                                                                                                        | Docling 或 python-pptx | Docling：转换为 Markdown；python-pptx：遍历 slides/shapes 提取 `text` |
| `.xlsx`                                                                                                                                                                                                                                        | Docling 或 openpyxl    | Docling：转换为 Markdown；openpyxl：遍历 sheet/rows 拼接文本            |
| **其他未知类型**                                                                                                                                                                                                                                     | 不支持                   | —                                                           |
### 2.3 存储文件数据
存储解析后的文件内容到 KV Store 中，默认使用的 KV Store 是本地的 JSON 文件形式。
```python
full_docs_data = {
    doc_id: {
        "content": contents[doc_id]["content"],
        "file_path": contents[doc_id]["file_path"],
    }
    for doc_id in new_docs.keys()
}
await self.full_docs.upsert(full_docs_data)
```
### 2.4 分词
默认使用 Tiktoken tokenizer 进行分词，模型默认使用 gpt-4o-mini。按照固定长度 + 语义符号分词，并设置重叠标记数量。
```python
def chunking_by_token_size(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
) 

# - `tokenizer`: A Tokenizer instance to use for tokenization.
# - `content`: The text to be split into chunks.
# - `split_by_character`: The character to split the text on. If None, the text is split into chunks of `chunk_token_size` tokens.
# - `split_by_character_only`: If True, the text is split only on the specified character.
# - `chunk_token_size`: The maximum number of tokens per chunk.
# - `chunk_overlap_token_size`: The number of overlapping tokens between consecutive chunks.

```
### 2.5 存储数据
将 chunks 数据存储到向量数据库中。向量数据库默认使用 NanoVectorDBStorage。
```python
chunks_vdb_task = asyncio.create_task(
    self.chunks_vdb.upsert(chunks)
)
```
## 3. 提取文件实体
### 3.1 单个 chunk 实体提取
对每个 chunk 通过调用 LLM（带缓存）抽取实体和实体关系，其中有几个重要的提取参数。
```python
# 使用全局配置中的 LLM 模型函数
use_llm_func = global_config["llm_model_func"]
# 最大抽取次数，默认 1 次
entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
# 语言
language = global_config["addon_params"].get("language")
# 实体类型，默认所有类型
entity_types = global_config["addon_params"].get("entity_types")
```
其中 LightRAG 支持的实体类型有：
```python
DEFAULT_ENTITY_TYPES = [
    "Person",
    "Creature",
    "Organization",
    "Location",
    "Event",
    "Concept",
    "Method",
    "Content",
    "Data",
    "Artifact",
    "NaturalObject",
]
```
### 3.1.1 构建 Prompt
一共会构建三个 prompt，分别是：公用 system prompt 初次抽取 prompt和重复抽取（gleaning）prompt。

**1）system prompt**

| 模块名称                            | 中文说明（作用）                                                                       | Prompt 原文示例片段（对应模块）                                                                                                                                           |          |                                                                                            |
| ------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------ |
| **角色设定（Role）**                  | 规定模型扮演“知识图谱专家”，确保输出专业性并避免闲聊或发挥。                                                | `You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the input text.`                                             |          |                                                                                            |
| **实体抽取规则：识别与结构要求**              | 定义如何识别实体、实体的命名规范、字段内容要求（名称、类型、描述）。                                             | `Identify clearly defined and meaningful entities… extract: entity_name, entity_type, entity_description`                                                     |          |                                                                                            |
| **实体输出格式规范**                    | 明确实体输出必须包含 4 个字段，并用 `{tuple_delimiter}` 分隔；指定第一字段必须是 `entity`。                 | `Format: entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`                                                   |          |                                                                                            |
| **关系抽取规则：识别与拆分**                | 定义如何识别关系、如何将多元关系拆分为二元关系，避免遗漏复杂关系。                                              | `If a single statement describes a relationship involving more than two entities… decompose into multiple binary relationship pairs`                          |          |                                                                                            |
| **关系字段要求**                      | 要求 source、target、relationship_keywords、relationship_description 四大字段；关键词用逗号分隔。 | `relationship_keywords: One or more high-level keywords… separated by a comma ','`                                                                            |          |                                                                                            |
| **关系输出格式规范**                    | 明确关系输出必须包含 5 个字段，第一字段必须是 `relation`。                                           | `Format: relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description` |          |                                                                                            |
| **分隔符使用规范（Delimiter Protocol）** | 强制禁止在字段内容中使用 `{tuple_delimiter}`；提供正确/错误示例说明。                                  | `Incorrect Example: ... Tokyo<                                                                                                                                | location | >Tokyo ...`<br>`Correct Example: entity{tuple_delimiter}Tokyo{tuple_delimiter}location...` |
| **关系方向性与去重规则**                  | 指定关系视为无向，防止重复输出 A→B 和 B→A。                                                     | `Treat all relationships as undirected… Avoid outputting duplicate relationships.`                                                                            |          |                                                                                            |
| **输出顺序要求**                      | 必须先实体再关系；关系内部按重要性排序。                                                           | `Output all extracted entities first, followed by all extracted relationships.`                                                                               |          |                                                                                            |
| **客观性与第三人称要求**                  | 禁止使用“this article”“I”等代词，避免含糊实体产生。                                             | `avoid using pronouns such as 'this article', 'I', 'he/she'`                                                                                                  |          |                                                                                            |
| **语言与专有名词规则**                   | 要求输出语言为 `{language}`；专有名词必须保留原文。                                               | `Proper nouns … should be retained in their original language`                                                                                                |          |                                                                                            |
| **结束信号**                        | 最后一行必须输出 `{completion_delimiter}`，方便程序检测输出结束。                                  | `Output the literal string {completion_delimiter} only after…`                                                                                                |          |                                                                                            |
| **示例（Examples）**                | few-shot 示例帮助模型学习正确格式和所需输出风格。                                                  | `{examples}`                                                                                                                                                  |          |                                                                                            |
| **输入数据声明**                      | 显示实际输入（entity_types + text），引导模型处理正确文本。                                        | `Text: `{input_text}`                                                                                                                                         |          |                                                                                            |

**2）初次抽取 prompt（user prompt）**
| 模块名称            | 中文说明（作用）                            | Prompt 原文示例片段（对应模块）                                                                               |
| --------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------- |
| **任务声明（Task）**  | 表明用户希望模型执行实体和关系抽取任务。                | `Extract entities and relationships from the input text…`                                         |
| **必须严格遵循格式要求**  | 强调输出必须符合 system prompt 规定的所有格式。     | `Strict Adherence to Format… including output order, field delimiters…`                           |
| **仅输出内容（禁止解释）** | 禁止输出任何介绍性语言、评注或解释，确保便于程序解析。         | `Output only the extracted list of entities and relationships. Do not include any…`               |
| **完成信号（结束标记）**  | 输出完成时必须加上 `{completion_delimiter}`。 | `Output {completion_delimiter} as the final line…`                                                |
| **语言要求**        | 输出语言必须为 `{language}`，专有名词不得翻译。      | `Ensure the output language is {language}. Proper nouns must be kept in their original language.` |
| **输出占位符**       | 用 `<Output>` 作为模型真正输出内容的位置标识。       | `<Output>`                                                                                        |

**3）重复抽取（gleaning）prompt（user prompt）**
| 模块         | 中文作用说明                            | 原文示例                                                                                  |
| ---------- | --------------------------------- | ------------------------------------------------------------------------------------- |
| **任务声明**   | 表示本次任务是“补漏/修正”而不是重新抽取。            | `identify and extract any missed or incorrectly formatted entities and relationships` |
| **仅补充缺失项** | 避免重复输出已正确生成的实体或关系。                | `Do NOT re-output entities and relationships that were correctly…`                    |
| **格式要求重申** | 强制保持 entity / relation 字段数量、顺序一致。 | `Output a total of 4 fields… Output a total of 5 fields…`                             |
| **仅输出内容**  | 同样禁止解释性文字。                        | `Output only the extracted list…`                                                     |
| **完成信号**   | 必须加入 `{completion_delimiter}`。    | `Output {completion_delimiter} as the final line…`                                    |
| **语言要求**   | 强制输出 `{language}`，专有名词保持原文。       | `Ensure the output language is {language}.… kept in their original language.`         |

#### 3.1.2 使用 LLM 提取（自建缓存机制）
使用 LLM 进行实体抽取和关系抽取，为了避免重复抽取浪费 token，LightRAG 在此设计了缓存机制，默认使用本地 json 文件进行缓存管理。
```python
aasync def use_llm_func_with_cache(
    user_prompt: str,
    use_llm_func: callable,
    llm_response_cache: "BaseKVStorage | None" = None,
    system_prompt: str | None = None,
    max_tokens: int = None,
    history_messages: list[dict[str, str]] = None,
    cache_type: str = "extract",
    chunk_id: str | None = None,
    cache_keys_collector: list = None,
) -> tuple[str, int]:
```
先清洗 prompt 和 history messages（如果该参数不为 None），然后构造完整的 prompt，system prompt + user prompt + history messages
```python
prompt_parts = []
if safe_user_prompt:
    prompt_parts.append(safe_user_prompt)
if safe_system_prompt:
    prompt_parts.append(safe_system_prompt)
if history:
    prompt_parts.append(history)
_prompt = "\n".join(prompt_parts)
```
对最终的 prompt 进行 hash，从缓存中获取缓存结果。如果命中直接返回缓存结构
```python
arg_hash = compute_args_hash(_prompt)
cache_key = generate_cache_key("default", cache_type, arg_hash)
cached_result = await handle_cache(
    llm_response_cache,
    arg_hash,
    _prompt,
    "default",
    cache_type=cache_type,
)
if cached_result:
    content, timestamp = cached_result
    logger.debug(f"Found cache for {arg_hash}")
    statistic_data["llm_cache"] += 1

    # Add cache key to collector if provided
    if cache_keys_collector is not None:
        cache_keys_collector.append(cache_key)

    return content, timestamp
```
无命中任何缓存，调用 LLM 进行提取后存入缓存中。
```python
res = await use_llm_func(
    safe_user_prompt, system_prompt=safe_system_prompt, **kwargs
)
...
await save_to_cache(
    llm_response_cache,
    CacheData(
        args_hash=arg_hash,
        content=res,
        prompt=_prompt,
        cache_type=cache_type,
        chunk_id=chunk_id,
    ),
)
```
#### 3.1.3 二次抽取补充
如果 entity_extract_max_gleaning 大于 0，说明需要补充抽取。
```python
if entity_extract_max_gleaning > 0:
    glean_result, timestamp = await use_llm_func_with_cache(
        entity_continue_extraction_user_prompt,
        use_llm_func,
        system_prompt=entity_extraction_system_prompt,
        llm_response_cache=llm_response_cache,
        history_messages=history,
        cache_type="extract",
        chunk_id=chunk_key,
        cache_keys_collector=cache_keys_collector,
    )
```
#### 3.1.4 实体结构转换
把 LLM 输出的文本格式结果（实体 + 关系）解析成结构化的数据：nodes 和 edges
```python
async def _process_extraction_result(
    result: str,
    chunk_key: str,
    timestamp: int,
    file_path: str = "unknown_source",
    tuple_delimiter: str = "<|#|>",
    completion_delimiter: str = "<|COMPLETE|>",
) -> tuple[dict, dict]:
```
以下根据代码推测的简单场景示例：
```js
// LLM 输出
entity<|#|> 乔布斯 <|#|> 人物
entity<|#|> 苹果公司 <|#|> 企业
relation<|#|> 苹果公司 <|#|> 创立 <|#|> 乔布斯
<|COMPLETE|>

// nodes
{
  "乔布斯": [
    {
      "entity_name": "乔布斯",
      "entity_type": "人物",
      "file_path": "doc1.pdf"
    }
  ],
  "苹果公司": [
    {
      "entity_name": "苹果公司",
      "entity_type": "企业"
    }
  ]
}

// edges
{
  ["苹果公司", "乔布斯"]: [
    {
      "src_id": "苹果公司",
      "relation": "创立",
      "tgt_id": "乔布斯"
    }
  ]
}
```

### 3.2 并发处理所有 chunk 实体提取
3.1 中介绍了单个 chunk 的实体抽取流程，在 LightRAG 中，会并发处理所有 chunk 的实体抽取，默认是 4 个 chunk 并发处理。
```python
chunk_max_async = global_config.get("llm_model_max_async", 4)
semaphore = asyncio.Semaphore(chunk_max_async)

async def _process_with_semaphore(chunk):
    async with semaphore:
        # Check for cancellation before processing chunk
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                if pipeline_status.get("cancellation_requested", False):
                    raise PipelineCancelledException(
                        "User cancelled during chunk processing"
                    )

        try:
            return await _process_single_content(chunk)
        except Exception as e:
            chunk_id = chunk[0]  # Extract chunk_id from chunk[0]
            prefixed_exception = create_prefixed_exception(e, chunk_id)
            raise prefixed_exception from e

tasks = []
for c in ordered_chunks:
    task = asyncio.create_task(_process_with_semaphore(c))
    tasks.append(task)
```


## 4. 合并知识图谱（关键）
采用经典图数据库两阶段合并法，先合并所有实体（节点），再合并所有关系（边）。因为关系（边）引用了实体（节点），所以必须先合并实体（节点）。

两阶段合并法能够避免以下场景：
```
A → B，关系中出现 B

但 B 的实体还没有被写入

关系 upsert 会失败
```
### 4.1 聚合所有节点 / 边
把多个 chunk 中同名实体放一起。按照边 key 排序，实现无向图的合并。例如 ("A","B") 和 ("B","A") 会归为同一条边。
```python
all_nodes = defaultdict(list)
all_edges = defaultdict(list)

for maybe_nodes, maybe_edges in chunk_results:
    for entity_name, entities in maybe_nodes.items():
        all_nodes[entity_name].extend(entities)

    for edge_key, edges in maybe_edges.items():
        sorted_edge_key = tuple(sorted(edge_key))
        all_edges[sorted_edge_key].extend(edges)

```
### 4.2 合并全部实体（节点）
```python
async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage | None,
    global_config: dict,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    entity_chunks_storage: BaseKVStorage | None = None,
):
```
#### 4.2.1 从当前知识图谱中获取实体
通过实体名称从当前知识图谱中获取实体信息。
```python
already_node = await knowledge_graph_inst.get_node(entity_name)
if already_node:
    already_entity_types.append(already_node["entity_type"])
    already_source_ids.extend(already_node["source_id"].split(GRAPH_FIELD_SEP))
    already_file_paths.extend(already_node["file_path"].split(GRAPH_FIELD_SEP))
    already_description.extend(already_node["description"].split(GRAPH_FIELD_SEP))
```
#### 4.2.2 合并实体所关联的 chunk 列表
合并新的 chunk id 列表到已存在的列表中。
```python
full_source_ids = merge_source_ids(existing_full_source_ids, new_source_ids)
```
#### 4.2.3 裁剪 chunk 列表长度
支持 KEEP 和 FIFO 策略，默认采用 KEEP 策略。为了保证后续摘要环节上下文溢出问题，默认限制每个实体最多关联 300 个 chunk。
- KEEP 策略：保旧去新，保留所有关联的 chunk id，但对新加入但超限的 chunk 后续不会进行摘要。
- FIFO 策略：保新去旧，按顺序保留关联的 chunk id，超限的 chunk 会被先进先出的策略丢弃。
```python
if limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP:
    allowed_source_ids = set(source_ids)
    filtered_nodes = []
    for dp in nodes_data:
        source_id = dp.get("source_id")
        # Skip descriptions sourced from chunks dropped by the limitation cap
        if (
            source_id
            and source_id not in allowed_source_ids
            and source_id not in existing_full_source_ids
        ):
            continue
        filtered_nodes.append(dp)
    nodes_data = filtered_nodes
else:  # In FIFO mode, keep all nodes - truncation happens at source_ids level only
    nodes_data = list(nodes_data)

if (
    limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP
    and len(existing_full_source_ids) >= max_source_limit
    and not nodes_data
):
    if already_node:
        logger.info(
            f"Skipped `{entity_name}`: KEEP old chunks {already_source_ids}/{len(full_source_ids)}"
        )
        existing_node_data = dict(already_node)
        return existing_node_data
```
#### 4.2.4 合并实体 description
传入的单个实体可能关联多个 chunk，同时对应多个 description。

需要对多个 description 去重后按时间排序，后将排序的 description 列表与知识图谱中已存在的相同实体 description 合并。
```python
description_list = already_description + sorted_descriptions
```

#### 4.2.5 生成最终摘要 description
将合并后的多个实体 description 传入 LLM 进行摘要，生成最终的 description。
```python
description, llm_was_used = await _handle_entity_relation_summary(
    "Entity",
    entity_name,
    description_list,
    GRAPH_FIELD_SEP,
    global_config,
    llm_response_cache,
)
```
#### 4.2.6 合并实体源文件路径
传入的单个实体可能关联多个 chunk，同时对应多个源文件路径。需要对多个源文件路径去重后合并，生成最终的源文件路径列表。与合并 chunk 列表类似，也需要限制长度，默认采用 KEEP 策略。
#### 4.2.7 更新图数据库实体信息
其中除了实体名称，还包含实体 id、类型、描述、源文件路径、创建时间、截断信息等。
```python
node_data = dict(
    entity_id=entity_name,
    entity_type=entity_type,
    description=description,
    source_id=source_id,
    file_path=file_path,
    created_at=int(time.time()),
    truncate=truncation_info,
)
await knowledge_graph_inst.upsert_node(
    entity_name,
    node_data=node_data,
)
```
#### 4.2.8 更新向量数据库信息
将实体名称，摘要添加到向量数据库中用于召回。
```python
entity_vdb_id = compute_mdhash_id(str(entity_name), prefix="ent-")
entity_content = f"{entity_name}\n{description}"
data_for_vdb = {
    entity_vdb_id: {
        "entity_name": entity_name,
        "entity_type": entity_type,
        "content": entity_content,
        "source_id": source_id,
        "file_path": file_path,
    }
}
await safe_vdb_operation_with_exception(
    operation=lambda payload=data_for_vdb: entity_vdb.upsert(payload),
    operation_name="entity_upsert",
    entity_name=entity_name,
    max_retries=3,
    retry_delay=0.1,
)
```
#### 4.2.9 返回最终实体节点信息
```python
node_data = dict(
    entity_id=entity_name,
    entity_type=entity_type,
    description=description,
    source_id=source_id,
    file_path=file_path,
    created_at=int(time.time()),
    truncate=truncation_info,
)
return node_data
```
### 4.3 合并全部关系（边）
其中 src_id 和 tgt_id 分别为关系的源节点和目标节点，edge_data 为节点之间的关系片段列表。
```python
async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage | None,
    entity_vdb: BaseVectorStorage | None,
    global_config: dict,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    added_entities: list = None,  # New parameter to track entities added during edge processing
    relation_chunks_storage: BaseKVStorage | None = None,
    entity_chunks_storage: BaseKVStorage | None = None,
):
```
#### 4.3.1 从当前知识图谱中获取已有关系
通过节点从当前知识库中获取已有关系信息，若节点之间存在已有关系，则提取关系相关信息
```python
# 1. Get existing edge data from graph storage
if await knowledge_graph_inst.has_edge(src_id, tgt_id):
    already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
    # Handle the case where get_edge returns None or missing fields
    if already_edge:
        # Get weight with default 1.0 if missing
        already_weights.append(already_edge.get("weight", 1.0))

        # Get source_id with empty string default if missing or None
        if already_edge.get("source_id") is not None:
            already_source_ids.extend(
                already_edge["source_id"].split(GRAPH_FIELD_SEP)
            )
        # Get file_path with empty string default if missing or None
        if already_edge.get("file_path") is not None:
            already_file_paths.extend(
                already_edge["file_path"].split(GRAPH_FIELD_SEP)
            )
        # Get description with empty string default if missing or None
        if already_edge.get("description") is not None:
            already_description.extend(
                already_edge["description"].split(GRAPH_FIELD_SEP)
            )
        # Get keywords with empty string default if missing or None
        if already_edge.get("keywords") is not None:
            already_keywords.extend(
                split_string_by_multi_markers(
                    already_edge["keywords"], [GRAPH_FIELD_SEP]
                )
            )
```
#### 4.3.2 合并 source id 列表
获取所有关系片段中的 source_id 列表，合并到已存在的 source_id 列表中。
```python
new_source_ids = [dp["source_id"] for dp in edges_data if dp.get("source_id")]
full_source_ids = merge_source_ids(existing_full_source_ids, new_source_ids)
```
#### 4.3.3 裁剪 edges_data 列表
和 4.2.3 裁剪 chunk 列表长度 类似，为了防止后续通过 LLM 进行摘要出现上下文溢出，需要限制 edges_data 列表长度，支持 KEEP 和 FIFO 策略，默认采用 KEEP 策略。
```python
 if limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP:
    allowed_source_ids = set(source_ids)
    filtered_edges = []
    for dp in edges_data:
        source_id = dp.get("source_id")
        # Skip relationship fragments sourced from chunks dropped by keep oldest cap
        if (
            source_id
            and source_id not in allowed_source_ids
            and source_id not in existing_full_source_ids
        ):
            continue
        filtered_edges.append(dp)
    edges_data = filtered_edges
else:  # In FIFO mode, keep all edges - truncation happens at source_ids level only
    edges_data = list(edges_data)

# 5. Check if we need to skip summary due to source_ids limit
if (
    limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP
    and len(existing_full_source_ids) >= max_source_limit
    and not edges_data
):
    if already_edge:
        logger.info(
            f"Skipped `{src_id}`~`{tgt_id}`: KEEP old chunks  {already_source_ids}/{len(full_source_ids)}"
        )
        existing_edge_data = dict(already_edge)
        return existing_edge_data
    else:
        logger.error(
            f"Internal Error: already_node missing for `{src_id}`~`{tgt_id}`"
        )
        raise ValueError(
            f"Internal Error: already_node missing for `{src_id}`~`{tgt_id}`"
        )
```
#### 4.3.4 合并相关字段
合并所有关系片段中的相关字段，包括 source_id, weight, keywords, description, file_path 等。
```python
# 6.1 Finalize source_id
source_id = GRAPH_FIELD_SEP.join(source_ids)

# 6.2 Finalize weight by summing new edges and existing weights
weight = sum([dp["weight"] for dp in edges_data] + already_weights)

# 6.2 Finalize keywords by merging existing and new keywords
all_keywords = set()
# Process already_keywords (which are comma-separated)
for keyword_str in already_keywords:
    if keyword_str:  # Skip empty strings
        all_keywords.update(k.strip() for k in keyword_str.split(",") if k.strip())
# Process new keywords from edges_data
for edge in edges_data:
    if edge.get("keywords"):
        all_keywords.update(
            k.strip() for k in edge["keywords"].split(",") if k.strip()
        )
# Join all unique keywords with commas
keywords = ",".join(sorted(all_keywords))

# 7. Deduplicate by description, keeping first occurrence in the same document
unique_edges = {}
for dp in edges_data:
    description_value = dp.get("description")
    if not description_value:
        continue
    if description_value not in unique_edges:
        unique_edges[description_value] = dp

# Sort description by timestamp, then by description length (largest to smallest) when timestamps are the same
sorted_edges = sorted(
    unique_edges.values(),
    key=lambda x: (x.get("timestamp", 0), -len(x.get("description", ""))),
)
sorted_descriptions = [dp["description"] for dp in sorted_edges]

# Combine already_description with sorted new descriptions
description_list = already_description + sorted_descriptions
if not description_list:
    logger.error(f"Relation {src_id}~{tgt_id} has no description")
    raise ValueError(f"Relation {src_id}~{tgt_id} has no description")
```
#### 4.3.5 对 description 列表进行摘要
```python
description, llm_was_used = await _handle_entity_relation_summary(
    "Relation",
    f"({src_id}, {tgt_id})",
    description_list,
    GRAPH_FIELD_SEP,
    global_config,
    llm_response_cache,
)
```
#### 4.3.6 添加/更新 图数据库和向量数据库中的实体信息
如果实体不存在，则添加到图数据库和向量数据库中；如果实体已存在，则更新图数据库和向量数据库中的实体信息。
```python
# graph db
node_data = {
    "entity_id": need_insert_id,
    "source_id": source_id,
    "description": description,
    "entity_type": "UNKNOWN",
    "file_path": file_path,
    "created_at": node_created_at,
    "truncate": "",
}
await knowledge_graph_inst.upsert_node(need_insert_id, node_data=node_data)
...
# vdb
vdb_data = {
    entity_vdb_id: {
        "content": entity_content,
        "entity_name": need_insert_id,
        "source_id": source_id,
        "entity_type": "UNKNOWN",
        "file_path": file_path,
    }
}
await safe_vdb_operation_with_exception(
    operation=lambda payload=vdb_data: entity_vdb.upsert(payload),
    operation_name="added_entity_upsert",
    entity_name=need_insert_id,
    max_retries=3,
    retry_delay=0.1,
)
```
#### 4.3.7 添加/更新 图数据库和向量数据库中的关系信息
如果关系不存在，则添加到图数据库和向量数据库中；如果关系已存在，则更新图数据库和向量数据库中的关系信息。
```python
# graph db
await knowledge_graph_inst.upsert_edge(
    src_id,
    tgt_id,
    edge_data=dict(
        weight=weight,
        description=description,
        keywords=keywords,
        source_id=source_id,
        file_path=file_path,
        created_at=edge_created_at,
        truncate=truncation_info,
    ),
)
# vdb
vdb_data = {
    rel_vdb_id: {
        "src_id": src_id,
        "tgt_id": tgt_id,
        "source_id": source_id,
        "content": rel_content,
        "keywords": keywords,
        "description": description,
        "weight": weight,
        "file_path": file_path,
    }
}
await safe_vdb_operation_with_exception(
    operation=lambda payload=vdb_data: relationships_vdb.upsert(payload),
    operation_name="relationship_upsert",
    entity_name=f"{src_id}-{tgt_id}",
    max_retries=3,
    retry_delay=0.2,
)
```
#### 4.3.8 返回最终关系信息
```python
edge_data = dict(
    src_id=src_id,
    tgt_id=tgt_id,
    description=description,
    keywords=keywords,
    source_id=source_id,
    file_path=file_path,
    created_at=edge_created_at,
    truncate=truncation_info,
    weight=weight,
)
eturn edge_data
```
### 4.3 KV 存储所有实体（节点）和关系（边）信息
将所有实体（节点）和关系（边）信息存储到 KV 数据库中，用于快速查询和展示。
```python
# store entities
await full_entities_storage.upsert(
    {
        doc_id: {
            "entity_names": list(final_entity_names),
            "count": len(final_entity_names),
        }
    }
)
# store relations
await full_relations_storage.upsert(
    {
        doc_id: {
            "relation_pairs": [
                list(pair) for pair in final_relation_pairs
            ],
            "count": len(final_relation_pairs),
        }
    }
)
```
## 5. 文档最终状态更新
更新文档的最终状态为已处理（PROCESSED）或失败（FAILED），并记录处理的 chunk 数量、处理时间、文档摘要等信息。
```python
self.doc_status.upsert(
    {
        doc_id: {
            "status": DocStatus.PROCESSED,
            "chunks_count": len(chunks),
            "chunks_list": list(chunks.keys()),
            "content_summary": status_doc.content_summary,
            "content_length": status_doc.content_length,
            "created_at": status_doc.created_at,
            "updated_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "file_path": file_path,
            "track_id": status_doc.track_id,  # Preserve existing track_id
            "metadata": {
                "processing_start_time": processing_start_time,
                "processing_end_time": processing_end_time,
            },
        }
    }
)
```
# 尾言
至此，我们已完整拆解了 LightRAG 文档解析的全流程。这套设计确保了其在“轻量”的定位下，依然能构建一个扩展性强、管控精细、成本可控的 GraphRAG 入库管线。

然而，优质的文档入库只是 RAG 系统的“生产侧”，是支撑高效召回的基石。接下来，我们将转向至关重要的“消费侧”——文本召回。如果您关心 LightRAG 在真实问答中的实战能力，下一篇章将是决定性的展现。