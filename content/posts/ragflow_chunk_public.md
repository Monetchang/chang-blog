---
title: "【解密源码】 RAGFlow 切分最佳实践-上传与解析全流程"
date: 2025-10-16T20:39:10+08:00
draft: true
tags: ["源码","技术",RAG]
categories: ["RAGFlow"]
---

*本系列文章带你从源码角度深度剖析 RAGFlow，从文件上传、解析、切分、向量化到最终入库。本文聚焦于文档解析与切分的全流程概述，为理解整个 RAGFlow 流程打下基础。*

# 引言

随着大模型在企业应用中的落地加速，RAG（Retrieval-Augmented Generation）技术逐渐成为知识问答系统的核心。
RAGFlow 是一个面向工程化的 RAG 工作流框架，它提供了从文档解析、向量化到检索问答的一整套流程，帮助开发者快速构建可扩展的知识增强系统。

本篇文章属于《RAGFlow 源码解密》系列第一期，目标是：理解文档上传与解析的整体流程。

# 省流版（快速理解核心逻辑）

**如果你只想快速理解流程，这里是最核心的内容👇**

**核心目标**：搞懂 RAGFlow 如何将你上传的一个原始文档（如PDF、PPT），变成可以被检索和问答的知识片段。

**全过程六步流程图**：
`0. 用户上传文档` → `1. 定位知识库` → `2. 存储并分析文档` → `3. 调用解析器进行智能切块` → `4. 为文本/图片/思维导图生成索引` → `5. 向量化并存入向量数据库` → `6. 返回文档ID`

**上传文档**
- **功能**：接收用户上传的 PDF / DOCX / Markdown / 图片文件，自动识别文件类型与编码格式。  
- **核心函数 / 调用点**：`/upload_and_parse`  

**绑定知识库**
- **功能**：根据用户上下文或选择确定目标知识库，实现文档与知识库的绑定。  
- **核心函数 / 调用点**：  
  - `ConversationService.get_by_id()`  
  - `KnowledgebaseService.get_by_id()`  

**文件存储**
- **功能**：将文档保存至对象存储（MinIO / S3 等），提取基础元数据（类型、大小、页数等）。  
- **核心函数 / 调用点**：`FileService.upload_document()`  

**内容解析**
- **功能**：根据文件类型调用对应解析器，将内容切分为语义块（chunks），生成结构化内容。  
- **核心函数 / 调用点**：`FACTORY.get(...).chunk()`  
- **设计亮点**：  
  - 对非图片类文档，系统会自动生成 **思维导图（MindMap）** 作为关联内容进行存储。  
  - 可扩展多种解析器（PDFParser、MarkdownParser、DocxParser、ImageParser）。

**索引生成**
- **功能**：为每个内容块创建唯一索引，支持文本、图片、思维导图等多模态内容统一检索。  
- **核心函数 / 调用点**：内部索引构建逻辑（`IndexService.build_index()`）  

**向量化入库**
- **功能**：调用 Embedding 模型将文本块转换为向量，并存入向量数据库（ES / infinity / opensearch）。  
- **核心函数 / 调用点**：  
  - `LLMBundle.encode()`  
  - `docStoreConn.insert()`  

**返回结果**
- **功能**：返回文档唯一 ID、向量数量及状态信息，表示入库完成。  
- **核心函数 / 调用点**：`DocumentService.increment_chunk_num()`  

# 手撕版（源码深解）

**接下来，我们将深入代码细节，逐一拆解"省流版"中的每一步**

## 入口与路由

在提供给前端的接口中，有个 `/upload_and_parse` 接口，通过接口语义可以知道这个接口的功能是用来接收上传文档并进行解析的

```python
@manager.route("/upload_and_parse", methods=["POST"])  # noqa: F821
@login_required
@validate_request("conversation_id")
def upload_and_parse():
  if "file" not in request.files:
      return get_json_result(data=False, message="No file part!", code=settings.RetCode.ARGUMENT_ERROR)

  file_objs = request.files.getlist("file")
  for file_obj in file_objs:
      if file_obj.filename == "":
          return get_json_result(data=False, message="No file selected!", code=settings.RetCode.ARGUMENT_ERROR)

  doc_ids = doc_upload_and_parse(request.form.get("conversation_id"), file_objs, current_user.id)

  return get_json_result(data=doc_ids)
```

整个函数中最主要的函数是 `doc_upload_and_parse`

跳转到 `doc_upload_and_parse` 函数后，可以看到第一部分代码

## 知识库关联
通过 conversation_id 获取关联的 Knowledgebase（知识库）

```python
e, conv = ConversationService.get_by_id(conversation_id)
if not e:
    e, conv = API4ConversationService.get_by_id(conversation_id)
assert e, "Conversation not found!"

e, dia = DialogService.get_by_id(conv.dialog_id)
if not dia.kb_ids:
    raise LookupError("No knowledge base associated with this conversation. "
                      "Please add a knowledge base before uploading documents")
kb_id = dia.kb_ids[0]
e, kb = KnowledgebaseService.get_by_id(kb_id)
if not e:
    raise LookupError("Can't find this knowledgebase!")

```

💡 说明：知识库在 RAGFlow 中的含义，这里不做重点介绍，可以理解一个独立的知识集合，包括多个文档，每个对话（Conversation）都绑定一个或多个知识库，以限定检索范围。。

## 文件存储与登记
文件上传逻辑由`FileService.upload_document()`实现。将文件存储到对应的知识库中，并返回相应文件信息。

```python
err, files = FileService.upload_document(kb, file_objs, user_id)
```

重点关注上传后返回的 files 结构体

```python
doc = {
    "id": doc_id,
    "kb_id": kb.id,
    "parser_id": self.get_parser(filetype, filename, kb.parser_id),
    "parser_config": kb.parser_config,
    "created_by": user_id,
    "type": filetype,
    "name": filename,
    "suffix": Path(filename).suffix.lstrip("."),
    "location": location,
    "size": len(blob),
    "thumbnail": thumbnail_location,
}
DocumentService.insert(doc)
```

🔍 关键点：
- parser_id 决定文件使用哪种解析器。
- 每个知识库可配置默认解析器（PDF、图片、音频等类型各不同）。

## 核心解析与分块
通过解析器工厂（FACTORY）动态选择不同的解析器，对不同格式文件进行解析。

```python
FACTORY = {
    ParserType.PRESENTATION.value: presentation,
    ParserType.PICTURE.value: picture,
    ParserType.AUDIO.value: audio,
    ParserType.EMAIL.value: email
}
```

主循环逻辑如下：
```python
parser_config = {"chunk_token_num": 4096, "delimiter": "\n!?;。；！？", "layout_recognize": "Plain Text"}
...
for d, blob in files:
    kwargs = {
        "callback": dummy,
        "parser_config": parser_config,
        "from_page": 0,
        "to_page": 100000,
        "tenant_id": kb.tenant_id,
        "lang": kb.language
    }
    threads.append(exe.submit(
        FACTORY.get(d["parser_id"], naive).chunk,
        d["name"], blob, **kwargs
    ))

```

通过 FACTORY 字典可以看到不同的 ParserType 值对应不同的解析方式 Presentation（PPT）， Picture（图片），Audio（音频），Email（邮件），如果没有匹配，就用默认的 Naive 解析。

## 内容增强与索引
每个 chunk 会生成一个文档片段，赋予唯一 id，以及其他 metadata。

```python
for (docinfo, _), th in zip(files, threads):
    docs = []
    for ck in th.result():
        d = deepcopy(doc)
        d.update(ck)
        d["id"] = xxhash.xxh64((ck["content_with_weight"] + str(d["doc_id"])).encode("utf-8")).hexdigest()
```

如果 chunk 存在图片，则会将图片转换成字节流的形式单独存储，并建立索引。

```python
output_buffer = BytesIO()
if isinstance(d["image"], bytes):
    output_buffer = BytesIO(d["image"])
else:
    d["image"].save(output_buffer, format='JPEG')

STORAGE_IMPL.put(kb.id, d["id"], output_buffer.getvalue())
d["img_id"] = "{}-{}".format(kb.id, d["id"])
```

如果不是图片类型文档，则会调用工具生成思维导图的作为相关内容进行存储。

```python
if parser_ids[doc_id] != ParserType.PICTURE.value:
from graphrag.general.mind_map_extractor import MindMapExtractor
mindmap = MindMapExtractor(llm_bdl)
try:
    mind_map = trio.run(mindmap, [c["content_with_weight"] for c in docs if c["doc_id"] == doc_id])
    mind_map = json.dumps(mind_map.output, ensure_ascii=False, indent=2)
    if len(mind_map) < 32:
        raise Exception("Few content: " + mind_map)
    cks.append({
        "id": get_uuid(),
        "doc_id": doc_id,
        "kb_id": [kb.id],
        "docnm_kwd": doc_nm[doc_id],
        "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", doc_nm[doc_id])),
        "content_ltks": rag_tokenizer.tokenize("summary summarize 总结 概况 file 文件 概括"),
        "content_with_weight": mind_map,
        "knowledge_graph_kwd": "mind_map"
    })
```

## 向量化与持久化

将 chunk 通过 embedding 模型进行向量化并存储在 d 结构体中。
```python
embd_mdl = LLMBundle(kb.tenant_id, LLMType.EMBEDDING, llm_name=kb.embd_id, lang=kb.language)
def embedding(doc_id, cnts, batch_size=16):
	  nonlocal embd_mdl, chunk_counts, token_counts
	  vects = []
	  for i in range(0, len(cnts), batch_size):
	      vts, c = embd_mdl.encode(cnts[i: i + batch_size])
	      vects.extend(vts.tolist())
	      chunk_counts[doc_id] += len(cnts[i:i + batch_size])
	      token_counts[doc_id] += c
	  return vects
vects = embedding(doc_id, [c["content_with_weight"] for c in cks])
```
随后写入向量数据库（如 Elasticsearch）：

```python
if not settings.docStoreConn.indexExist(idxnm, kb_id):
    settings.docStoreConn.createIdx(idxnm, kb_id, len(vects[0]))
	      
settings.docStoreConn.insert(cks[b:b + es_bulk_size], idxnm, kb_id)
```

在向量数据库中创建索引进行存储，以下是系统内置支持的向量数据库，在系统初始化时默认使用的是 Elasticsearch。

```python
DOC_ENGINE = os.environ.get("DOC_ENGINE", "elasticsearch")
# DOC_ENGINE = os.environ.get('DOC_ENGINE', "opensearch")
lower_case_doc_engine = DOC_ENGINE.lower()
if lower_case_doc_engine == "elasticsearch":
    docStoreConn = rag.utils.es_conn.ESConnection()
elif lower_case_doc_engine == "infinity":
    docStoreConn = rag.utils.infinity_conn.InfinityConnection()
elif lower_case_doc_engine == "opensearch":
    docStoreConn = rag.utils.opensearch_conn.OSConnection()
else:
    raise Exception(f"Not supported doc engine: {DOC_ENGINE}")
```

## 返回结果
更新 chunk 信息，并返回对应上传文档的 id 列表。

```python
DocumentService.increment_chunk_num(
    doc_id, kb.id, token_counts[doc_id], chunk_counts[doc_id], 0)
return [d["id"] for d, _ in files]
```
# 下期预告
下期我们将正式走进 RAGFlow 的核心解析器体系，聚焦默认的 Naive Parser。