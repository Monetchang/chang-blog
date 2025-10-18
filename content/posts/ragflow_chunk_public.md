---
title: "【解密源码】 RAGFlow 切分最佳-实践公共部分"
date: 2025-10-16T20:39:10+08:00
draft: true
tags: ["源码","技术",RAG]
categories: ["RAGFlow"]
---

# 引言

在大模型落地的趋势下，RAG（Retrieval-Augmented Generation）技术逐渐成为企业级应用的核心。RAGFlow 是一个工程化的 RAG 工作流框架，能帮助开发者快速构建知识检索 + 问答系统。本系列文章将分阶段拆解 RAGFlow 的源码，实现从入门到深入的学习。本期目标是快速跑通 RAGFlow Demo，并对框架做一个整体初探。

# 省流版

代码文件阅读顺序：

原理：

# 手撕版

在提供给前端的接口中，有个 /upload_and_parse 接口，通过接口语义可以知道这个接口的功能是用来接收上传文档并进行解析的

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

## PART 1 通过 *conversation_id 获取关联的 Knowledgebase（知识库）*

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

TODO：知识库在 RAGFlow 中的含义，这里不做重点介绍

## PART 2 将 file 存储到对应的 *Knowledgebase 中，并返回相应 file信息*

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

跳转 upload_document 函数可以看到 files 中元素的结构体，其中重点关注 parser_id 字段，文件的parser_id 字段与 k*nowledgebase（知识库）的 parser_id 相关联。*

## PART 3 通过对应解析器和配置对不同格式文件进行解析

```python
FACTORY = {
    ParserType.PRESENTATION.value: presentation,
    ParserType.PICTURE.value: picture,
    ParserType.AUDIO.value: audio,
    ParserType.EMAIL.value: email
}
parser_config = {"chunk_token_num": 4096, "delimiter": "\n!?;。；！？", "layout_recognize": "Plain Text"}
exe = ThreadPoolExecutor(max_workers=12)
threads = []
doc_nm = {}
for d, blob in files:
    doc_nm[d["id"]] = d["name"]
for d, blob in files:
    kwargs = {
        "callback": dummy,
        "parser_config": parser_config,
        "from_page": 0,
        "to_page": 100000,
        "tenant_id": kb.tenant_id,
        "lang": kb.language
    }
    threads.append(exe.submit(FACTORY.get(d["parser_id"], naive).chunk, d["name"], blob, **kwargs))
```

通过 FACTORY 字典可以看到不同的 ParserType 值对应不同的解析方式 Presentation（PPT）， Picture（图片），Audio（音频），Email（邮件），如果没有匹配，就用默认的 Naive 解析。

## PART 4 处理切分结果

```python
for (docinfo, _), th in zip(files, threads):
    docs = []
    for ck in th.result():
        d = deepcopy(doc)
        d.update(ck)
        d["id"] = xxhash.xxh64((ck["content_with_weight"] + str(d["doc_id"])).encode("utf-8")).hexdigest()
```

每个 chunk 会生成一个文档片段，赋予唯一 id，以及其他 metadata

如果 chunk 存在图片，则会将图片转换成字节流的形式单独存储，并建立索引

```python
output_buffer = BytesIO()
if isinstance(d["image"], bytes):
    output_buffer = BytesIO(d["image"])
else:
    d["image"].save(output_buffer, format='JPEG')

STORAGE_IMPL.put(kb.id, d["id"], output_buffer.getvalue())
d["img_id"] = "{}-{}".format(kb.id, d["id"])
```

如果不是图片类型文档，则会调用工具生成思维导图的作为相关内容进行存储（TODO：思维导图生成）

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

## PART 5 结果向量化存储

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
assert len(cks) == len(vects)
for i, d in enumerate(cks):
		v = vects[i]
		d["q_%d_vec" % len(v)] = v
```

将 chunk 通过 embedding 模型进行向量化并存储在 d 结构体中。

```python
idxnm = search.index_name(kb.tenant_id)
try_create_idx = True
for b in range(0, len(cks), es_bulk_size):
	  if try_create_idx:
	      if not settings.docStoreConn.indexExist(idxnm, kb_id):
	          settings.docStoreConn.createIdx(idxnm, kb_id, len(vects[0]))
	      try_create_idx = False
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

## PART 6 返回结果

```python
DocumentService.increment_chunk_num(
    doc_id, kb.id, token_counts[doc_id], chunk_counts[doc_id], 0)
return [d["id"] for d, _ in files]
```

更新 chunk 信息，并返回对应上传文档的 id 列表。

