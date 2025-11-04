---
title: "【解密源码】 RAGFlow 召回策略全解"
date: 2025-10-16T20:39:10+08:00
draft: false
tags: ["源码","技术","RAG"]
categories: ["RAGFlow"]
---

# 引言

# 省流版

# 手撕版

## 查询配置初始化


## 重排序分页策略
## 构建搜索参数
```python
RERANK_LIMIT = math.ceil(64/page_size) * page_size if page_size>1 else 1
req = {"kb_ids": kb_ids, "doc_ids": doc_ids, "page": math.ceil(page_size*page/RERANK_LIMIT), "size": RERANK_LIMIT,
        "question": question, "vector": True, "topk": top,
        "similarity": similarity_threshold,
        "available_int": 1}
```

## 执行搜索

```python
sres = self.search(req, [index_name(tid) for tid in tenant_ids],
                    kb_ids, embd_mdl, highlight, rank_feature=rank_feature)
```
### 查询配置初始化
```python
def search(self, req, idx_names: str | list[str],
           kb_ids: list[str],
           emb_mdl=None,
           highlight: bool | list | None = None,
           rank_feature: dict | None = None):
    
    if highlight is None:
        highlight = False

    filters = self.get_filters(req)  # 获取过滤条件
    orderBy = OrderByExpr()  # 排序表达式

    pg = int(req.get("page", 1)) - 1 # 页码（从0开始）
    topk = int(req.get("topk", 1024)) # 初始召回数量
    ps = int(req.get("size", topk)) # 返回结果大小
    offset, limit = pg * ps, ps # 计算偏移量
    # 默认返回字段列表
    src = req.get("fields",
                    ["docnm_kwd", "content_ltks", "kb_id", "img_id", "title_tks", "important_kwd", "position_int",
                    "doc_id", "page_num_int", "top_int", "create_timestamp_flt", "knowledge_graph_kwd",
                    "question_kwd", "question_tks", "doc_type_kwd",
                    "available_int", "content_with_weight", PAGERANK_FLD, TAG_FLD])
```

### 无查询词的简单搜索
针对没有查询问题的场景，返回对应的 topk chunk。
```python
qst = req.get("question", "")

if not qst:
    if req.get("sort"):
        orderBy.asc("page_num_int")        
        orderBy.asc("top_int")  
        orderBy.desc("create_timestamp_flt")
    
    # 执行无查询条件的搜索
    res = self.dataStore.search(src, [], filters, [], orderBy, offset, limit, idx_names, kb_ids)
    total = self.dataStore.getTotal(res)
```

### 有查询词的智能搜索
#### 解析查询问题
```python
matchText, keywords = self.qryr.question(qst, min_match=0.3)

def question(self, txt, tbl="qa", min_match: float = 0.6):
    ...
```
**1. 规范查询问题文本格式**

**在英文和中文文本之间自动添加空格，使文本格式更加规范，提高可读性。**
```python
txt = FulltextQueryer.add_space_between_eng_zh(txt)

def add_space_between_eng_zh(txt):
    # (ENG/ENG+NUM) + ZH
    txt = re.sub(r'([A-Za-z]+[0-9]+)([\u4e00-\u9fa5]+)', r'\1 \2', txt)
    # ENG + ZH
    txt = re.sub(r'([A-Za-z])([\u4e00-\u9fa5]+)', r'\1 \2', txt)
    # ZH + (ENG/ENG+NUM)
    txt = re.sub(r'([\u4e00-\u9fa5]+)([A-Za-z]+[0-9]+)', r'\1 \2', txt)
    txt = re.sub(r'([\u4e00-\u9fa5]+)([A-Za-z])', r'\1 \2', txt)
    return txt
```
**全角转半角、繁体转简体、小写、去除标点。**
```python
txt = re.sub(
    r"[ :|\r\n\t,，。？?/`!！&^%%()\[\]{}<>]+",
    " ",
    rag_tokenizer.tradi2simp(rag_tokenizer.strQ2B(txt.lower())),
).strip()
```
**移除对查询核心语义影响不大的辅助词汇，保留关键信息。**
```python
txt = FulltextQueryer.rmWWW(txt)
def rmWWW(txt):
    patts = [
        (
            r"是*(怎么办|什么样的|哪家|一下|那家|请问|啥样|咋样了|什么时候|何时|何地|何人|是否|是不是|多少|哪里|怎么|哪儿|怎么样|如何|哪些|是啥|啥是|啊|吗|呢|吧|咋|什么|有没有|呀|谁|哪位|哪个)是*",
            "",
        ),
        (r"(^| )(what|who|how|which|where|why)('re|'s)? ", " "),
        (
            r"(^| )('s|'re|is|are|were|was|do|does|did|don't|doesn't|didn't|has|have|be|there|you|me|your|my|mine|just|please|may|i|should|would|wouldn't|will|won't|done|go|for|with|so|the|a|an|by|i'm|it's|he's|she's|they|they're|you're|as|by|on|in|at|up|out|down|of|to|or|and|if) ",
            " ")
    ]
    otxt = txt
    for r, p in patts:
        txt = re.sub(r, p, txt, flags=re.IGNORECASE)
    if not txt:
        txt = otxt
    return txt
```

**2. 英文查询处理**

**分词后进行分词的权重计算。**
```python
if not self.isChinese(txt):
    txt = FulltextQueryer.rmWWW(txt)
    tks = rag_tokenizer.tokenize(txt).split()
    keywords = [t for t in tks if t]
    # 分词权重计算
    tks_w = self.tw.weights(tks, preprocess=False)
    tks_w = [(re.sub(r"[ \\\"'^]", "", tk), w) for tk, w in tks_w]
    tks_w = [(re.sub(r"^[a-z0-9]$", "", tk), w) for tk, w in tks_w if tk]
    tks_w = [(re.sub(r"^[\+-]", "", tk), w) for tk, w in tks_w if tk]
    tks_w = [(tk.strip(), w) for tk, w in tks_w if tk.strip()]
```
`self.tw.weights` 是计算分词权重的核心方法。这里简单介绍一下其中使用到的权重策略。
- `ner(t)`：根据分词是否为数字、短字母或特定类型的命名实体（如公司名、地名、学校名等）分配不同权重。
- `postag(t)`：根据分词的词性（连词，代词等）分配不同权重，反映不同词性在文本中的重要性差异。
- `freq(t)`：计算分词的频率特征，并对未识别的分词和长词进行特殊处理。
- `df(t)`：计算分词在文档集合中的分布情况，反映分词的文档间区分度。
- `idf(s, N)`：经典的 IDF 计算公式，用于衡量分词的重要性，其中 N 是文档总数，s 是包含该分词的文档数。
  
权重计算输出示例
```
[('Chinese', 0.35), ('economy', 0.28), ('development', 0.22), ('quickly', 0.15)]
```

**分词同义词扩展，并赋予 1/4 分词权重。**
```python
for tk, w in tks_w[:256]:
    # 分词同义词查询
    syn = self.syn.lookup(tk)
    syn = rag_tokenizer.tokenize(" ".join(syn)).split()
    keywords.extend(syn)
    syn = ["\"{}\"^{:.4f}".format(s, w / 4.) for s in syn if s.strip()]
    syns.append(" ".join(syn))
```
`self.syn.lookup(tk)` 是实现分词同义词查找的核心方法。主要通过词表来进行同义词的查询，英文使用 wordnet 词库，中文使用自构建的词库。

**结合权重，同义词信息，构建查询表达式**
```python
# 1. 单个词 + 同义词查询
q = [
    "({}^{:.4f}".format(tk, w) + " {})".format(syn) 
    for (tk, w), syn in zip(tks_w, syns) 
    if tk and not re.match(r"[.^+\(\)-]", tk)
]

# 2. 相邻词短语查询（提升相邻词权重）
for i in range(1, len(tks_w)):
    left, right = tks_w[i - 1][0].strip(), tks_w[i][0].strip()
    if not left or not right:
        continue
    q.append(
        '"%s %s"^%.4f' % (
            tks_w[i - 1][0],
            tks_w[i][0],
            max(tks_w[i - 1][1], tks_w[i][1]) * 2,  # 短语权重加倍
        )
    )
```

**3. 构建最终查询参数**
```python
return MatchTextExpr(
    self.query_fields, query, 100
), keywords

# MatchTextExpr 结构
{
    "fields": [
        "title_tks^10",
        "title_sm_tks^5",
        "important_kwd^30",
        "important_tks^20",
        "question_tks^20",
        "content_ltks^2",
        "content_sm_ltks",
    ] # 在这些对应字段进行查询
    "query": "" # 查询表达式
    "topn": 100 # 返回结果数
    "extra_options": "" # 其他配置
}
```

**4. 中文查询处理**

**与英文的处理流程大致相同。**
```python
# 前置处理，按照空格对中文进行分词
for tt in self.tw.split(txt)[:256]:
    if not tt:
        continue
    keywords.append(tt)
    # 分词权重计算
    twts = self.tw.weights([tt])
    # 同义词查询
    syns = self.syn.lookup(tt)
    # 限制同义词查找结果
    if syns and len(keywords) < 32:
        keywords.extend(syns)
    tms = []
    for tk, w in sorted(twts, key=lambda x: x[1] * -1):
        sm = (
            # 精细分词 
            rag_tokenizer.fine_grained_tokenize(tk).split()
            if need_fine_grained_tokenize(tk)
            else []
        )
    ...
    # 构建最终查询参数
    query = " OR ".join([f"({t})" for t in qs if t])
    if not query:
        query = otxt
    return MatchTextExpr(
        self.query_fields, query, 100, {"minimum_should_match": min_match}
    ), keywords
```

**最终查询问题解析输出：**
```python
matchText, keywords = self.qryr.question(qst, min_match=0.3)
# 查询信息结构体，同上方 MatchTextExpr 结构
{
    "fields": [
        "title_tks^10",
        "title_sm_tks^5",
        "important_kwd^30",
        "important_tks^20",
        "question_tks^20",
        "content_ltks^2",
        "content_sm_ltks",
    ] # 在这些对应字段进行查询
    "query": "" # 查询表达式
    "topn": 100 # 返回结果数
    "extra_options": "" # 其他配置
}
# keywords 关键字列表
keywords = []
```

#### 稀疏检索
没有设置 Embedding model，通过查询语句和筛选项进行检索。
```python
if emb_mdl is None:
    # 查询语句
    matchExprs = [matchText]
    
    res = self.dataStore.search(
        src, highlightFields, filters, matchExprs, orderBy, 
        offset, limit, idx_names, kb_ids, rank_feature=rank_feature
    )
    total = self.dataStore.getTotal(res)
```

#### 混合检索（稠密+稀疏）
先获取查询文本的向量表示
```python
matchDense = self.get_vector(qst, emb_mdl, topk, req.get("similarity", 0.1))
q_vec = matchDense.embedding_data
```
创建检索表达式，分词匹配权重 5%，向量匹配权重 95%，进行检索。
```python
fusionExpr = FusionExpr("weighted_sum", topk, {"weights": "0.05,0.95"})
matchExprs = [matchText, matchDense, fusionExpr]
res = self.dataStore.search(src, highlightFields, filters, matchExprs, orderBy, offset, limit,
                            idx_names, kb_ids, rank_feature=rank_feature)
total = self.dataStore.getTotal(res)
```

#### 空结果回退策略
当通过上述方案未检索到任何结果，则尝试放宽条件重新搜索

如果过滤条件中有指定文档 id，则进入无查询词的简单搜索返回 limit 切块。
```python
if total == 0:
    if filters.get("doc_id"):
        res = self.dataStore.search(src, [], filters, [], orderBy, offset, limit, idx_names, kb_ids)
        total = self.dataStore.getTotal(res)
```

调整匹配阈值，分词匹配度 30% -> 10%，向量匹配度 0.1 -> 0.17。
```python
matchText, _ = self.qryr.question(qst, min_match=0.1)
matchDense.extra_options["similarity"] = 0.17

res = self.dataStore.search(
    src, highlightFields, filters, [matchText, matchDense, fusionExpr],
    orderBy, offset, limit, idx_names, kb_ids, rank_feature=rank_feature
)
total = self.dataStore.getTotal(res)
```

## 重排序
如果指定重排序模型，检索分块数大于 0，则使用指定重排序模型对结果进行重排序。
```python
if rerank_mdl and sres.total > 0:
    sim, tsim, vsim = self.rerank_by_model(rerank_mdl,
                                            sres, question, 1 - vector_similarity_weight,
                                            vector_similarity_weight,
                                            rank_feature=rank_feature)
```

如果没有指定重排序模型，则根据搜索引擎进行重排序逻辑
```python
lower_case_doc_engine = os.getenv('DOC_ENGINE', 'elasticsearch')
if lower_case_doc_engine == "elasticsearch":
    # ElasticSearch rerank
    sim, tsim, vsim = self.rerank(
        sres, question, 1 - vector_similarity_weight, vector_similarity_weight,
        rank_feature=rank_feature)
else:
    # Infinity 在融合前会对每种方式的分数进行标准化，所以这里无需重新排序
    sim = [sres.field[id].get("_score", 0.0) for id in sres.ids]
    sim = [s if s is not None else 0. for s in sim]
    tsim = sim
    vsim = sim
```

### 指定模型重排序
```python
sim, tsim, vsim = self.rerank_by_model(rerank_mdl,
                                            sres, question, 1 - vector_similarity_weight,
                                            vector_similarity_weight,
                                            rank_feature=rank_feature)
```

### ES 重排序
```python
sim, tsim, vsim = self.rerank(
        sres, question, 1 - vector_similarity_weight, vector_similarity_weight,
        rank_feature=rank_feature)
```