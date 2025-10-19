---
title: "【解密源码】 RAGFlow 切分最佳实践-navie 分词器原理"
date: 2025-10-17T10:39:10+08:00
draft: true
tags: ["源码","技术",RAG]
categories: ["RAGFlow"]
---

# 引言

通过上一章《上传与解析全流程》中提到的 FACTORY.get(d["parser_id"], naive).chunk，我们知道 RAGFlow 的文档解析逻辑核心在 rag/app 下的多个解析器文件中。

本期我们聚焦最基础的 naive 方案，对其分词器（Tokenizer）的原理进行详细拆解。这一模块作为 **navie parser 所有文档类型解析流程的公共组件**，承担了语义分段、文本归一化、chunk 边界控制等底层职责，是后续所有文件类型解析逻辑的基础。

# 省流版（快速理解核心逻辑）

**解析前配置**
- 设置语言（如 English / Chinese）  
- 设定分块 Token 数上限（默认 512）  
- 指定分隔符（`\n!?。；！？`）与识别策略（DeepDOC）  
- 初始化基础与精细分词器：`tokenize` 和 `fine_grained_tokenize`

**基础分词器 `tokenize`**  
- **文本预处理**：移除非字母数字字符 → 全角转半角 → 繁体转简体  
- **语言切分**：使用正则表达式区分中文、英文及标点序列  
- **非中文处理**：使用 `nltk` 完成英文的词形还原与词干提取  
- **中文处理**：采用 **正向最大匹配（MM） + 反向最大匹配（RMM）** 双向分词  
- **冲突解决**：使用 DFS 穷举 + 多维度评分机制，选取最优分词路径  
  > 💡 评分标准包括：惩罚过度切分、长词比例、词频加权等维度

**精细分词器 `fine_grained_tokenize`**  
- 对 `tokenize` 输出结果进行再次细化  
- 中文占比低（<20%）时使用轻量切分策略  
- 对短词（<3）和纯数字直接保留  
- 对长度适中（≤10）的词使用 DFS 进行多方案分词并排序选优  
- 对长词（>10）直接保留，避免破坏固定搭配  
- 英文短词不分割，短语使用空格分割

**设计亮点**
- **双向匹配 + DFS + 评分机制** 的融合方案平衡了精度与性能  
- **语言自适应切分**：自动识别中文、英文、数字混排  
- **精细二次分词** 提高 RAG 检索上下文的一致性与召回率


# 手撕版（源码深解）
分词器配置初始化
```python
is_english = lang.lower() == "english"  # is_english(cks)
parser_config = kwargs.get(
	  "parser_config", {
      "chunk_token_num": 512, "delimiter": "\n!?。；！？", "layout_recognize": "DeepDOC"})
doc = {
	  "docnm_kwd": filename,
	  "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename))
}
doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])
```

进行解析前的基础配置，包括语言，chunk token 数，分隔符，识别策略，分词器 tokenize。这里重点看下分词器的实现，可以看到这里有存在两个分词器 rag_tokenizer.tokenize 和 rag_tokenizer.fine_grained_tokenize，可以理解成基础分词器和二次分词器。

## tokenize

文本前置处理

```python
line = re.sub(r"\W+", " ", line) # 移除非字母数字字符，用空格替换
line = self._strQ2B(line).lower() # 全角转半角，并转为小写
line = self._tradi2simp(line) # 繁体中文转简体中文
```

根据语言进行初步切分，切分规则 `r"([ ,\.<>/?;:'\[\]\\`!@#$%^&*\(\)\{\}\|_+=《》，。？、；‘’：“”【】~！￥%……（）——-]+|[a-zA-Z0-9,\.-]+)"`

```python
arr = self._split_by_lang(line)
```


1. **标点符号分割**
    - **英文标点**：`,.<>/?;:'[]\`!@#$%^&*(){}|_+-=`
    - **中文标点**：`《》，。？、；‘’：“”【】~！￥%……（）——`
    - **匹配模式**：一个或多个连续的标点符号
2. **英文数字序列**
    - **字母**：a-z, A-Z
    - **数字**：0-9
    - **特定符号**：逗号、点号、连字符
    - **匹配模式**：一个或多个连续的英文数字字符

如果输入语言非中文，使用 nltk 库提取文本，通过分词，词干提取和词性还原三个步骤处理非中文文本。对短文本，纯英文和纯数字三种情况的文本进行直接保留。
```python
for L, lang in arr:
    if not lang:  # 非中文文本（英文等）
        # 使用NLTK进行英文分词、词形还原和词干提取
        res.extend([self.stemmer.stem(self.lemmatizer.lemmatize(t)) 
                   for t in word_tokenize(L)])
        continue
    
    if len(L) < 2 or re.match(r"[a-z\.-]+$", L) or re.match(r"[0-9\.-]+$", L):
        res.append(L)  # 短文本、纯英文或纯数字直接保留
        continue
```

使用  datrie 库对文本使用正，反双向匹配分词方案。

```python
tks, s = self.maxForward_(L) # 正向最大匹配分词
tks1, s1 = self.maxBackward_(L) # 反向最大匹配分词
```

1. **正向分词的优点：**
    - **符合阅读习惯**：从左到右，与人眼阅读方向一致
    - **实现简单**：逻辑直观，易于理解和调试
    - **效率较高**：通常比反向分词稍快
2. **反向分词的优点：**
    - **解决歧义能力强**：对某些结构能获得更准确的结果
    - **处理未登录词更好**：对后缀丰富的语言更有效

```python
while i + same < len(tks1) and j + same < len(tks) and tks1[i + same] == tks[j + same]:
    same += 1
if same > 0:
    res.append(" ".join(tks[j: j + same]))
...

self.dfs_("".join(tks[_j:j]), 0, [], tkslist)
res.append(" ".join(self.sortTks_(tkslist)[0][0]))

```

对正，反两种分词结果，分词相同的部分进行合并；分词有歧义的部分使用 DFS 算法穷举出所有分词路径，后对各种分词路径进行多维度评分，评分基于惩罚过度细分的分词结果，长词比例项，词频等维度，消除分词分歧，返回最终的分词结果。

### fine_grained_tokenize

对经过基础分词器 tokenize 的分词结果再次进行精细分词。

文本中中文占比小于 20% 的文本采用简单处理。

```python
def fine_grained_tokenize(self, tks):
    tks = tks.split()
    zh_num = len([1 for c in tks if c and is_chinese(c[0])])
    if zh_num < len(tks) * 0.2:
        res = []
        for tk in tks:
            res.extend(tk.split("/"))
        return " ".join(res)
```


对短词短语直接保留

```python
for tk in tks:
    if len(tk) < 3 or re.match(r"[0-9,\.-]+$", tk):
        res.append(tk)  # 直接保留
        continue
```

长度大于 10 的认为是固定搭配，不做处理，小于 10 的通过 DFS 分词 + 评分处理。

```python
tkslist = []
if len(tk) > 10:
    tkslist.append(tk)  # 超长词不进一步切分
else:
    self.dfs_(tk, 0, [], tkslist)  # 使用DFS寻找所有可能切分
if len(tkslist) < 2:  # 只有一种切分方式
    res.append(tk)     # 直接使用原词
    continue

stk = self.sortTks_(tkslist)[1][0]  # 选择评分第二的方案
```
短英文单词不被切分，英文短语简单空格切分。

```python
 if re.match(r"[a-z\.-]+$", tk):
    for t in stk:
        if len(t) < 3:
            stk = tk
            break
    else:
        stk = " ".join(stk)
```

# 下期预告
在 RAGFlow 的 `Naive Parser` 中，语义切块（Chunking）不仅针对文本文件，还针对多种不同类型的文档进行了适配与优化处理。源码中通过正则匹配文件扩展名，选择对应的解析与切块逻辑：
```python
if re.search(r"\.docx$", filename, re.IGNORECASE):
		...
elif re.search(r"\.pdf$", filename, re.IGNORECASE):
		...
elif re.search(r"\.(csv|xlsx?)$", filename, re.IGNORECASE):
		...
elif re.search(r"\.(txt|py|js|java|c|cpp|h|php|go|ts|sh|cs|kt|sql)$", filename, re.IGNORECASE):
		...
elif re.search(r"\.(md|markdown)$", filename, re.IGNORECASE):
		...
elif re.search(r"\.(htm|html)$", filename, re.IGNORECASE):
		...
elif re.search(r"\.(json|jsonl|ldjson)$", filename, re.IGNORECASE):
		...
elif re.search(r"\.doc$", filename, re.IGNORECASE):
		...
else:
	  raise NotImplementedError(
	      "file type not supported yet(pdf, xlsx, doc, docx, txt supported)")
```
在下一期中，我们将深入剖析 Naive Parser 下 .docx 文件的语义切块方案。