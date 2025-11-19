---
title: "【解密源码】 轻量 GrapghRAG - LightRAG 工程实践"
date: 2025-11-02T11:33:10+08:00
draft: false
tags: ["源码","技术","RAG"]
categories: ["LightRAG"]
---

# 引言

# 省流版

# 手撕版
## 1. 上传文件
## 1.1 上传文件前置检查
### 防止路径穿越攻击
### 文件名类型检查
### 重复上传检查（上传中，已上传）


## 2. 解析文件
### 2.1 读取文件内容（详细的异常处理，记录异常信息）
### 2.2 解析文件内容（分类处理）
### 2.3 存储文件数据
### 2.4 分词（默认使用 Tiktoken tokenizer，model：gpt-4o-mini）
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
### 2.5 存储文件数据（默认向量数据库 NanoVectorDBStorage ）


