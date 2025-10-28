---
title: "DeepSeek 与 PaddleOCR-VL 常见场景下能力横评"
date: 2025-10-28T10:39:10+08:00
draft: false
tags: ["技术","VLM","OCR"]
categories: ["VLM"]
---

# 引言
OCR 不再只是识字。随着多模态大模型的发展，新一代的 OCR 模型（如 DeepSeek-OCR）已经具备了对图文内容进行理解、总结、翻译的能力。而传统 OCR 模型（如 PaddleOCR-VL）在识别精度与稳定性上依然有着无可替代的优势。本文将通过多个真实场景，横向评测这两种模型的差异与适用性。

# 模型简介
## PaddleOCR-VL

来自百度 PaddleOCR 团队，属于传统 OCR 扩展，支持文档版式分析（layout）、表格结构提取等

### 安装

### 功能


## DeepSeek-OCR

属于视觉语言模型（VLM）范畴，不仅识别，还能基于 prompt 输出总结、翻译、问答等

### 安装

### 功能
```python
# PROMPT = '<image>\nFree OCR'
PROMPT = '<image>\nParse the figure.'
# PROMPT='<image>\n<|grounding|>OCR this image.'
# PROMPT='<image>\nDescribe this image in detail.'
```

原理上更多对比可参考《》

# 测评场景

## 常见 PDF 文档
![图片示例](/images/e_trans.jpg)

