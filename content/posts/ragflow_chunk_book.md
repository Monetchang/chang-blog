---
title: "【解密源码】 RAGFlow 切分最佳实践- book 篇"
date: 2025-11-02T11:33:10+08:00
draft: false
tags: ["源码","技术","RAG"]
categories: ["RAGFlow"]
---

# 引言
书籍文件往往篇幅巨大、结构复杂，不同章节、目录、致谢等混杂在同一文档中。RAGFlow 在处理 “book” 类文档时，整合了多种解析器（docx、pdf、txt、html、doc），通过自动识别版面结构、过滤非正文、并结合视觉模型生成图片摘要，实现对长文档的精准切分与高质量抽取。本篇将带你逐步拆解源码，了解 RAGFlow 如何优雅地解析一本“书”。

# 省流版

**核心逻辑**

book 模式 面向长篇书籍类文档（docx/pdf/txt/html/doc），通过对不同文件类型自动选择最优解析策略，融合文本语义分块与视觉摘要，实现高质量内容抽取。其目标是：在保持上下文完整性的同时最大限度地过滤无效内容（如目录、致谢页等），为后续知识检索提供干净、结构化的输入。

**设计亮点**
- 语义级正文过滤：通过正则与上下文规则剔除目录、致谢等非正文部分；
- 文本块连续性优化：pdf 文档通过 _naive_vertical_merge 与 _merge_with_same_bullet 修复 OCR 切分断裂、项目符号分裂等问题。

# 手撕版
书籍的解析支持文件格式为 docx|pdf|txt|html|doc 五种格式。其中官方建议由于一本书篇幅较长，并非所有部分都有用，如果是 PDF 格式，请为每本书设置页码范围，以消除负面影响并节省计算时间。
```python
if re.search(r"\.docx$", filename, re.IGNORECASE):
    ...
elif re.search(r"\.pdf$", filename, re.IGNORECASE):
    ...
elif re.search(r"\.txt$", filename, re.IGNORECASE):
    ...
elif re.search(r"\.(htm|html)$", filename, re.IGNORECASE):
    ...
elif re.search(r"\.doc$", filename, re.IGNORECASE):
    ...
```

## docx
### 1. 解析器初始化
docx 的解析器是直接引用的 naive 模式下的 docx 解析器，主要对 docx 文档中的段落和表格分别进行处理。详细的 docx 解析器技术拆解可参考《naive parser 语义切块（docx 篇）》。
```python
doc_parser = naive.Docx()
```
段落和表格处理后的最终输出格式如下：
```python
sections, tbls = doc_parser(
            filename, binary=binary, from_page=from_page, to_page=to_page)
'''
# sections
[
    (清洗后的文本, 合并后的图片对象, 样式名),
    ("这是段落文本", PILImage对象, "Normal"),
    ("这是图注", PILImage对象, "Caption"),
    ...
]
# tbls
[
    ((None, "<table>...</table>"), ""),
    ((None, "<table>...</table>"), ""),
    ...
]
'''
```
### 2. 目录表格 过滤
过滤 docx 文档中的目录表格。
```python
tbls = _remove_docx_toc_tables(tbls)
```

通过关键字正则判断是否是目录表格，满足其中一条则认为是目录表格。
```python
_TOC_TABLE_KEYWORDS = re.compile(r"(table of contents|目录|目次)", re.IGNORECASE)
_TOC_DOT_PATTERN = re.compile(r"[\.|·]{3,}\s*\d+")
```

### 3. 非正文内容过滤
过滤解析后文本中的非正文内容，例如：目录，致谢等。
```python
remove_contents_table(sections, eng=is_english(
            random_choices([t for t, _ in sections], k=200)))
```

通过关键字正则判断是否是非正文内容。
```python
re.match(r"(contents|目录|目次|table of contents|致谢|acknowledge)$",
        re.sub(r"( | |\u3000)+", "", get(i).split("@@")[0], flags=re.IGNORECASE))
```

### 4. 使用视觉模型识别并总结图片摘要
使用 VLM 对文档中的图片进行摘要总结，并以规定格式输出。详细的 vision_figure_parser_docx_wrapper 技术拆解可参考《naive parser 语义切块（docx 篇）》。
```python
tbls=vision_figure_parser_docx_wrapper(sections=sections,tbls=tbls,callback=callback,**kwargs)
```
将包含图片信息的对象数组转换成 figures_data，如以下格式：

```python
(
		(figure_data[1], [figure_data[0]]), # 原始图片信息，图片描述信息
		[(0, 0, 0, 0, 0)], # 位置信息
)
```

### 5. 移除正文中的图片内容
```python
sections=[(item[0],item[1] if item[1] is not None else "") for item in sections if not isinstance(item[1], Image.Image)]
```

docx 处理流程结束后输出，sections（正文文本内容）和 tbls（表格内容，图片内容）。

## pdf
### 1. 布局识别器
与 naive parser 下 pdf 文档的处理一样，分为 DeepDOC 和 Plain Text 两种布局识别器。
```python
if parser_config.get("layout_recognize", "DeepDOC") == "Plain Text":
        pdf_parser = PlainParser()
    else:
        pdf_parser = Pdf()
```

Plain Text 布局识别器实现请参考《naive parser 语义切块（pdf 篇）》 下【Plain Text 布局识别器】模块。

### 2. DeepDOC 布局识别器
与《naive parser 语义切块（pdf 篇）》中相同，Pdf 继承基类 PdfParser。
```python
pdf_parser = Pdf()

class Pdf(PdfParser)
```
PdfParser 基类的核心功能在《naive parser 语义切块（PDF 篇）》中已有详细说明，这里作简要概述：

- **\_\_images__**：负责将 PDF 页面数字化，生成可结构化的页面图像数据，为后续的布局分析与表格提取奠定基础。

- **_layouts_rec**：基于页面图像与 OCR 文本框信息，执行每页的版面分析与坐标重建。

- **_table_transformer_job**：针对识别出的表格区域，提取和解析表格内容，实现表格结构化处理。

- **_text_merge**：通过规则合并相邻文本块，解决 OCR 输出中文本碎片化的问题。

- **_concat_downward**：按照阅读顺序（Y 坐标从上到下，X 坐标从左到右）对文本框进行排序，恢复自然的文本流。

- **_filter_forpages**：检测并过滤非正文页面，如目录页、致谢页等，以提升后续文本分析质量。

- **_extract_table_figure**：负责表格与图像的提取与输出，支持跨页表格合并及图像截取。

- **__filterout_scraps**：对碎片化文本进行二次清理与组装，进一步优化 OCR 文本的完整性。

#### Pdf 类
Pdf 类作为入口点，调用 PdfParser 中提供的功能实现整个复杂的文档处理流程，并记录了各阶段耗时，解析进度等信息，这点与 naive parser 下的 Pdf 类职能一致，**但具体实现内容存在差异**。

```python
class Pdf(PdfParser):
    def __call__(self, filename, binary=None, from_page=0,
                 to_page=100000, zoomin=3, callback=None):
         # 静态的 pdf 页面转换为可结构化数据 
        self.__images__(
            filename if not binary else binary,
            zoomin,
            from_page,
            to_page,
            callback)
        # 对文档每页的布局分析，和坐标重建
        self._layouts_rec(zoomin)
        # 表格数据内容提取
        self._table_transformer_job(zoomin)
        # 基于规则合并文本
        self._text_merge()
        # 提取表格，图片内容输出
        tbls = self._extract_table_figure(True, zoomin, True, True)
        self._naive_vertical_merge()
        # 检测并过滤非正文页面
        self._filter_forpages()
        self._merge_with_same_bullet()
        callback(0.8, "Text extraction ({:.2f}s)".format(timer() - start))

        return [(b["text"] + self._line_tag(b, zoomin), b.get("layoutno", ""))
                for b in self.boxes], tbls
```
##### _naive_vertical_merge
其中 _naive_vertical_merge 主要目的是将同一列中垂直方向相邻的文本框进行合并。

先过滤无效文本:

**1. 跨页数字符号过滤：移除跨页的页码、编号等无关文本**
```python
if b["page_number"] < b_["page_number"] and re.match(r"[0-9  •一—-]+$", b["text"]):
    bxs.pop(i)
    continue
```
**2. 空文本过滤**
```python
if not b["text"].strip():
    bxs.pop(i)
    continue
```

再通过文本框布局识别，确认文本布局是否符合合并规则：

**1. 布局一致性检查：确保合并的文本框属于同一布局区域**
```python
if not b["text"].strip() or b.get("layoutno") != b_.get("layoutno"):
    i += 1
    continue
```

**2. 垂直距离阈值检查：防止合并距离过远的文本框**
```python
if b_["top"] - b["bottom"] > mh * 1.5:
    i += 1
    continue
```

**3. 水平重叠度检查：确保文本框在水平方向有足够重叠**
```python
overlap = max(0, min(b["x1"], b_["x1"]) - max(b["x0"], b_["x0"]))
if overlap / max(1, min(b["x1"] - b["x0"], b_["x1"] - b_["x0"])) < 0.3:
    i += 1
    continue
```

最后通过上述筛选的文本框，对其文本语义连续性分析，决定是否需要合并。
```python
# 支持合并的特征（连接性标点）
concatting_feats = [
    b["text"].strip()[-1] in ",;:'\"，、‘“；：-",
    len(b["text"].strip()) > 1 and b["text"].strip()[-2] in ",;:'\"，‘“、；：", 
    b_["text"].strip() and b_["text"].strip()[0] in "。；？！?”）),，、：",
]

# 反对合并的特征
feats = [
    b.get("layoutno", 0) != b_.get("layoutno", 0),
    b["text"].strip()[-1] in "。？！?",
    self.is_english and b["text"].strip()[-1] in ".!?",
    b["page_number"] == b_["page_number"] and b_["top"] - b["bottom"] > self.mean_height[b["page_number"] - 1] * 1.5,
    b["page_number"] < b_["page_number"] and abs(b["x0"] - b_["x0"]) > self.mean_width[b["page_number"] - 1] * 4,
]

# 强制分离特征
detach_feats = [b["x1"] < b_["x0"], b["x0"] > b_["x1"]]

if (any(feats) and not any(concatting_feats)) or any(detach_feats):
    i += 1
    continue
```

##### _merge_with_same_bullet
将具有相同项目符号的连续文本框合并为一个文本块，保持项目符号列表的结构。

pdf 中可能存在以下项目列表， 一个完整的项目符号列表可能被识别为多个独立的文本框，_merge_with_same_bullet 主要是将同一个项目列表内容进行合并。
```python
• 项目一：产品介绍
• 项目二：技术规格
• 项目三：价格信息
```

### 3. 使用视觉模型识别并总结图片摘要
与 docx 处理一致，需要使用 VLM 对文档中的图片进行摘要总结，并以规定格式输出。
```python
tbls=vision_figure_parser_pdf_wrapper(tbls=tbls,callback=callback,**kwargs)
```

## TXT
与 naive 模式下获取 txt 文档方案一致，使用 `get_text`。如果传入的二进制内容，则使用从 rag.nlp 引入的方式自动推断出正确的编码，进行解码；否则直接从文件路径读取文本进行拼接返回。
```python
txt = get_text(filename, binary)
```
与 docx 文档处理方案一致，获取文本后过滤文本中的非正文内容，例如：目录，致谢等。
```python
remove_contents_table(sections, eng=is_english(
            random_choices([t for t, _ in sections], k=200)))
```

## HTML
与 naive 模式下处理 html 文档方案一致，使用 `HtmlParser` 解析器进行文档解析。详细技术实现可参考《naive parser 语义切块（html & json & doc 篇）》
```python
sections = HtmlParser()(filename, binary)
```
与 docx 文档处理方案一致，获取文本后过滤文本中的非正文内容，例如：目录，致谢等。
```python
remove_contents_table(sections, eng=is_english(
            random_choices([t for t, _ in sections], k=200)))
```

## doc
与 naive 模式下处理 doc 文档方案一致，详细技术实现可参考《naive parser 语义切块（html & json & doc 篇）》
```python
doc_parsed = parser.from_buffer(binary)
```
与 docx 文档处理方案一致，获取文本后过滤文本中的非正文内容，例如：目录，致谢等。
```python
remove_contents_table(sections, eng=is_english(
            random_choices([t for t, _ in sections], k=200)))
```