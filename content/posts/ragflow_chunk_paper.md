---
title: "【解密源码】 RAGFlow 切分最佳实践- paper 篇"
date: 2025-10-31T10:43:10+08:00
draft: false
tags: ["源码","技术","RAG"]
categories: ["RAGFlow"]
---

# 引言
论文类文档（paper）是 RAG 应用中最具挑战性的解析类型之一。  

与普通 pdf 或 ppt 不同的是，paper 通常包含复杂的版面结构：标题层级、摘要、公式、表格、图片、参考文献等，且跨页、双栏、脚注等情况极为常见。

在 RAGFlow 的源码中，paper 切分方案继承自 PdfParser 的核心逻辑，并针对学术论文场景进行了增强。本文将带你深入剖析其源码实现，了解 RAGFlow 如何实现从原始论文 pdf 到可用知识块（chunk）的高精度语义切分。

>*Tips：想更系统地理解切分逻辑，可先阅读《naive parser 语义切块（PDF 篇）》。*

# 省流版
RAGFlow 在解析论文类文档时，与处理普通 pdf 文档的思路一致，采用 **DeepDOC 布局识别 + 多层文本语义切分** 的组合方案，实现从版面识别 → 元信息抽取（标题/作者/摘要）→ 正文层级化切分 → Chunk 分词的完整流程。

**设计亮点** 
- **语义层级切分**：自动检测标题编号规则（如“第 X 节”、“1.2.3”）并按层级生成逻辑块；
- **结构化抽取**：精准提取标题、作者、摘要、章节与表格信息，形成标准结构化数据；
- **自适应切块**：基于章节层级动态合并文本，兼顾语义完整性与上下文连续性。  

# 手撕版
## 1. 布局识别器
与 naive parser 下 pdf 文档的处理一样，分为 DeepDOC 和 Plain Text 两种布局识别器。
```python
if parser_config.get("layout_recognize", "DeepDOC") == "Plain Text":
        pdf_parser = PlainParser()
    else:
        pdf_parser = Pdf()
```

Plain Text 布局识别器实现请参考《naive parser 语义切块（pdf 篇）》 下【Plain Text 布局识别器】模块。

## 2. DeepDOC 布局识别器
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

### 2.1 Pdf 类
Pdf 类作为入口点，调用 PdfParser 中提供的功能实现整个复杂的文档处理流程，并记录了各阶段耗时，解析进度等信息，这点与 naive parser 下的 Pdf 类职能一致，**但具体实现内容存在差异**。

#### 2.1.1 _line_tag
这里需要先提到 PdfParser 中另外一个功能函数 _line_tag，它主要为文本框（bx）生成一个位置信息标签字符串，用于标识该文本框在所在 pdf 页中的具体位置。
```python
def _line_tag(self, bx, ZM):
    pn = [bx["page_number"]]
    top = bx["top"] - self.page_cum_height[pn[0] - 1]
    bott = bx["bottom"] - self.page_cum_height[pn[0] - 1]
    page_images_cnt = len(self.page_images)
    if pn[-1] - 1 >= page_images_cnt:
        return ""
    # 检测文本框是否跨页
    while bott * ZM > self.page_images[pn[-1] - 1].size[1]:
        bott -= self.page_images[pn[-1] - 1].size[1] / ZM
        pn.append(pn[-1] + 1)
        if pn[-1] - 1 >= page_images_cnt:
            return ""

    return "@@{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}##".format("-".join([str(p) for p in pn]), bx["x0"], bx["x1"], top, bott)
```
对于复杂场景跨页文本框举例：
```python
# pdf 基础信息
self.page_images = [Page1, Page2, Page3]  # 共三页
self.page_cum_height = [0, 1200, 2400]    # 每页累计高度（从0开始）

# 输入 bx 文本框位置信息，bx 跨页
bx = {
    "page_number": 2,
    "x0": 100.0,
    "x1": 300.0,
    "top": 2300.0, # 文本框顶部针对整个 pdf 垂直坐标
    "bottom": 2550.0  # 文本框底部针对整个 pdf 垂直坐标
}

# 输出
@@2-3	100.0	300.0	1100.0	150.0##
# 2-3 文本框所在页码
# 100.0	300.0 文本框左右水平坐标
# 1100.0 文本框顶部相较于 page2 顶部垂直坐标
# 150 文本框底部相较于 page3 顶部垂直坐标
```
#### 2.1.2 Pdf 主流程
1）调用 PdfParser 中提供的功能进行内容提取和排序。
```python
 # 静态的 pdf 页面转换为可结构化数据 
 self.__images__(
    filename if not binary else binary,
    zoomin,
    from_page,
    to_page,
    callback
)
# 对文档每页的布局分析，和坐标重建
self._layouts_rec(zoomin)
# 表格数据内容提取
self._table_transformer_job(zoomin)
# 基于规则合并文本
self._text_merge()
# 提取表格，图片内容输出
tbls = self._extract_table_figure(True, zoomin, True, True)
column_width = np.median([b["x1"] - b["x0"] for b in self.boxes])
# 简单文本排序
self._concat_downward()
# 检测并过滤非正文页面
self._filter_forpages()
```

2）文本前置处理，水平排序和处理文本空格。
```python
if column_width < self.page_images[0].size[0] / zoomin / 2:
    # 水平排序
    self.boxes = self.sort_X_by_page(self.boxes, column_width / 2)
for b in self.boxes:
    b["text"] = re.sub(r"([\t 　]|\u3000){2,}", " ", b["text"].strip())
```

3）如果启示 page 不是首页，默认前面的标题、作者、摘要等已经被提取过，直接返回正文 section 内容： (文本内容 + 位置标签, 布局类型)。
```python
if from_page > 0:
    return {
        "title": "",
        "authors": "",
        "abstract": "",
        "sections": [(b["text"] + self._line_tag(b, zoomin), b.get("layoutno", "")) for b in self.boxes if
                        re.match(r"(text|title)", b.get("layoutno", "text"))],
        "tables": tbls
    }
```

4）从首页开始，提取标题，作者信息。这里默认这些信息位于文档开头，所以只在前 32 个文本框中进行提取。
```python
title = ""
authors = []
i = 0
while i < min(32, len(self.boxes)-1):
    b = self.boxes[i]
    i += 1
    if b.get("layoutno", "").find("title") >= 0:
        title = b["text"]
        if _begin(title):
            title = ""
            break
        for j in range(3):
            if _begin(self.boxes[i + j]["text"]):
                break
            authors.append(self.boxes[i + j]["text"])
            break
        break
```

5）提取摘要，方案与提取标题基本相同。遍历前 32 个文本框，查找“Abstract”或“摘要”段落，若该段落足够长（字数或单词数超过阈值），就认为是完整摘要。
```python
abstr = ""
i = 0
while i + 1 < min(32, len(self.boxes)):
    b = self.boxes[i]
    i += 1
    txt = b["text"].lower().strip()
    if re.match("(abstract|摘要)", txt):
        if len(txt.split()) > 32 or len(txt) > 64:
            abstr = txt + self._line_tag(b, zoomin)
            break
        txt = self.boxes[i]["text"].lower().strip()
        if len(txt.split()) > 32 or len(txt) > 64:
            abstr = txt + self._line_tag(self.boxes[i], zoomin)
        i += 1
        break
```

6）返回最终结构化结果
```python
return {
    "title": title,
    "authors": " ".join(authors),
    "abstract": abstr,
    "sections": [(b["text"] + self._line_tag(b, zoomin), b.get("layoutno", "")) for b in self.boxes[i:] if
                    re.match(r"(text|title)", b.get("layoutno", "text"))],
    "tables": tbls
}
# 以下是结果示例
{
  "title": "A Novel PDF Parsing Framework",
  "authors": "Alice Bob",
  "abstract": "This paper proposes ...@@1	50.0	520.0	300.0	900.0##",
  "sections": [
      ("Introduction@@2	100.0	400.0	80.0	220.0##", "title"),
      ("In this paper, we propose ...@@2-3	120.0	480.0	200.0	150.0##", "text"),
      ...
  ],
  "tables": [...]
}
```

## 3. 数据后处理
从 Pdf 类实例化方法获取到结构化数据后，进行一系列的数据后处理，最终生成 chunk 数据。

1）标题，作者，表格分词。分词器实现可参考《分词器原理》。
```python
doc = {"docnm_kwd": filename, "authors_tks": rag_tokenizer.tokenize(paper["authors"]),
        "title_tks": rag_tokenizer.tokenize(paper["title"] if paper["title"] else filename)}
doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])
doc["authors_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["authors_tks"])

res = tokenize_table(paper["tables"], doc, eng)
```

2）摘要内容处理。将内容类型标记为 “abstract” 等优化召回，同时处理摘要内容中的图片，位置，和对内容信息进行分词。
```python
if paper["abstract"]:
    d = copy.deepcopy(doc)
    txt = pdf_parser.remove_tag(paper["abstract"])
    d["important_kwd"] = ["abstract", "总结", "概括", "summary", "summarize"]
    d["important_tks"] = " ".join(d["important_kwd"])
    d["image"], poss = pdf_parser.crop(
        paper["abstract"], need_position=True)
    add_positions(d, poss)
    tokenize(d, txt, eng)
    res.append(d)
```

3）正文层级识别。先检测文档标题使用的编号样式类别，后结合编号样式确定标题的层级深度。
```python
sorted_sections = paper["sections"]
# set pivot using the most frequent type of title,
# then merge between 2 pivot
bull = bullets_category([txt for txt, _ in sorted_sections])
most_level, levels = title_frequency(bull, sorted_sections)
```

`bullets_category` 主要是检测文档标题使用的编号样式类别，编号样式类别可能如下：
```python
[[
    r"第[零一二三四五六七八九十百0-9]+(分?编|部分)",
    r"第[零一二三四五六七八九十百0-9]+章",
    r"第[零一二三四五六七八九十百0-9]+节",
    r"第[零一二三四五六七八九十百0-9]+条",
    r"[\(（][零一二三四五六七八九十百]+[\)）]",
], [
    r"第[0-9]+章",
    r"第[0-9]+节",
    r"[0-9]{,2}[\. 、]",
    r"[0-9]{,2}\.[0-9]{,2}[^a-zA-Z/%~-]",
    r"[0-9]{,2}\.[0-9]{,2}\.[0-9]{,2}",
    r"[0-9]{,2}\.[0-9]{,2}\.[0-9]{,2}\.[0-9]{,2}",
], [
    r"第[零一二三四五六七八九十百0-9]+章",
    r"第[零一二三四五六七八九十百0-9]+节",
    r"[零一二三四五六七八九十百]+[ 、]",
    r"[\(（][零一二三四五六七八九十百]+[\)）]",
    r"[\(（][0-9]{,2}[\)）]",
], [
    r"PART (ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)",
    r"Chapter (I+V?|VI*|XI|IX|X)",
    r"Section [0-9]+",
    r"Article [0-9]+"
], [
    r"^#[^#]",
    r"^##[^#]",
    r"^###.*",
    r"^####.*",
    r"^#####.*",
    r"^######.*",
]
]
```
检测出编号样式后，通过 `title_frequency` 确定整个文档最常出现的章节标题层级 `most_level`，作为切分基准；同时也获取整个文档所有标题等级列表 `levels`。

4）按照 `most_level` 切分基准对内容进行切分，`<= most_level` 的内容会进行统一标记并**与同级内容和上一级内容**进行合并作为一个 chunk。
```python
sec_ids = []
sid = 0
for i, lvl in enumerate(levels):
    if lvl <= most_level and i > 0 and lvl != levels[i - 1]:
        sid += 1
    sec_ids.append(sid)
    logging.debug("{} {} {} {}".format(lvl, sorted_sections[i][0], most_level, sid))

chunks = []
last_sid = -2
for (txt, _), sec_id in zip(sorted_sections, sec_ids):
    if sec_id == last_sid:
        if chunks:
            chunks[-1] += "\n" + txt
            continue
    chunks.append(txt)
    last_sid = sec_id
```
5）使用分词器对 chunk 进行分词输出最终结果
```python
res.extend(tokenize_chunks(chunks, doc, eng, pdf_parser))
```
