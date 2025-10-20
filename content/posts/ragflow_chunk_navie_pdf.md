---
title: "【解密源码】 RAGFlow 切分最佳实践- naive Parser 语义切块（pdf 篇）"
date: 2025-10-18T10:39:10+08:00
draft: true
tags: ["源码","技术",RAG]
categories: ["RAGFlow"]
---
# 引言

在上一期《naive Parser 语义切块（docx 篇）》中

本期我们将从通用机制深入到 **具体文件类型的实现逻辑** —— 聚焦 `.pdf` 文件在 navie parser 下的语义切块原理。  `.docx` 文档在结构上拥有丰富的层次信息（段落、样式、标题、表格等），这使得其语义切块策略必须兼顾 **格式解析与语义连贯性**。

# 省流版

# 手撕版

## 1. 布局识别器
两种布局识别器分别为 `DeepDOC` 和 `Plain Text`，分别对应不同的 pdf 解析方案。如果不属于这两种布局，保底使用视觉模型进行 pdf 解析。

``` python
layout_recognizer = parser_config.get("layout_recognize", "DeepDOC")
    if isinstance(layout_recognizer, bool):
        layout_recognizer = "DeepDOC" if layout_recognizer else "Plain Text"
if layout_recognizer == "DeepDOC":
    pdf_parser = Pdf()
else:
    if layout_recognizer == "Plain Text":
        pdf_parser = PlainParser()
    else:
        pdf_parser = VisionParser(vision_model=vision_model, **kwargs)
```

## 2. DeepDOC 布局识别器
解析器初始化
``` python
pdf_parser = Pdf()
```

Pdf 继承基类 PdfParser

```python
class Pdf(PdfParser):
```

### PdfParser 基类
PdfParser 实例化实现了以下功能：

**\_\_images__**：

**_layouts_rec**：

**_table_transformer_job**：

**_text_merge**：

**_concat_downward**：

**_filter_forpages**：

**_extract_table_figure**：

**__filterout_scraps**：
``` python
def __call__(self, fnm, need_image=True, zoomin=3, return_html=False):
    self.__images__(fnm, zoomin)
    self._layouts_rec(zoomin)
    self._table_transformer_job(zoomin)
    self._text_merge()
    self._concat_downward()
    self._filter_forpages()
    tbls = self._extract_table_figure(need_image, zoomin, return_html, False)
    return self.__filterout_scraps(deepcopy(self.boxes), zoomin), tbls
```
#### \_\_images__
函数中采用了 pdfplumber 和 pypdf 两个库来识别 pdf 文档。

**pdfplumber**： 主要负责文档内容提取，内容图像生成，页面布局分析。

```python
with pdfplumber.open(fnm) if isinstance(fnm, str) else pdfplumber.open(BytesIO(fnm)) as pdf:
    # 将内容页转换成图像，兼容文档型和扫描型 pdf，统一用 OCR 进行处理
    self.page_images = [p.to_image(resolution=72 * zoomin, antialias=True).annotated for i, p in enumerate(self.pdf.pages[page_from:page_to])]
    # 提取文档内容
    self.page_chars = [[c for c in page.dedupe_chars().chars if self._has_color(c)] for page in self.pdf.pages[page_from:page_to]]
```

**pypdf**：主要负责文档大纲提取，使用 DFS 提取大纲标题和层级。

```python
with pdf2_read(fnm if isinstance(fnm, str) else BytesIO(fnm)) as pdf:
    outlines = self.pdf.outline
    dfs(outlines, 0)
```
文档语言检测
```python
self.is_english = [
    re.search(r"[a-zA-Z0-9,/¸;:'\[\]\(\)!@#$%^&*\"?<>._-]{30,}", "".join(random.choices([c["text"] for c in self.page_chars[i]], k=min(100, len(self.page_chars[i])))))
    for i in range(len(self.page_chars))
]
```
使用 OCR 对 pdf 文档进行识别，这里对 OCR 流程做简单介绍：
1. 文本框检测，检测文档中文本框的位置
```python
bxs = self.ocr.detect(np.array(img), device_id)
```
2. 格式还原，坐标除以缩放因子（ZM）还原到原始 pdf 尺寸后排序文本框。
``` python
bxs = [(line[0], line[1][0]) for line in bxs]
bxs = Recognizer.sort_Y_firstly(
    [
        {
            "x0": b[0][0] / ZM,      # 左边界（还原到原始尺寸）
            "x1": b[1][0] / ZM,      # 右边界
            "top": b[0][1] / ZM,     # 上边界
            "text": "",              # 初始化文本为空
            "txt": t,                # OCR初步识别的文本
            "bottom": b[-1][1] / ZM, # 下边界
            "chars": [],             # 存储匹配的PDF字符
            "page_number": pagenum   # 页码
        }
        for b, t in bxs
        if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]  # 过滤无效框
    ],
    self.mean_height[pagenum - 1] / 3,  # 排序阈值
)
```
3. 字符与文本框匹配
```python
for c in chars:
    ii = Recognizer.find_overlapped(c, bxs)  # 找到字符所属的文本框
    if ii is None:
        self.lefted_chars.append(c)  # 未匹配的字符
        continue
    
    # 高度一致性检查
    ch = c["bottom"] - c["top"]      # 字符高度
    bh = bxs[ii]["bottom"] - bxs[ii]["top"]  # 文本框高度
    if abs(ch - bh) / max(ch, bh) >= 0.7 and c["text"] != " ":
        self.lefted_chars.append(c)   # 高度差异过大，排除
        continue
        
    bxs[ii]["chars"].append(c)       # 将字符加入对应文本框
```
4. 文档重建，使用 pdf 提取文字结合 OCR 检测文本框结构重建文档。
```python
for b in bxs:
    if not b["chars"]:
        del b["chars"]
        continue
        
    # 计算平均字符高度用于排序
    m_ht = np.mean([c["height"] for c in b["chars"]])
    
    # 按位置排序字符
    for c in Recognizer.sort_Y_firstly(b["chars"], m_ht):
        if c["text"] == " " and b["text"]:
            # 智能空格插入：只在英文/数字后加空格
            if re.match(r"[0-9a-zA-Zа-яА-Я,.?;:!%%]", b["text"][-1]):
                b["text"] += " "
        else:
            b["text"] += c["text"]  # 拼接字符文本
            
    del b["chars"]  # 清理临时数据
```
5. OCR 提取文字，pdf 未提取到文字情况下，使用 OCR 方案进行文字识别。
```python
texts = self.ocr.recognize_batch([b["box_image"] for b in boxes_to_reg], device_id)
for i in range(len(boxes_to_reg)):
    boxes_to_reg[i]["text"] = texts[i]  # 更新识别结果
    del boxes_to_reg[i]["box_image"]
```
6. 整理重建结果
```python
bxs = [b for b in bxs if b["text"]]  # 过滤空文本框

# 更新字符高度统计（用于后续页面处理）
if self.mean_height[pagenum - 1] == 0:
    self.mean_height[pagenum - 1] = np.median([b["bottom"] - b["top"] for b in bxs])

self.boxes.append(bxs)  # 存储最终结果
```
#### _layouts_rec
结合 pdfplumber 获取的内容业图像和 OCR 识别的文本框信息，进行文档每页的布局分析，和坐标重建。

1. 布局分析，布局相较于文档框提供更高层次的文档语义信息。文档框中只包含位置坐标，所在页码，文本内容等信息，布局信息中包含文本类型信息，如是标题，正文，表格，页眉等，为后续针对性分析提供必要信息。
```python
self.boxes, self.page_layout = self.layouter(self.page_images, self.boxes, ZM, drop=drop)
```
2. 坐标重建，文本框中的坐标信息是对于所在页的，坐标都是独立从 0 开始，重建坐标是基于整个文档的全局坐标。
```python
for i in range(len(self.boxes)):
    self.boxes[i]["top"] += self.page_cum_height[self.boxes[i]["page_number"] - 1]
    self.boxes[i]["bottom"] += self.page_cum_height[self.boxes[i]["page_number"] - 1]
```