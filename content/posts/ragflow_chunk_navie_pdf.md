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
PdfParser() 实例化实现了以下功能：

**\_\_images__**：

**_layouts_rec**：

**_table_transformer_job**：

**_text_merge**：基于规则合并文本，解决 OCR 识别中文本碎片化问题。

**_concat_downward**：按照阅读习惯将文本框按照 Y 坐标由上至下，X 坐标从左到右排序。

**_filter_forpages**：检测并过滤掉文档中的非正文内容页面，如目录页、致谢页等，提高后续处理的文本质量。

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

#### _table_transformer_job
表格处理系统，根据布局分析得到的表格类型，对其表格数据内容进行提取处理。
1. 根据布局分析中的表格，记录每页表格数量，裁剪出表格图像并记录位置坐标。
```python
for p, tbls in enumerate(self.page_layout):
    tbls = [f for f in tbls if f["type"] == "table"]
    tbcnt.append(len(tbls))
    
    for tb in tbls: 
        # 添加边距并裁剪表格图像
        left, top, right, bott = tb["x0"]-MARGIN, tb["top"]-MARGIN, tb["x1"]+MARGIN, tb["bottom"]+MARGIN
        left *= ZM; top *= ZM; right *= ZM; bott *= ZM  # 缩放回原始分辨率
        
        pos.append((left, top))  # 记录表格在页面中的位置
        imgs.append(self.page_images[p].crop((left, top, right, bott)))  # 裁剪表格图像
```
2. **核心步骤：表结构识别**

`self.tbl_det(imgs)` 实际调用 `TableStructureRecognizer.__call__()` 方法,`TableStructureRecognizer.__call__()` 方法中兼容 onnx 和 ascend 两种推理后端.
``` python
recos = self.tbl_det(imgs)
```
**TableStructureRecognizer.__call__()**

通过模型推理返回检测文本框，并对检测出的文本框进行置信度计算过滤和坐标还原。
```python
 if table_structure_recognizer_type == "onnx":
    logging.debug("Using Onnx table structure recognizer", flush=True)
    tbls = super().__call__(images, thr)
else:  # ascend
    logging.debug("Using Ascend table structure recognizer", flush=True)
    tbls = self._run_ascend_tsr(images, thr)
```
对文本框进行坐标对齐，消除检测的偏移误差。
```python
# 对齐行和列的边界
left = [b["x0"] for b in lts if b["label"].find("row") > 0 or b["label"].find("header") > 0]
right = [b["x1"] for b in lts if b["label"].find("row") > 0 or b["label"].find("header") > 0]
left = np.mean(left) if len(left) > 4 else np.min(left)
right = np.mean(right) if len(right) > 4 else np.max(right)

# 统一行的左右边界
for b in lts:
    if b["label"].find("row") > 0 or b["label"].find("header") > 0:
        if b["x0"] > left: b["x0"] = left
        if b["x1"] < right: b["x1"] = right
```
最终返回结构预测：
```python
# recos 的结构示例:
recos = [
    [  # 第一个表格的结构组件
        {"label": "table row", "x0": 10, "x1": 200, "top": 5, "bottom": 25, "score": 0.95},
        {"label": "table column", "x0": 10, "x1": 100, "top": 5, "bottom": 150, "score": 0.92},
        {"label": "table column header", "x0": 10, "x1": 100, "top": 5, "bottom": 25, "score": 0.88},
        # ... 更多行、列、表头、合并单元格等组件
    ],
    [  # 第二个表格的结构组件
        # ...
    ]
]
```

3. 坐标系统转换和整合
将表格局部坐标转换回基于 pdf 文档的全局坐标，最终形成的 self.tb_cpns 列表结构与 recos 类似，self.tb_cpns 中 x0，x1，top，bottom 坐标值基于整个 pdf 文档。
```python
for i in range(len(tbcnt) - 1):  # for page
    pg = []
    for j, tb_items in enumerate(recos[tbcnt[i] : tbcnt[i + 1]]):  # for table
        poss = pos[tbcnt[i] : tbcnt[i + 1]]
        for it in tb_items:  # for table components
            it["x0"] = it["x0"] + poss[j][0]
            it["x1"] = it["x1"] + poss[j][0]
            it["top"] = it["top"] + poss[j][1]
            it["bottom"] = it["bottom"] + poss[j][1]
            for n in ["x0", "x1", "top", "bottom"]:
                it[n] /= ZM
            it["top"] += self.page_cum_height[i]
            it["bottom"] += self.page_cum_height[i]
            it["pn"] = i
            it["layoutno"] = j
            pg.append(it)
    self.tb_cpns.extend(pg)
```

4. 定义排序函数 gather
`sort_Y_firstly` 按 Y 坐标排序由上至下排序，若在同一行则按 X 坐标由左到右排序；`layouts_cleanup` 去除重叠和低质量的布局组件。
```python
def gather(kwd, fzy=10, ption=0.6):
    eles = Recognizer.sort_Y_firstly([r for r in self.tb_cpns if re.match(kwd, r["label"])], fzy)
    eles = Recognizer.layouts_cleanup(self.boxes, eles, 5, ption)
    return Recognizer.sort_Y_firstly(eles, 0)
```

5. 表格结构组件排序

按类型收集表格结构组件进行排序
```python
headers = gather(r".*header$")      # 收集所有表头组件
rows = gather(r".* (row|header)")   # 收集所有行和表头组件  
spans = gather(r".*spanning")       # 收集所有合并单元格组件

# 列组件的特殊处理（需要按位置排序）
clmns = sorted([r for r in self.tb_cpns if re.match(r"table column$", r["label"])], 
               key=lambda x: (x["pn"], x["layoutno"], x["x0"]))  # 按页码、表格索引、X坐标排序
clmns = Recognizer.layouts_cleanup(self.boxes, clmns, 5, 0.5)    # 布局清理
```
6.文本框与表结构关联

将 OCR 识别的文本框与排序后的表行，标题，文本通过坐标基于重叠度进行匹配，表列通过水平坐标进行匹配。最终输出
```python
for b in self.boxes:
    ii = Recognizer.find_overlapped_with_threshold(b, rows, thr=0.3)
    ii = Recognizer.find_overlapped_with_threshold(b, headers, thr=0.3)
    ii = Recognizer.find_horizontally_tightest_fit(b, clmns)
    ii = Recognizer.find_overlapped_with_threshold(b, spans, thr=0.3)
```
最终输出完整的符合文本框坐标的相应的表格内容。

#### _filter_forpages
1. 识别目录页等非正文内容页面
```python
i = 0
while i < len(self.boxes):
    # 检查文本框内容是否匹配目录页特征
    text_content = re.sub(r"( | |\u3000)+", "", self.boxes[i]["text"].lower())
    if not re.match(r"(contents|目录|目次|table of contents|致谢|acknowledge)$", text_content):
        i += 1
        continue
```
2. 识别关键非正文信息进行过滤

识别目录标题，目录项前缀等关键非正文信息，对文本框中内容进行匹配，对匹配的信息进行删除
```python
# 删除目录标题本身
self.boxes.pop(i)

# 获取目录项前缀，中文前三个字符如：第一章；英文前两个单词如：Chapter 1
eng = re.match(r"[0-9a-zA-Z :'.-]{5,}", self.boxes[i]["text"].strip())
if not eng:
    prefix = self.boxes[i]["text"].strip()[:3]
else:
    prefix = " ".join(self.boxes[i]["text"].strip().split()[:2])

# 删除前缀以及相关非正文内容
while not prefix:
    self.boxes.pop(i)
    if i >= len(self.boxes): break
    prefix = self.boxes[i]["text"].strip()[:3] if not eng else " ".join(self.boxes[i]["text"].strip().split()[:2])

self.boxes.pop(i)

if i < len(self.boxes) and prefix:
    for j in range(i, min(i + 128, len(self.boxes))):
        if not re.match(prefix, self.boxes[j]["text"]):
            continue
        for k in range(i, j):
            self.boxes.pop(i)
        break
```

3. 对“脏页”进行删除
通过“脏”字符匹配计算页面“脏度”，进行标记删除
```python
# 计算页面脏度
page_dirty = [0] * len(self.page_images)
for b in self.boxes:
    if re.search(r"(··|··|··)", b["text"]):
        page_dirty[b["page_number"] - 1] += 1
# 标记脏页
page_dirty = set([i + 1 for i, t in enumerate(page_dirty) if t > 3])
# 删除脏页
if page_dirty:  # 如果存在脏页
    i = 0
    while i < len(self.boxes):
        if self.boxes[i]["page_number"] in page_dirty:
            self.boxes.pop(i)  # 删除该页的所有文本框
            continue  # 不增加i，因为删除了当前元素
        i += 1

```

#### _extract_table_figure
提取文档图像，表格内容图像，关联相应标题