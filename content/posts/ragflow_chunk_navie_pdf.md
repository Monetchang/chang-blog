---
title: "【解密源码】 RAGFlow 切分最佳实践- naive parser 语义切块（pdf 篇）"
date: 2025-10-18T10:39:10+08:00
draft: true
tags: ["源码","技术","RAG"]
categories: ["RAGFlow"]
---
# 引言

在上一期《naive parser 语义切块（docx 篇）》中，我们深入剖析了 RAGFlow 如何处理结构丰富的 .docx 文档，从基础解析到视觉增强，展现了完整的文档理解流水线。我们看到了智能表格识别、标题链构建、多模态融合等核心技术如何协同工作，将格式化的办公文档转化为高质量的语义块。

本期我们将挑战文档处理领域的"终极 BOSS"—— pdf 格式。作为最复杂、最通用的文档格式，pdf 兼具了扫描件与数字文档的双重特性，版面布局千变万化，文字提取难度极高。RAGFlow 通过三重解析引擎的智能协作，为不同特性的 pdf 文档提供了针对性的解决方案。

# 省流版
RAGFlow 针对 pdf 文档提供了**三种不同文档解析方案**，根据文档特性和处理需求智能选择最优方案：
### DeepDOC 深度解析（主力方案

    适用场景：复杂版面、图文混排、需要精确结构提取的文档

    **解析流程**
    ```
    PDF文档
        ↓
    【结构转换层】pdfplumber & pypdf → 结构化数据
        ↓  
    【文本提取层】双引擎协同 → 原生文本 + OCR 识别
        ↓
    【布局分析层】深度学习模型 → 标题/正文/表格/图形分类
        ↓
    【表格处理层】Table Transformer → 行列结构 + 跨页合并
        ↓
    【后处理层】智能清洗 → 碎片合并 + 坐标统一
        ↓
    结构化语义块
    ```

    **设计亮点**
    - **双重解析各司其职**，pdfplumber 和 pypdf 库各自处理擅长的部分；
    - **双源文本保障信息不丢失**，优先使用解析的 pdf 原生文本，OCR 识别文本作为补充；
    - **模型推理重构表格信息**，兼容 onnx 和 ascend 两种推理后端，根据 OCR 识别文本框重构表格信息；
    - **跨页复杂场景，合并信息处理**，基于一系列规则，对文本，表格等跨页场景进行合理信息的合并，保证信息的完整性；
    - **“脏度”计算，删除低质页**，识别出目录，致谢等非正文页，进行删除，保证信息纯度。

    **DeepDOC** 代表了 RAGFlow 在文档理解领域的最高技术水平，通过工业化级的流水线设计和多重保障机制，为各类 pdf 文档提供可靠、精准、完整的解析服务。

### Plain Text 轻量解析（快速方案）

    适用场景：原生数字PDF、纯文本文档、快速预览需求
    - 极速提取：直接获取PDF原生文本，毫秒级响应
    - 资源节省：无需OCR计算，内存占用极少
### Vision Model 视觉解析（多模态方案）

    适用场景：扫描文档、图像丰富、传统OCR效果不佳的文档
    - 多模态理解：视觉大模型直接"看懂"页面内容
    - 语义描述：生成自然语言描述而非碎片文本
    - 复杂布局：处理传统方法难以解析的版面

# 手撕版

## 1. 布局识别器选择
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

### 2.1 PdfParser 基类
PdfParser() 中实现了以下重要功能：

**\_\_images__**： pdf 文档的数字化转换，将静态的 pdf 页面转换为可结构化数据，为后续的布局分析、表格提取等高级处理奠定基础。

**_layouts_rec**：结合上一步获取的内容页图像和 OCR 识别的文本框信息，进行文档每页的布局分析，和坐标重建。

**_table_transformer_job**：表格处理系统，根据布局分析得到的表格类型，对其表格数据内容进行提取处理。

**_text_merge**：基于规则合并文本，解决 OCR 识别中文本碎片化问题。

**_concat_downward**：按照阅读习惯将文本框按照 Y 坐标由上至下，X 坐标从左到右排序。

**_filter_forpages**：检测并过滤掉文档中的非正文内容页面，如目录页、致谢页等，提高后续处理的文本质量。

**_extract_table_figure**：提取表格，图片内容输出。针对表格的提取，进行了跨页合并，图像截取操作。

**__filterout_scraps**：与 `_text_merge` 功能类似，再次处理 OCR 识别碎片化文本，基于规则将碎片化文本进行组装还原。

按照 PdfParser() 实例化方法 /_/_call__ 中的功能来一一拆解功能的实现
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
#### 2.1.1 \_\_images__
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

**1）文本框检测，检测文档中文本框的位置**
```python
bxs = self.ocr.detect(np.array(img), device_id)
```
**2）格式还原，坐标除以缩放因子（ZM）还原到原始 pdf 尺寸后排序文本框。**
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
**3）字符与文本框匹配**
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
**4）文档重建，使用 pdfplumber 提取文字结合 OCR 检测文本框结构重建文档。**
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
**5）OCR 提取文字重建文档，pdfplumber 未提取到文字情况下，使用 OCR 方案进行文字识别填充重建文档。**
```python
texts = self.ocr.recognize_batch([b["box_image"] for b in boxes_to_reg], device_id)
for i in range(len(boxes_to_reg)):
    boxes_to_reg[i]["text"] = texts[i]  # 更新识别结果
    del boxes_to_reg[i]["box_image"]
```
**6）整理重建结果**
```python
bxs = [b for b in bxs if b["text"]]  # 过滤空文本框

# 更新字符高度统计（用于后续页面处理）
if self.mean_height[pagenum - 1] == 0:
    self.mean_height[pagenum - 1] = np.median([b["bottom"] - b["top"] for b in bxs])

self.boxes.append(bxs)  # 存储最终结果
```
#### 2.1.2 _layouts_rec
结合 pdfplumber 获取的内容业图像和 OCR 识别的文本框信息，进行文档每页的布局分析，和坐标重建。

1）布局分析，布局相较于文档框提供更高层次的文档语义信息。文档框中只包含位置坐标，所在页码，文本内容等信息，布局信息中包含文本类型信息，如是标题，正文，表格，页眉等，为后续针对性分析提供必要信息。
```python
self.boxes, self.page_layout = self.layouter(self.page_images, self.boxes, ZM, drop=drop)
```
2）坐标重建，文本框中的坐标信息是对于所在页的，坐标都是独立从 0 开始，重建坐标是基于整个文档的全局坐标。
```python
for i in range(len(self.boxes)):
    self.boxes[i]["top"] += self.page_cum_height[self.boxes[i]["page_number"] - 1]
    self.boxes[i]["bottom"] += self.page_cum_height[self.boxes[i]["page_number"] - 1]
```

#### 2.1.3 _table_transformer_job
表格处理系统，根据布局分析得到的表格类型，对其表格数据内容进行提取处理。

1）根据布局分析中的表格，记录每页表格数量，裁剪出表格图像并记录位置坐标。
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
**2）核心步骤：表结构识别**

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

3）坐标系统转换和整合

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

4）定义排序函数 gather

`sort_Y_firstly` 按 Y 坐标排序由上至下排序，若在同一行则按 X 坐标由左到右排序；`layouts_cleanup` 去除重叠和低质量的布局组件。
```python
def gather(kwd, fzy=10, ption=0.6):
    eles = Recognizer.sort_Y_firstly([r for r in self.tb_cpns if re.match(kwd, r["label"])], fzy)
    eles = Recognizer.layouts_cleanup(self.boxes, eles, 5, ption)
    return Recognizer.sort_Y_firstly(eles, 0)
```

5）表格结构组件排序

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
6）文本框与表结构关联

将 OCR 识别的文本框与排序后的表行，标题，文本通过坐标基于重叠度进行匹配，表列通过水平坐标进行匹配。最终输出
```python
for b in self.boxes:
    ii = Recognizer.find_overlapped_with_threshold(b, rows, thr=0.3)
    ii = Recognizer.find_overlapped_with_threshold(b, headers, thr=0.3)
    ii = Recognizer.find_horizontally_tightest_fit(b, clmns)
    ii = Recognizer.find_overlapped_with_threshold(b, spans, thr=0.3)
```
最终输出完整的符合文本框坐标的相应的表格内容。

#### 2.1.4 _filter_forpages
1）识别目录页等非正文内容页面
```python
i = 0
while i < len(self.boxes):
    # 检查文本框内容是否匹配目录页特征
    text_content = re.sub(r"( | |\u3000)+", "", self.boxes[i]["text"].lower())
    if not re.match(r"(contents|目录|目次|table of contents|致谢|acknowledge)$", text_content):
        i += 1
        continue
```
2）识别关键非正文信息进行过滤

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

3）对“脏页”进行删除

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

#### 2.1.5 _extract_table_figure
**1）提取文档图像，表格内容图像，并规划合并方案。**
```python
tables = {}
figures = {}
# extract figure and table boxes
i = 0
lst_lout_no = "" # 记录上一个处理的布局标识符
nomerge_lout_no = [] # 不合并的布局列表
while i < len(self.boxes):
    if "layoutno" not in self.boxes[i]:
        i += 1
        continue
    # 生成当前文本框的布局标识符： "页码-布局号"
    lout_no = str(self.boxes[i]["page_number"]) + "-" + str(self.boxes[i]["layoutno"])
    # 检查当前文本框是否是标题类内容
    if TableStructureRecognizer.is_caption(self.boxes[i]) or self.boxes[i]["layout_type"] in ["table caption", "title", "figure caption", "reference"]:
        # 关键：将前一个布局标识符加入不合并列表
        nomerge_lout_no.append(lst_lout_no)
    if self.boxes[i]["layout_type"] == "table":
        if re.match(r"(数据|资料|图表)*来源[:： ]", self.boxes[i]["text"]):
            self.boxes.pop(i)
            continue
        if lout_no not in tables:
            tables[lout_no] = []
        tables[lout_no].append(self.boxes[i])
        self.boxes.pop(i)
        lst_lout_no = lout_no
        continue
    if need_image and self.boxes[i]["layout_type"] == "figure":
        if re.match(r"(数据|资料|图表)*来源[:： ]", self.boxes[i]["text"]):
            self.boxes.pop(i)
            continue
        if lout_no not in figures:
            figures[lout_no] = []
        figures[lout_no].append(self.boxes[i])
        self.boxes.pop(i)
        lst_lout_no = lout_no
        continue
    i += 1
```
这里针对跨页表格合并的场景有一个十分精妙的设计，`nomerge_lout_no` 记录上一个布局内容，表示不需要与下一个布局内容进行合并。
```
场景1: 表格+标题的正常情况
页面内容顺序：
[表格内容] → [表格标题] → [下一个表格内容]

处理过程：
1. 处理表格内容：lout_no = "1-0", lst_lout_no = "1-0"
2. 遇到表格标题：检测为caption → 将前一个lst_lout_no("1-0")加入nomerge_lout_no
3. 继续处理...

结果：
表格"1-0"被标记为不合并，防止与后续内容错误合并。
-----------------------------------------------------------------------------------------
场景2: 跨页表格
第1页: [表格部分A] → [表格标题]
第2页: [表格部分B]

处理过程：
第1页：
1. 处理表格A：lout_no = "1-0", lst_lout_no = "1-0"  
2. 遇到标题：标记"1-0"为不合并

第2页：
1. 处理表格B：lout_no = "2-0", lst_lout_no = "2-0"
2. 跨页合并检查时："1-0"在nomerge_lout_no中 → 跳过合并

结果：
防止了表格A（带标题）与表格B错误合并。
-----------------------------------------------------------------------------------------
场景3: 图形标题的处理
页面内容：
[图形内容] → [图形标题] → [正文文本]

处理过程：
1. 处理图形：lout_no = "1-1", lst_lout_no = "1-1"
2. 遇到图形标题：标记"1-1"为不合并
3. 后续正文不会被错误合并到图形区域
```
**2）跨页合并**

针对三种情况不进行合并：标记为不合并，同页不合并，跨多页不合并，垂直距离检查大于23倍字符高度不合并（这可能是经过大量实验分析得出的最优经验值）
```python
nomerge_lout_no = set(nomerge_lout_no)
tbls = sorted([(k, bxs) for k, bxs in tables.items()], key=lambda x: (x[1][0]["top"], x[1][0]["x0"]))

i = len(tbls) - 1
while i - 1 >= 0:
    k0, bxs0 = tbls[i - 1]
    k, bxs = tbls[i]
    i -= 1
    if k0 in nomerge_lout_no:
        continue
    if bxs[0]["page_number"] == bxs0[0]["page_number"]:
        continue
    if bxs[0]["page_number"] - bxs0[0]["page_number"] > 1:
        continue
    mh = self.mean_height[bxs[0]["page_number"] - 1]
    if self._y_dis(bxs0[-1], bxs[0]) > mh * 23:
        continue
    tables[k0].extend(tables[k])
    del tables[k]
```
**3）标题检测和关联**

标题坐标检测
```python
c = self.boxes[i]
def nearest(tbls):
mink = ""; minv = 1000000000
for k, bxs in tbls.items():
    for b in bxs:
        if b.get("layout_type", "").find("caption") >= 0: continue
        y_dis = self._y_dis(c, b)  # 垂直距离
        x_dis = self._x_dis(c, b) if not x_overlapped(c, b) else 0  # 水平距离
        dis = y_dis * y_dis + x_dis * x_dis  # 欧氏距离平方
        if dis < minv:
            mink = k; minv = dis
return mink, minv
```
标题关联
```python
if tv < fv and tk:
    tables[tk].insert(0, c)
    logging.debug("TABLE:" + self.boxes[i]["text"] + "; Cap: " + tk)
elif fk:
    figures[fk].insert(0, c)
    logging.debug("FIGURE:" + self.boxes[i]["text"] + "; Cap: " + tk)
```

**4）对图像内容进行裁剪处理**
```python
if separate_tables_figures:
    figure_results.append((cropout(bxs, "figure", poss), [txt]))
    figure_positions.append(poss)
else:
    res.append((cropout(bxs, "figure", poss), [txt]))
    positions.append(poss)
```
**5）对表格内容进行图像裁剪处理**
```python
res.append((cropout(bxs, "table", poss), 
               self.tbl_det.construct_table(bxs, html=return_html, is_english=self.is_english)))
```
**6）最终结果组装返回**
```python
if separate_tables_figures:
    assert len(positions) + len(figure_positions) == len(res) + len(figure_results)
    if need_position:
        return list(zip(res, positions)), list(zip(figure_results, figure_positions))
    else:
        return res, figure_results
else:
    assert len(positions) == len(res)
    if need_position:
        return list(zip(res, positions))
    else:
        return res
```
> *Tips：已经对表格内容进行过文本提取，为什么还需要进行图像截取*

> "文本+图像"的双重提取策略确保了文档处理系统既能够理解内容，又能够保留形式，满足了真实业务场景中的多样化需求。
>1. 在 pdf 文档提取中 OCR 识别是有误差的，图像是作为校验和保底方案。
>2. 表格中的排版信息只通过文本会丢失很多视觉信息；有很多复杂表格是无法只用文本描述的。
>3. 在特殊场景，如法律，审计等，是要求原始视觉档案。
>4. 可以结合多模态进一步校准信息。

基于对 PdfParser 类的分析，所有功能方法都在处理和丰富 self.boxes 列表，最终输出的 self.boxes 列表中的元素主要包含以下结构化信息：
**1）基础位置信息**
```python
{
    "page_number": 1,           # 元素所在的页码（从1开始）
    "x0": 100.5,               # 元素左边界X坐标
    "x1": 300.2,               # 元素右边界X坐标  
    "top": 50.8,               # 元素上边界Y坐标（相对于文档顶部）
    "bottom": 70.3,            # 元素下边界Y坐标（相对于文档顶部）
}
```
**2）文本内容**
```python
{
    "text": "这是文本内容",      # 元素的文本内容
    "layout_type": "text",      # 布局类型：text, table, figure, equation等
}
```
**3）布局和结构信息**
```python
{
    "layoutno": 0,             # 布局编号，同一布局内的元素有相同编号
    "col_id": 0,               # 列ID，用于多列布局
    "in_row": 3,               # 同一行中的元素数量
}
```
**4）表格相关字段（如果是表格元素）**
```python
{
    "R": 2,                    # 行索引
    "R_top": 100.0,            # 行顶部位置
    "R_bott": 120.0,           # 行底部位置
    "C": 1,                    # 列索引  
    "C_left": 150.0,           # 列左边界
    "C_right": 250.0,          # 列右边界
    "H": 0,                    # 表头索引
    "H_top": 80.0,             # 表头顶部
    "H_bott": 95.0,            # 表头底部
    "SP": 1,                   # 跨行/跨列标识
}
```
**5）图像和位置标签**
```python
{
    "position_tag": "@@1\t100.5\t300.2\t50.8\t70.3##",  # 位置标签，用于图像裁剪
    "image": <PIL.Image>,       # 对应的图像片段
    "positions": [              # 详细位置信息数组
        [1, 100, 300, 50, 70]  # [页码, x0, x1, top, bottom]
    ]
}
```

### 2.2 Pdf 类
Pdf 类作为入口点，调用 PdfParser 中提供的功能实现整个复杂的文档处理流程，并记录了各阶段耗时，解析进度等信息，并对解析后的文档最后通过规则进行多栏排版识别，垂直文档排序，跨页文本合并等操作。

```python
def __call__(self, filename, binary=None, from_page=0,
                 to_page=100000, zoomin=3, callback=None, separate_tables_figures=False):
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
    self._text_merge(zoomin=zoomin)
    if separate_tables_figures:
        # 提取表格，图片内容输出
        tbls, figures = self._extract_table_figure(True, zoomin, True, True, True)
        # 简单文本排序
        self._concat_downward()
        return [(b["text"], self._line_tag(b, zoomin)) for b in self.boxes], tbls, figures
    else:
        # 提取表格，图片内容输出
        tbls = self._extract_table_figure(True, zoomin, True, True)
        # 垂直方向上智能合并相邻的文本框
        self._naive_vertical_merge()
        self._concat_downward()
        # 按照阅读习惯排序
        self._final_reading_order_merge()
        return [(b["text"], self._line_tag(b, zoomin)) for b in self.boxes], tbls
```

在 `_naive_vertical_merge` 函数中实现垂直方向上智能合并相邻的文本框功能，可以简单看以下两个设计点：

**1）多栏布局排版识别，重排序**
```python
# 计算典型列宽
column_width = np.median([b["x1"] - b["x0"] for b in self.boxes])
if not column_width or math.isnan(column_width):
    column_width = self.mean_width[0]  # 备用方案

# 推断页面列数
self.column_num = int(self.page_images[0].size[0] / zoomin / column_width)
if column_width < self.page_images[0].size[0] / zoomin / self.column_num:
    logging.info("Multi-column................... {} {}".format(column_width, self.page_images[0].size[0] / zoomin / self.column_num))
    # 多栏文档：按X坐标重新排序
    self.boxes = self.sort_X_by_page(self.boxes, column_width / self.column_num)
```

**2）合并规则**
```python
# 合并规则
concatting_feats = [
    b["text"].strip()[-1] in ",;:'\"，、‘“；：-",                    # 以连接符号结尾
    len(b["text"].strip()) > 1 and b["text"].strip()[-2] in ",;:'\"，‘“、；：",  # 倒数第二个字符是连接符
    b_["text"].strip() and b_["text"].strip()[0] in "。；？！?”）),，、：",  # 以下一句符号开头
]
# 不合并规则
feats = [
    b.get("layoutno", 0) != b_.get("layoutno", 0),  # 不同布局区域
    b["text"].strip()[-1] in "。？！?",              # 以句子结束符结尾
    self.is_english and b["text"].strip()[-1] in ".!?",  # 英文句子结束
    b["page_number"] == b_["page_number"] and b_["top"] - b["bottom"] > self.mean_height[b["page_number"] - 1] * 1.5,  # 垂直间距过大
    b["page_number"] < b_["page_number"] and abs(b["x0"] - b_["x0"]) > self.mean_width[b["page_number"] - 1] * 4,  # 跨页水平错位
]
```

通过 `pdf_parser = Pdf()` 解析后的 `sections, tables, figures` 按照与 docx 文档相同的处理流程，先通过视觉模型（Vision Model） 识别图片内容，后进行分词器 `tokenize_table` 处理后期望输出的数据结构。这两个功能具体实现可参考上一章《naive Parser 语义切块（docx 篇）》最后部分。

## 3. Plain Text 布局识别器
如果不采用 DeepDOC，而是指定的是 Plain Text 布局识别器。

解析器初始化
``` python
if layout_recognizer == "Plain Text":
    pdf_parser = PlainParser()
```

`PlainParser()` 只是基于`pypdf` 库与 DFS 算法实现的极简的PDF文本提取方案，只提取原始文本内容，不进行复杂的OCR、布局分析或图像处理。这部分实现与 DeepDOC 布局识别器中 `__images__` 方法中使用 `pypdf` 库部分实现的功能相同。

## 4. 视觉模型布局识别器
若没有指定任何布局识别器，则会采用视觉模型（Vision Model）对 pdf 页面图像进行识别并生成文本描述。
```python
vision_model = LLMBundle(kwargs["tenant_id"], LLMType.IMAGE2TEXT, llm_name=layout_recognizer, lang=lang)
pdf_parser = VisionParser(vision_model=vision_model, **kwargs)
```

VisionParser 中包含两个重要步骤：

1）使用 `pdfplumber` 库将 pdf 文档转换成图像
```python
self.pdf = pdfplumber.open(fnm) if isinstance(fnm, str) else pdfplumber.open(BytesIO(fnm))
```
2）使用视觉大模型识别图像生成文本
```python
text = picture_vision_llm_chunk(
    binary=img_binary,
    vision_model=self.vision_model,
    prompt=vision_llm_describe_prompt(page=pdf_page_num + 1),
    callback=callback,
)
```

## 5. sections 后处理

经过上述一系列处理得到 sections，针对有图和无图 sections 两种场景，分别对 sections 进行后处理输出。有图 sections 处理逻辑只有在处理 markdown 文档时才用到，在拆解 markdown 介绍，这里来看卡“无图”文档处理逻辑。
```python
if section_images:
   # 有图文档处理
else:
   # 无图文档处理
```
> *注意：这里的有图和无图 sections 不是指整个文档中是否包含图片，而是指 sections 中是否包含图片信息，也可以理解为图片是否额外解析出来存储在单独的变量中。*

> *docx, pdf 文档解析器输出的结构体（sections）中已经包含了文本和图片的信息，在这个判断中被认为无图（section_images==False）*

>*markdown 解析器会单独解析出文档中的图片存储在 section_images 变量中，还需要将文本与图片进行对应关联，在这个判断中被认为有图（section_images==True）*

### 5.1 无图
```python
chunks = naive_merge(
    sections, int(parser_config.get(
        "chunk_token_num", 128)), parser_config.get(
        "delimiter", "\n!?。；！？"))
if kwargs.get("section_only", False):
    return chunks

res.extend(tokenize_chunks(chunks, doc, is_english, pdf_parser))
```

#### 5.1.1 naive_merge
根据传入的 chunk size 和分隔符，将文本进行规则切分后输出。与 docx 文档的**合并切块**处理逻辑类似。
#### 5.1.2 tokenize_chunks
返回统一格式的结构化文档块，作为后续向量化输入。
- 非 pdf 文档处理，添加位置信息后经过分词器处理输出；
- pdf 文档处理，会经过 `pdf_parser.crop(ck, need_position=True)` 处理，它是根据文本在 PDF 中的位置，将对应区域裁剪成图片（可多页），并可选返回坐标信息。处理后添加位置信息后经过分词器处理输出。
```python
def tokenize_chunks(chunks, doc, eng, pdf_parser=None):
    res = []
    # wrap up as es documents
    for ii, ck in enumerate(chunks):
        if len(ck.strip()) == 0:
            continue
        logging.debug("-- {}".format(ck))
        d = copy.deepcopy(doc)
        if pdf_parser:
            try:
                d["image"], poss = pdf_parser.crop(ck, need_position=True)
                add_positions(d, poss)
                ck = pdf_parser.remove_tag(ck)
            except NotImplementedError:
                pass
        else:
            add_positions(d, [[ii]*5])
        tokenize(d, ck, eng)
        res.append(d)
    return res
``` 

# 下期预告
在本期《【解密源码】RAGFlow 切分最佳实践- naive parser 语义切块（pdf 篇）》中，我们深入剖析了 .pdf 文档在 RAGFlow 中的完整解析流水线，看到了 RAGFlow 如何将结构丰富的 .pdf 文档转化为高质量的语义块，为后续的向量化和检索奠定坚实基础。

在下一期中，我们将深入剖析 naive parser 下 .csv|xlsx 文件和 txt 文件以及各种代码文件的语义切块方案。

# 版本更新补充信息
在最新的 0.21.1 版本，RAGFlow 中布局识别器支持 MinerU，Docling 和 TecentADP 知识库应用的接入。流程上是用这些工具的解析能力获取文本内容，将文本内容加工成 RAGFlow 期望的格式进行后续向量化的操作，这里不做过多介绍，有兴趣的可以去对应平台了解研究。
```python
elif layout_recognizer == "MinerU":
    mineru_executable = os.environ.get("MINERU_EXECUTABLE", "mineru")
    pdf_parser = MinerUParser(mineru_path=mineru_executable)
    sections, tables = pdf_parser.parse_pdf(
        filepath=filename,
        binary=binary,
        callback=callback,
        output_dir=os.environ.get("MINERU_OUTPUT_DIR", ""),
        backend=os.environ.get("MINERU_BACKEND", "pipeline"),
        delete_output=bool(int(os.environ.get("MINERU_DELETE_OUTPUT", 1))),
    )

elif layout_recognizer == "Docling":
    pdf_parser = DoclingParser()
    sections, tables = pdf_parser.parse_pdf(
        filepath=filename,
        binary=binary,
        callback=callback,
        output_dir=os.environ.get("MINERU_OUTPUT_DIR", ""),
        delete_output=bool(int(os.environ.get("MINERU_DELETE_OUTPUT", 1))),
    )
    parser_config["chunk_token_num"] = 0
    res = tokenize_table(tables, doc, is_english)

elif layout_recognizer == "TCADP Parser":
    tcadp_parser = TCADPParser()
    sections, tables = tcadp_parser.parse_pdf(
        filepath=filename,
        binary=binary,
        callback=callback,
        output_dir=os.environ.get("TCADP_OUTPUT_DIR", ""),
        file_type="PDF"
    )
```
