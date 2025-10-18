---
title: "【解密源码】 RAGFlow 切分最佳实践-navie 模式 docx 文件"
date: 2025-10-18T10:39:10+08:00
draft: true
tags: ["源码","技术",RAG]
categories: ["RAGFlow"]
---
# 引言

通过公共部分 PART 3 中的引用，可以看到 RAGFlow 的解析器实现在 rag/app 下各个文件中的 chunk 函数。这里对 naive 方案进行一个详细的拆解和分析。

# 省流版

代码文件阅读顺序：

原理：

# 手撕版

## 文件分类

**以上是针对所有文件类型的统一解析配置，接下来就是根据不同文件后缀名分别处理**

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

## .docx 文件解析

重点部分，对 docx 文件内容进行切分

```python
sections, tables = Docx()(filename, binary)
```

---

可以看到 Docx 继承基类 DocxParser

```python
class Docx(DocxParser):
  def __init__(self):
    pass
```

### DocxParser 基类

```python
class RAGFlowDocxParser:
	def __extract_table_content(self, tb):
			df = []
	    for row in tb.rows:
	        df.append([c.text for c in row.cells])
	    return self.__compose_table_content(pd.DataFrame(df))
	def __compose_table_content(self, df):
			...
	def __call__(self, fnm, from_page=0, to_page=100000000):
		self.doc = Document(fnm) if isinstance(fnm, str) else Document(BytesIO(fnm))
    pn = 0 # parsed page
    secs = [] # parsed contents
    for p in self.doc.paragraphs:
        if pn > to_page:
            break

        runs_within_single_paragraph = [] # save runs within the range of pages
        for run in p.runs:
            if pn > to_page:
                break
            if from_page <= pn < to_page and p.text.strip():
                runs_within_single_paragraph.append(run.text) # append run.text first

            # wrap page break checker into a static method
            if 'lastRenderedPageBreak' in run._element.xml:
                pn += 1

        secs.append(("".join(runs_within_single_paragraph), p.style.name if hasattr(p.style, 'name') else '')) # then concat run.text as part of the paragraph

    tbls = [self.__extract_table_content(tb) for tb in self.doc.tables]
    return secs, tbls
```

基类 DocxParser ****中有以下函数：

**__extract_table_content**：表格内容提取入口

**__compose_table_content**：表格内容提取核心

**\_\_call__**：docx 文档中的段落和表格分别进行处理

#### __compose_table_content

```python
 def blockType(b):
    pattern = [
        ("^(20|19)[0-9]{2}[年/-][0-9]{1,2}[月/-][0-9]{1,2}日*$", "Dt"), # 日期-年月日
        (r"^(20|19)[0-9]{2}年$", "Dt"), # 日期-年
        (r"^(20|19)[0-9]{2}[年/-][0-9]{1,2}月*$", "Dt"), # 日期-年月
        ("^[0-9]{1,2}[月/-][0-9]{1,2}日*$", "Dt"), # 日期-月日
        (r"^第*[一二三四1-4]季度$", "Dt"), # 日期-季度
        (r"^(20|19)[0-9]{2}年*[一二三四1-4]季度$", "Dt"), # 日期-年季度
        (r"^(20|19)[0-9]{2}[ABCDE]$", "DT"), # 日期-年分类
        ("^[0-9.,+%/ -]+$", "Nu"), # 纯数字
        (r"^[0-9A-Z/\._~-]+$", "Ca"), # 代码类数据
        (r"^[A-Z]*[a-z' -]+$", "En"), # 英文文本
        (r"^[0-9.,+-]+[0-9A-Za-z/$￥%<>（）()' -]+$", "NE"), # 数字+文本
        (r"^.{1}$", "Sg") # 单字符
    ]
    for p, n in pattern:
        if re.search(p, b):
            return n
    tks = [t for t in rag_tokenizer.tokenize(b).split() if len(t) > 1]
    if len(tks) > 3:
        if len(tks) < 12:
            return "Tx" # 短文本
        else:
            return "Lx" # 长文本

    if len(tks) == 1 and rag_tokenizer.tag(tks[0]) == "nr":
        return "Nr" # 人名

    return "Ot" # 其他
```

设计了 11 种文本类型，通过 tokenize 对表格中的文本进行类型判定，打上相应标签。

```python
max_type = Counter([blockType(str(df.iloc[i, j])) for i in range(
    1, len(df)) for j in range(len(df.iloc[i, :]))])
max_type = max(max_type.items(), key=lambda x: x[1])[0]
```

从表格第二行开始逐个对每行每列中的信息进行类型分析，汇总各行中所有类型取最多频率最高的类型作为该表格类型。

*Tips：从第二行获取是避免表格表头的影响。*

```python
hdrows = [0]  # header is not necessarily appear in the first line
```

考虑表头不在第一行的场景。

```python
if max_type == "Nu":
  for r in range(1, len(df)):
      tys = Counter([blockType(str(df.iloc[r, j]))
                    for j in range(len(df.iloc[r, :]))])
      tys = max(tys.items(), key=lambda x: x[1])[0]
      if tys != max_type:  # 数据类型不是数值类型
          hdrows.append(r) # 识别为表头
```

对数值类型的表格进行表头的确认，因为数值类型的表格可能存在中间表头，且有明显的结构特点，如下：

| … | … | … | … | … | … |
| --- | --- | --- | --- | --- | --- |
| 部门 | 季度 | 2023Q1 | 2023Q2 | 2023Q3 | 2023Q4 |
| 销售部 | 收入 | 100 | 120 | 130 | 140 |
| 销售部 | 成本 | 80 | 90 | 95 | 100 |
| 技术部 | 收入 | 200 | 210 | 220 | 230 |

例如上表中第二行的时间类型。结合代码针对数值类型的表格通过类型判断，识别出是表头。

```python
lines = []
for i in range(1, len(df)):
    if i in hdrows: 
        continue
    
    # 关键步骤：计算相对表头位置
    hr = [r - i for r in hdrows]
    hr = [r for r in hr if r < 0]
```

计算表头行和内容行位置，只保留当前内容行上方的表头。

```python
t = len(hr) - 1
while t > 0:
  if hr[t] - hr[t - 1] > 1:  # 检查表头之间是否存在其他内容
      hr = hr[t:]
      break
  t -= 1
```

检查相邻表头之间是否存在内容行，如果表头间隔大于 1，说明存在内容行，取最近的表头。这个操作主要是解决多层表头问题。

```python
headers = []
for j in range(len(df.iloc[i, :])):
  t = []
  for h in hr:
      x = str(df.iloc[i + h, j]).strip()
      if x in t:
          continue
      t.append(x)
  t = ",".join(t)
  if t:
      t += ": "
  headers.append(t)
```

遍历表头信息中每一列信息。

```python
cells = []
for j in range(len(df.iloc[i, :])):
    if not str(df.iloc[i, j]): # 跳过空格单元格
        continue
    cells.append(headers[j] + str(df.iloc[i, j]))
lines.append(";".join(cells))
```

遍历内容行中每一列信息，与对应比表头进行组合。

```python
colnm = len(df.iloc[0, :])
if colnm > 3:
    return lines
return ["\n".join(lines)]
```

输出格式美化，列数多的表格按照分割符形式单行输出，列数少的表格按照更易读的换行形式输出。

**总结：DocxParser 基类中所有功能主要包括解析出 docx 文件中 paragraphs 和 tables，通过固定格式输出。**

---

### Docx 类

```python
class Docx(DocxParser):
		def get_picture(self, document, paragraph):
				...
		def __clean(self, line):
        line = re.sub(r"\u3000", " ", line).strip()
        return line
    def __get_nearest_title(self, table_index, filename):
		    ...
		def __call__(self, filename, binary=None, from_page=0, to_page=100000):
				...
```

类 **Docx** 中有以下函数：

**get_picture**：从指定的 word 段落中提取所有内嵌图片，并合并为一张图片返回。输出的 **PIL (Pillow) Image 对象，颜色模式是 RGB。**

**__clean**：替换全角空格为半角。

**__get_nearest_title**：获取内容相关标题，构建标题链。

**\_\_call__**：对 docx 文档中的段落和表格分别进行处理

#### __get_nearest_title

```python
# Get document name from filename parameter
doc_name = re.sub(r"\.[a-zA-Z]+$", "", filename)
if not doc_name:
    doc_name = "Untitled Document"
```

从文件名中提取文档标题。

```python
blocks = []
for i, block in enumerate(self.doc._element.body):
    if block.tag.endswith('p'):  # 段落
        p = Paragraph(block, self.doc)
        blocks.append(('p', i, p))
    elif block.tag.endswith('tbl'):  # 表格
        blocks.append(('t', i, None))
```

构建完整的文档结构映射（包含段落，表格）。

```python
target_table_pos = -1
table_count = 0
for i, (block_type, pos, _) in enumerate(blocks):
    if block_type == 't':
        if table_count == table_index: # 表格索引
            target_table_pos = pos
            break
        table_count += 1
```

通过外部参数传入的表格索引 table_index，完成当前表格在文档中绝对位置的映射。

```python
nearest_title = None
for i in range(len(blocks)-1, -1, -1):
    block_type, pos, block = blocks[i]
    if pos >= target_table_pos:  # Skip blocks after the table
        continue

    if block_type != 'p':
        continue

    if block.style and block.style.name and re.search(r"Heading\s*(\d+)", block.style.name, re.I):
        try:
            level_match = re.search(r"(\d+)", block.style.name)
            if level_match:
                level = int(level_match.group(1))
                if level <= 7:  # Support up to 7 heading levels
                    title_text = block.text.strip()
                    if title_text:  # Avoid empty titles
                        nearest_title = (level, title_text)
                        break
        except Exception as e:
            logging.error(f"Error parsing heading level: {e}")
```

反向遍历文档结构，获取当前表格的最近的标题以及标题等级，进行关联。

```python
if nearest_title:
    # Add current title
    titles.append(nearest_title)
    current_level = nearest_title[0]

    # Find all parent headings, allowing cross-level search
    while current_level > 1:
        found = False
        for i in range(len(blocks)-1, -1, -1):
            block_type, pos, block = blocks[i]
						...
            if block.style and re.search(r"Heading\s*(\d+)", block.style.name, re.I):
                try:
                    level_match = re.search(r"(\d+)", block.style.name)
                    if level_match:
                        level = int(level_match.group(1))
                        # Find any heading with a higher level
                        if level < current_level:
                            title_text = block.text.strip()
                            if title_text:  # Avoid empty titles
                                titles.append((level, title_text))
                                current_level = level
                                found = True
                                break
				...
```

如果关联不是一级标题，则逐级向上查找副标题，在 titles 中构建完整的标题链。

#### \_\_call__

```python
lines = []
last_image = None
for p in self.doc.paragraphs:
	...
	if p.text.strip():
	    if p.style and p.style.name == 'Caption': # 图注段落
	        former_image = None
	        if lines and lines[-1][1] and lines[-1][2] != 'Caption':
	            former_image = lines[-1][1].pop()
	        elif last_image:
	            former_image = last_image
	            last_image = None
	        lines.append((self.__clean(p.text), [former_image], p.style.name))
	    else: # 常规段落
	        current_image = self.get_picture(self.doc, p)
	        image_list = [current_image]
	        if last_image:
	            image_list.insert(0, last_image)
	            last_image = None
	        lines.append((self.__clean(p.text), image_list, p.style.name if p.style else ""))
	else: # 纯图片段落
	    if current_image := self.get_picture(self.doc, p):
	        if lines:
	            lines[-1][1].append(current_image)
	        else:
	            last_image = current_image
	            
...
new_line = [(line[0], reduce(concat_img, line[1]) if line[1] else None) for line in lines]
```

图片与段落文本内容建立关联，并将每个段落中的多张图片合并成单张图片。

```python
for run in p.runs:
    if 'lastRenderedPageBreak' in run._element.xml:
        pn += 1
        continue
    if 'w:br' in run._element.xml and 'type="page"' in run._element.xml:
        pn += 1
```

通过 XML 元素检测分页，进行页面计算。

```python
tbls = []
for i, tb in enumerate(self.doc.tables):
    title = self.__get_nearest_title(i, filename) # 获取层级标题
    html = "<table>"
    if title:
        html += f"<caption>Table Location: {title}</caption>"
    for r in tb.rows:
        html += "<tr>"
        i = 0
        try:
            while i < len(r.cells):
                span = 1
                c = r.cells[i]
                # 合并单元格检测，连续相同内容的单元格
                for j in range(i + 1, len(r.cells)):
                    if c.text == r.cells[j].text:
                        span += 1
                        i = j
                    else:
                        break
                i += 1
                html += f"<td>{c.text}</td>" if span == 1 else f"<td colspan='{span}'>{c.text}</td>"
        except Exception as e:
            logging.warning(f"Error parsing table, ignore: {e}")
        html += "</tr>"
    html += "</table>"
    tbls.append(((None, html), ""))
```

处理表格信息，获取表格多层级标题，以及表格内容构建 HTML table 内容。

```python
return new_line, tbls
```

最终输出内容格式：

```python
# new_line
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
```

**总结：Docx 类中主要针对 .docx 文档中的内容，包括图片和表格进行了格式化处理并输出。**

---

让我们回到对于 .docx 文档解析的主流程中

```python
sections, tables = Docx()(filename, binary)
```

sections 是包含图片信息的段落对象数组，tables 是包含表格信息的对象数组。

```python
# 创建 vision 模型对象
try:
    vision_model = LLMBundle(kwargs["tenant_id"], LLMType.IMAGE2TEXT)
    callback(0.15, "Visual model detected. Attempting to enhance figure extraction...")
except Exception:
    vision_model = None
...

# 使用 vision 模型对 sections 信息进行处理
if vision_model:
    figures_data = vision_figure_parser_figure_data_wrapper(sections) # 数据格式转换，将 sections 格式转换成后续需要处理的格式
    try:
        docx_vision_parser = VisionFigureParser(vision_model=vision_model, figures_data=figures_data, **kwargs)
        boosted_figures = docx_vision_parser(callback=callback)
        tables.extend(boosted_figures)
    except Exception as e:
        callback(0.6, f"Visual model error: {e}. Skipping figure parsing enhancement.")
```

vision_figure_parser_figure_data_wrapper 将包含图片信息的对象数组转换成 figures_data，如以下格式：

```python
(
		(figure_data[1], [figure_data[0]]), # 原始图片信息，图片描述信息
		[(0, 0, 0, 0, 0)], # 位置信息
)
```

### VisionFigureParser 类

```python
 def __init__(self, vision_model, figures_data, *args, **kwargs):
    self.vision_model = vision_model # 视觉模型
    self._extract_figures_info(figures_data) # 提取数据
    # 验证数据
    assert len(self.figures) == len(self.descriptions)
    assert not self.positions or (len(self.figures) == len(self.positions))
def _extract_figures_info(self, figures_data):
		...
def _assemble(self):
		...
def __call__(self, **kwargs):
		...
```

类 **VisionFigureParser** 中有以下函数：

**_extract_figures_info**：数据提取。将转换后的输入数据 figures_data 中的信息提取到 figures（原始图片信息），descriptions（图片描述信息），positions （位置信息）三个数组中。

**_assemble**：数据格式转换。将数据转换成输入数据 figures_data 格式。

**\_\_call__**：使用 vision 模型将图像转换成描述文本，与原描述文本合并后输出。

---

让我们再次回到对于 .docx 文档解析的主流程中

```python
res = tokenize_table(tables, doc, is_english)
```

对于表格内容进行分词处理。

### tokenize_table —— 表格分词

```python
if isinstance(rows, str):
    d = copy.deepcopy(doc)
    tokenize(d, rows, eng)
    d["content_with_weight"] = rows
    if img:
        d["image"] = img
        d["doc_type_kwd"] = "image"
    if poss:
        add_positions(d, poss)
    res.append(d)
    continue
```

对于已预处理成单个字符串的表格内容进行处理。

```python
de = "; " if eng else "； "
```

根据语言进行分隔符的选择。

```python
for i in range(0, len(rows), batch_size):
    d = copy.deepcopy(doc)
    r = de.join(rows[i:i + batch_size])
    tokenize(d, r, eng)
    if img:
        d["image"] = img
        d["doc_type_kwd"] = "image"
    add_positions(d, poss)
    res.append(d)
```

对多列大表格进行列分批处理。

```python
res = [
    {
        "content": "分词后的表格内容",
        "content_with_weight": "原始表格内容",
        "image": 可选的图片对象,
        "doc_type_kwd": "image",
        "positions": 位置信息,
        ... # 其他字段
    },
    ... # 多个文档对象
]
```

经过 tokenize_table 处理后期望输出的数据结构。

---

让我们再再次回到对于 .docx 文档解析的主流程中。

```python
chunks, images = naive_merge_docx(
    sections, int(parser_config.get(
        "chunk_token_num", 128)), parser_config.get(
        "delimiter", "\n!?。；！？"))
```

### naive_merge_docx —— chunk 处理

```python
 cks = [""]
images = [None]
tk_nums = [0]
def add_chunk(t, image, pos=""):
    nonlocal cks, tk_nums, delimiter
    tnum = num_tokens_from_string(t)
    if tnum < 8:
        pos = ""
    if cks[-1] == "" or tk_nums[-1] > chunk_token_num:
        if t.find(pos) < 0:
            t += pos
        cks.append(t)
        images.append(image)
        tk_nums.append(tnum)
    else:
        if cks[-1].find(pos) < 0:
            t += pos
        cks[-1] += t
        images[-1] = concat_img(images[-1], image)
        tk_nums[-1] += tnum
```

add_chunk 对不同大小的 chunk 块进行处理：

- 小于 8 token 大小不进行处理
- 大于 chunk_token_num（默认128）进行新块创建
- 小于 chunk_token_num（默认128）进行合并

输出结构保持同一个段落下所有 chunks 与 images 的关联性。

```python
dels = get_delimiters(delimiter)
for sec, image in sections:
    if not image:
        line += sec + "\n"
        continue
    split_sec = re.split(r"(%s)" % dels, line + sec)
    for sub_sec in split_sec:
        if re.match(f"^{dels}$", sub_sec):
            continue
        add_chunk(sub_sec, image,"")
    line = ""
```

获取分隔符正则表达式，进行段落 chunk 处理。

---

让我们最后一次回到对于 .docx 文档解析的主流程中。

```python
res.extend(tokenize_chunks_with_images(chunks, doc, is_english, images))
return res # 整个 .docx 文档解析的输出
```

将上个步骤的 chunks, images 进行二次处理符合最终输出格式 res