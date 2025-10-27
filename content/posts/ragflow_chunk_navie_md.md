---
title: "【解密源码】 RAGFlow 切分最佳实践- naive parser 语义切块（markdown 篇）"
date: 2025-10-26T11:10:00+08:00
draft: false
tags: ["源码","技术","RAG"]
categories: ["RAGFlow"]
---

# 引言

# 省流版

# 手撕版

1. 解析器初始化
```python
markdown_parser = Markdown(int(parser_config.get("chunk_token_num", 128)))
```

## Markdown 类
继承 `MarkdownParser` 基类，解析表格；使用 `MarkdownElementExtractor().extract_elements()` 解析文本内容。
```python
class Markdown(MarkdownParser):
    def __call__(self, filename, binary=None, separate_tables=True):
        if binary:
            encoding = find_codec(binary)
            txt = binary.decode(encoding, errors="ignore")
        else:
            with open(filename, "r") as f:
                txt = f.read()

        remainder, tables = self.extract_tables_and_remainder(f'{txt}\n', separate_tables=separate_tables)

        extractor = MarkdownElementExtractor(txt)
        element_sections = extractor.extract_elements()
        sections = [(element, "") for element in element_sections]

        tbls = []
        for table in tables:
            tbls.append(((None, markdown(table, extensions=['markdown.extensions.tables'])), ""))
        return sections, tbls
```

## MarkdownParser 基类

**extract_tables_and_remainder**：提取文档中的表格以及其他部分。

### extract_tables_and_remainder
**1. 两种表格识别方案，兼容 markdown 中两种表格的语法形式。**
```python
if "|" in markdown_text:  # for optimize performance
    # 标准 Markdown 表格
    border_table_pattern = re.compile(
        r"""
        (?:\n|^)
        (?:\|.*?\|.*?\|.*?\n)
        (?:\|(?:\s*[:-]+[-| :]*\s*)\|.*?\n)
        (?:\|.*?\|.*?\|.*?\n)+
    """,
        re.VERBOSE,
    )
    working_text = replace_tables_with_rendered_html(border_table_pattern, tables)

    # 无边框 Markdown 表格
    no_border_table_pattern = re.compile(
        r"""
        (?:\n|^)
        (?:\S.*?\|.*?\n)
        (?:(?:\s*[:-]+[-| :]*\s*).*?\n)
        (?:\S.*?\|.*?\n)+
        """,
        re.VERBOSE,
    )
    working_text = replace_tables_with_rendered_html(no_border_table_pattern, tables)
```
**标准 Markdown 表格**

    1. 使用管道符 | 包围所有单元格
    2. 必须有分隔线行（第二行）
    3. 单元格内容可以包含空格

    | 姓名 | 年龄 | 部门 | 薪资 |
    |------|------|------|------|
    | 张三 | 25   | 技术部 | 15000 |
    | 李四 | 30   | 销售部 | 12000 |
    | 王五 | 28   | 市场部 | 13000 |

**无边框 Markdown 表格**

    1. 只有分隔线行使用 | 和 -
    2. 数据行的单元格不被 | 包围

    姓名 年龄 部门 薪资
    ---- ---- ---- -----
    张三 25   技术部 15000
    李四 30   销售部 12000
    王五 28   市场部 13000

**2. 两种表格解析方案**
```python
if separate_tables:
    # Skip this match (i.e., remove it)
    new_text += working_text[last_end : match.start()] + "\n\n"
else:
    # Replace with rendered HTML
    html_table = markdown(raw_table, extensions=["markdown.extensions.tables"]) if render else raw_table
    new_text += working_text[last_end : match.start()] + html_table + "\n\n"
```
**表格分离模式 (separate_tables=True)**

将整个表格转换成文本，保证语义完整性，作为后续切分和量化的数据基础。
```
输入 Markdown:
正文...
| 姓名 | 年龄 | 部门 |
|------|------|------|
| 张三 | 25   | 技术 |
正文...

输出:
正文...
正文...

表格单独存储: ["| 姓名 | 年龄 | 部门 |\n|------|------|------|\n| 张三 | 25   | 技术 |"]
```

**表格渲染模式 (separate_tables=False)**

保持原始格式和样式，满足重现文档外观需要。
```
输入 Markdown:
正文...
| 姓名 | 年龄 | 部门 |
|------|------|------|
| 张三 | 25   | 技术 |
正文...

输出:
正文...
<table>
  <tr><th>姓名</th><th>年龄</th><th>部门</th></tr>
  <tr><td>张三</td><td>25</td><td>技术</td></tr>
</table>
正文...
```

## MarkdownElementExtractor 类
建立针对 Markdown 语法规则，通过不同规则解析文本中的各个元素。
```python
def extract_elements(self):
    sections = []
    i = 0
    while i < len(self.lines):
        line = self.lines[i]
        
        # 优先级解析顺序
        if re.match(r"^#{1,6}\s+.*$", line):           # 1. 标题
            element = self._extract_header(i)
        elif line.strip().startswith("```"):           # 2. 代码块
            element = self._extract_code_block(i)
        elif re.match(r"^\s*[-*+]\s+.*$", line) or \   # 3. 列表
             re.match(r"^\s*\d+\.\s+.*$", line):
            element = self._extract_list_block(i)
        elif line.strip().startswith(">"):             # 4. 引用块
            element = self._extract_blockquote(i)
        elif line.strip():                             # 5. 文本块
            element = self._extract_text_block(i)
        else:                                          # 6. 空行
            i += 1
            continue
        
        sections.append(element["content"])
        i = element["end_line"] + 1  # 跳到下一个元素
    
    return [section for section in sections if section.strip()]
```

------
2. 通过解析器 `markdown_parser` 解析出文本结构和表格内容。
```python
sections, tables = markdown_parser(filename, binary, separate_tables=False)
```

------
3. 获取 section 中的图片
```python
section_images = []
for idx, (section_text, _) in enumerate(sections):
    images = markdown_parser.get_pictures(section_text) if section_text else None
```

### markdown_parser.get_pictures

1. 获取图片链接

```python
image_urls = self.get_picture_urls(text)

def get_picture_urls(self, sections):
    ...
    from bs4 import BeautifulSoup
    # 将 markdown 文档转换成 html
    html_content = markdown(text)
    soup = BeautifulSoup(html_content, 'html.parser')
    # 提取图片链接
    html_images = [img.get('src') for img in soup.find_all('img') if img.get('src')]
    return html_images
```

2. 获取图片（远程 or 本地）
```python
if url.startswith(('http://', 'https://')):
    # 远程 URL：下载图片
    response = requests.get(url, stream=True, timeout=30)
    if response.status_code == 200 and response.headers['Content-Type'].startswith('image/'):
        img = Image.open(BytesIO(response.content)).convert('RGB')
        images.append(img)
else:
    # 本地文件路径：直接打开
    from pathlib import Path
    local_path = Path(url)
    if not local_path.exists():
        logging.warning(f"Local image file not found: {url}")
        continue
    img = Image.open(url).convert('RGB')
    images.append(img)
```

------
4. 合并同一个 section 中图片，使用视觉模型将图片解析成文本描述
```python
combined_image = reduce(concat_img, images) if len(images) > 1 else images[0]
section_images.append(combined_image)
markdown_vision_parser = VisionFigureParser(vision_model=vision_model, figures_data= [((combined_image, ["markdown image"]), [(0, 0, 0, 0, 0)])], **kwargs)
boosted_figures = markdown_vision_parser(callback=callback)
sections[idx] = (section_text + "\n\n" + "\n\n".join([fig[0][1] for fig in boosted_figures]), sections[idx][1])
```
 > *VisionFigureParser 的详细解析可参考《naive parser 语义切块（docx 篇）》中的 VisionFigureParser 类模块*

-------
5. 对文本和表格内容进行分词
```python
res = tokenize_table(tables, doc, is_english)
```
 > *tokenize_table 的详细解析可参考《navie 分词器原理》*

--------
6. sections 后处理
```python
chunks, images = naive_merge_with_images(sections, section_images,
                                int(parser_config.get(
                                    "chunk_token_num", 128)), parser_config.get(
                                    "delimiter", "\n!?。；！？"))
res.extend(tokenize_chunks_with_images(chunks, doc, is_english, images))
```
