---
title: "【解密源码】 RAGFlow 切分最佳实践- naive parser 语义切块（excel & csv & txt 篇）"
date: 2025-10-25T19:36:20+08:00
draft: false
tags: ["源码","技术","RAG"]
categories: ["RAGFlow"]
---

# 引言

在前两期深度解析中，我们见证了 RAGFlow 如何驯服文档处理领域的"两大巨兽"——结构复杂的 PDF 和格式丰富的 DOCX。从布局分析到表格识别，从视觉增强到语义关联，RAGFlow 展现出了处理复杂文档的卓越能力。

然而，在真实的企业知识库中，除了这些"重量级"文档外，还存在着大量"轻量级"但同样重要的数据源：结构化的 Excel 报表、纯净的 CSV 数据、简洁的文本文件，以及各种源代码文档。这些格式看似简单，却在数据处理中扮演着不可或缺的角色。

本期我们将聚焦这些结构化与半结构化数据的处理方案。与 PDF/DOCX 的复杂解析不同，Excel/CSV/TXT 文档的处理更注重数据完整性、格式保真和高效提取。RAGFlow 通过精巧的设计，为每种格式提供了最优的语义切块策略，确保从表格数据到代码文件的全面覆盖。

# 省流版

RAGFlow 为**结构化与半结构化文档**提供了精准高效的轻量级解析方案，针对不同格式特性采用最优处理策略：

### Excel & CSV - 结构化数据处理
**针对不同场景双模式输出。**
#### 键值对文本

统一格式解析。使用 openpyxl 库解析 excel 文件，结构化解析 csv 文件后转换成 openpyxl.Workbook 格式。

键值对文本输出。通过对 openpyxl.Workbook 格式输出数据进行解析组装后输出。
```python
# 输出：语义化的键值对文本
"姓名：张三; 部门：销售部; 销售额：150万 ——销售报表"
```
#### HTML 表格

统一格式解析后，将表格按照指定大小切分，添加 html 标签输出。
```python
# 输出：完整的HTML表格，保持原始格式
<table>
  <tr><th>姓名</th><th>部门</th><th>销售额</th></tr>
  <tr><td>张三</td><td>销售部</td><td>150万</td></tr>
</table>
```
**设计亮点**
- 统一格式处理：CSV→Excel格式转换，统一处理流水线；
- 分块优化：大表格按行分块，避免token超限。


### TXT & 代码文档 - 半结构化文本处理
识别输入文件编码格式，解码后按照规则进行切分

**设计亮点**
- **多编码自适应**：chardet智能编码检测，完美处理中文；
- **语义边界识别**：基于标点和逻辑结构的分块。

# 手撕版
## 1. Excel & CSV 文档

根据解析配置，将文档按照 html 方案和默认方案进行切块。
```python
elif re.search(r"\.(csv|xlsx?)$", filename, re.IGNORECASE):
    callback(0.1, "Start to parse.")
    excel_parser = ExcelParser()
    if parser_config.get("html4excel"):
        sections = [(_, "") for _ in excel_parser.html(binary, 12) if _]
    else:
        sections = [(_, "") for _ in excel_parser(binary) if _]
    parser_config["chunk_token_num"] = 12800
```
### 1.1 ExcelParser 类

**_load_excel_to_workbook**：判断输入文件类型 excel 或 csv，进行相应处理。

**_clean_dataframe**：清理掉 Excel 中非法控制字符（\x00 - \x1F）。

**_dataframe_to_workbook**：统一输出格式，将 csv 文件内容转换成 openpyxl.Workbook 格式，与 excel 文件处理后格式统一，便于后续处理。

**html**：将输入的 excel 或 csv 文件内容转换成 html 表格字符串，并按照指定的 chunk_row 对行进行分块输出。例如 chunk_row 指定 12，则输入的表格会被切分为多个 12 行的小表格输出。

#### 1.1.1 _load_excel_to_workbook

判断输入文件是 excel 还是 csv
- excel 文件使用 python 库 openpyxl 直接打开处理；
- csv 文件使用 _dataframe_to_workbook 方法处理。

```python
if not (file_head.startswith(b"PK\x03\x04") or file_head.startswith(b"\xd0\xcf\x11\xe0")):
    logging.info("Not an Excel file, converting CSV to Excel Workbook")

    try:
        file_like_object.seek(0)
        df = pd.read_csv(file_like_object)
        return RAGFlowExcelParser._dataframe_to_workbook(df)
try:
    return load_workbook(file_like_object, data_only=True)
```

> *Tips：判断 Excel 文件，因为 _load_excel_to_workbook 接收的参数是内存流（BytesIO）格式，无文件名，所以需要通过判断文件流的头 4 个字符*

> file_head.startswith(b"PK\x03\x04"）：b"PK\x03\x04" 是 ZIP 文件头 的标志，.xlsx（Office Open XML）实际上是一个 ZIP 容器。

> file_head.startswith(b"\xd0\xcf\x11\xe0")"：b"\xd0\xcf\x11\xe0"（十六进制 D0 CF 11 E0）对应 OLE Compound File（Compound File Binary Format），这是传统的 .xls（BIFF）和早期 Office 二进制文件的标识。

#### 1.1.2 ExcelParser实例化

使用解析后的表格数据，遍历**每行内容**拼接成文本作为文本块输出。
```python
wb = RAGFlowExcelParser._load_excel_to_workbook(file_like_object)
res = []
for sheetname in wb.sheetnames:
    ws = wb[sheetname]
    rows = list(ws.rows)
    if not rows:
        continue
    ti = list(rows[0])
    for r in list(rows[1:]):
        fields = []
        for i, c in enumerate(r):
            if not c.value:
                continue
            t = str(ti[i].value) if i < len(ti) else ""
            t += ("：" if t else "") + str(c.value)
            fields.append(t)
        line = "; ".join(fields)
        if sheetname.lower().find("sheet") < 0:
            line += " ——" + sheetname
        res.append(line)
return res
```

## 2. Txt 以及其他代码文档
```python
elif re.search(r"\.(txt|py|js|java|c|cpp|h|php|go|ts|sh|cs|kt|sql)$", filename, re.IGNORECASE):
    callback(0.1, "Start to parse.")
    sections = TxtParser()(filename, binary,
                            parser_config.get("chunk_token_num", 128),
                            parser_config.get("delimiter", "\n!?;。；！？"))
    callback(0.8, "Finish parsing.")
```

### 2.1 TxtParser

**parser_txt**：按照传入的分隔符和 chunk size，切分文档输出。

这里简单介绍下 `TxtParser()` 实例化中获取文档内容部分 `get_text` 实现。
```python
def __call__(self, fnm, binary=None, chunk_token_num=128, delimiter="\n!?;。；！？"):
    txt = get_text(fnm, binary)
    return self.parser_txt(txt, chunk_token_num, delimiter)
```

#### 2.1.1 get_text

如果传入的二进制内容，则使用从 rag.nlp 引入的方式自动推断出正确的编码，进行解码；否则直接从文件路径读取文本进行拼接返回。

```python
from rag.nlp import find_codec
def get_text(fnm: str, binary=None) -> str:
    txt = ""
    if binary:
        encoding = find_codec(binary)
        txt = binary.decode(encoding, errors="ignore")
    else:
        with open(fnm, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                txt += line
    return txt
```
> *find_codec 主要在不确定文件编码的情况下，使用 chardet 库自动推断出正确的编码方式，以便后续正确解码为 UTF-8 字符串。*

经过切分得到 sections 后，excel，csv，txt及其它代码文档的 sections 后处理可参考《naive parser 语义切块（pdf 篇）》中 **sections 后处理模块中的无图 sections 处理逻辑**，经过后处理后得到最终输出的 res。

# 下期预告
在本期《【解密源码】RAGFlow 切分最佳实践- naive parser 语义切块（excel & csv & txt 篇）》中，我们深入剖析了 Excel 等文档在 RAGFlow 中的完整解析流水线，相比较 pdf 和 docx 文档，这类文档的处理较为简单直接。

在下一期中，我们将深入剖析 naive parser 下 .md 文件的语义切块方案。