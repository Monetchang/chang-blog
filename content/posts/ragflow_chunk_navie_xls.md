---
title: "【解密源码】 RAGFlow 切分最佳实践- naive Parser 语义切块（excel & csv & txt 篇）"
date: 2025-10-25T19:36:20+08:00
draft: false
tags: ["源码","技术","RAG"]
categories: ["RAGFlow"]
---

# 引言

在上一期《naive Parser 语义切块（docx 篇）》中，我们深入剖析了 RAGFlow 如何处理结构丰富的 .docx 文档，从基础解析到视觉增强，展现了完整的文档理解流水线。我们看到了智能表格识别、标题链构建、多模态融合等核心技术如何协同工作，将格式化的办公文档转化为高质量的语义块。

本期我们将挑战文档处理领域的"终极BOSS"—— pdf 格式。作为最复杂、最通用的文档格式，pdf 兼具了扫描件与数字文档的双重特性，版面布局千变万化，文字提取难度极高。RAGFlow 通过三重解析引擎的智能协作，为不同特性的 pdf 文档提供了针对性的解决方案。

# 省流版

# 手撕版
## Excel & csv 文档

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
### ExcelParser 类

**_load_excel_to_workbook**：判断输入文件类型 excel 或 csv，进行相应处理。

**_clean_dataframe**：清理掉 Excel 中非法控制字符（\x00 - \x1F）。

**_dataframe_to_workbook**：统一输出格式，将 csv 文件内容转换成 openpyxl.Workbook 格式，与 excel 文件处理后格式统一，便于后续处理。

**html**：将输入的 excel 或 csv 文件内容转换成 html 表格字符串，并按照指定的 chunk_row 对行进行分块输出。例如 chunk_row 指定 12，则输入的表格会被切分为多个 12 行的小表格输出。

#### _load_excel_to_workbook

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

#### ExcelParser实例化

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

## Txt 以及其他代码文档
```python
elif re.search(r"\.(txt|py|js|java|c|cpp|h|php|go|ts|sh|cs|kt|sql)$", filename, re.IGNORECASE):
    callback(0.1, "Start to parse.")
    sections = TxtParser()(filename, binary,
                            parser_config.get("chunk_token_num", 128),
                            parser_config.get("delimiter", "\n!?;。；！？"))
    callback(0.8, "Finish parsing.")
```

### TxtParser

**parser_txt**：按照传入的分隔符和 chunk size，切分文档输出。

这里简单介绍下 `TxtParser()` 实例化中获取文档内容部分 `get_text` 实现。
```python
def __call__(self, fnm, binary=None, chunk_token_num=128, delimiter="\n!?;。；！？"):
    txt = get_text(fnm, binary)
    return self.parser_txt(txt, chunk_token_num, delimiter)
```

#### get_text

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

经过切分得到 sections 后，excel，csv，txt及其它代码文档的 sections 后处理可参考《naive Parser 语义切块（pdf 篇）》中 **sections 后处理模块中的无图 sections 处理逻辑**，经过后处理后得到最终输出的 res。

# 下期预告
在本期《【解密源码】RAGFlow 切分最佳实践- naive Parser 语义切块（excel & csv & txt 篇）》中，我们深入剖析了 Excel 等文档在 RAGFlow 中的完整解析流水线，相比较 pdf 和 docx 文档，这类文档的处理较为简单直接。

在下一期中，我们将深入剖析 Naive Parser 下 .md 文件的语义切块方案。