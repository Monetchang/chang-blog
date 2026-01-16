---
title: "【解密源码】WeKnora 文档切分与 Chunk 构建解析：腾讯生产级 RAG 的底层设计 "
date: 2026-01-16T17:06:10+08:00
draft: false
tags: ["源码","技术","RAG"]
categories: ["RAG"]
---

# 引言
WeKnora 是腾讯开源的一套**生产级 RAG 框架**，定位非常明确：解决真实业务场景下“文档复杂、类型多样、规模可控但质量要求极高”的知识增强问题。社区中有人将其视为 *ima* 的开源实现之一，虽然这一说法无从官方考证，但可以确定的是，WeKnora 在工程完整度、边界处理和异常降级策略上，是一套经过实战打磨的系统方案。

从文档接入、解析、切分、向量化、多模态增强，到知识图谱、问题生成与摘要生成，WeKnora 几乎覆盖了一个完整 RAG 系统在生产环境中可能遇到的所有关键问题。尤其是在**文档解析与 Chunk 构建**这一最容易被低估、却最影响检索与生成质量的环节，WeKnora 给出了一套相当成熟且可复用的设计。

本文将聚焦 WeKnora 的**文档接入与解析体系**，从文件/URL/手动创建三种入口开始，深入拆解其解析器架构、Markdown 统一中间表示、语义切分策略、多模态处理，以及最终如何构建可用于检索与推理的 Chunk 数据结构。

# 上传方式
## 文件上传
上传文件 → 计算Hash去重 → 存储文件 → 异步解析
```go
// Service 层 - 核心逻辑
func (s *knowledgeService) CreateKnowledgeFromFile(ctx context.Context, kbID string, 
    file *multipart.FileHeader, metadata map[string]string, enableMultimodel *bool, customFileName string,
) (*types.Knowledge, error) {
    // 1. 计算文件 Hash（去重检测）
    fileContent, _ := file.Open()
    fileHash := calculateFileHash(fileContent)
    
    // 2. 检查是否已存在相同文件
    existing, _ := s.repo.FindByFileHash(ctx, kbID, fileHash)
    if existing != nil {
        return existing, &types.DuplicateKnowledgeError{
            ExistingID: existing.ID,
            Message:    "相同文件已存在",
        }
    }
    
    // 3. 上传文件到存储服务（MinIO/COS）
    filePath, _ := s.storage.Upload(fileContent, fileName)
    
    // 4. 创建 Knowledge 记录
    knowledge := &types.Knowledge{
        ID:          uuid.New().String(),
        Type:        "file",
        FileName:    fileName,
        FileType:    getFileType(fileName),  // pdf, docx, xlsx...
        FileHash:    fileHash,
        FilePath:    filePath,
        FileSize:    file.Size,
        ParseStatus: types.ParseStatusPending,  // 等待异步处理
        Metadata:    metadata,
    }
    s.repo.CreateKnowledge(ctx, knowledge)
    
    // 5. 入队异步解析任务
    payload := types.DocumentProcessPayload{
        KnowledgeID:     knowledge.ID,
        EnableMultimodel: enableMultimodel,
    }
    task := asynq.NewTask(types.TypeDocumentProcess, payload)
    s.task.Enqueue(task)
    
    return knowledge, nil
}
```
## URL 创建
接收URL → URL去重 → 异步抓取解析
```go
// Service 层 - 核心逻辑
func (s *knowledgeService) CreateKnowledgeFromURL(ctx context.Context, 
    kbID, url string, enableMultimodel *bool, title string,
) (*types.Knowledge, error) {
    // 1. 验证 URL 格式
    if !isValidURL(url) {
        return nil, errors.NewBadRequestError("无效的URL格式")
    }
    
    // 2. 检查是否已存在相同 URL
    existing, _ := s.repo.FindBySourceURL(ctx, kbID, url)
    if existing != nil {
        return existing, &types.DuplicateKnowledgeError{
            ExistingID: existing.ID,
            Message:    "相同URL已存在",
        }
    }
    
    // 3. 创建 Knowledge 记录
    knowledge := &types.Knowledge{
        ID:          uuid.New().String(),
        Type:        "url",
        SourceURL:   url,
        Title:       title,
        FileName:    extractFileNameFromURL(url),  // 从 URL 提取文件名
        FileType:    "html",
        ParseStatus: types.ParseStatusPending,  // 等待异步处理
    }
    s.repo.CreateKnowledge(ctx, knowledge)
    
    // 4. 入队异步抓取任务
    payload := types.DocumentProcessPayload{
        KnowledgeID:      knowledge.ID,
        SourceURL:        url,  // 传递 URL 供异步任务抓取
        EnableMultimodel: enableMultimodel,
    }
    task := asynq.NewTask(types.TypeDocumentProcess, payload)
    s.task.Enqueue(task)
    
    return knowledge, nil
}
```
## 手动创建
接收Markdown内容 → 无去重 → 同步处理
```go
// Service 层 - 核心逻辑（同步处理）
func (s *knowledgeService) CreateKnowledgeFromManual(ctx context.Context, 
    kbID string, req *types.ManualKnowledgePayload,
) (*types.Knowledge, error) {
    // 1. 创建 Knowledge 记录
    knowledge := &types.Knowledge{
        ID:          uuid.New().String(),
        Type:        "manual",
        Title:       req.Title,
        FileName:    req.Title + ".md",
        FileType:    "md",
        ParseStatus: types.ParseStatusProcessing,  // 直接开始处理
    }
    s.repo.CreateKnowledge(ctx, knowledge)
    
    // 2. 同步解析 Markdown 内容（无需异步任务）
    chunks := s.parseMarkdown(req.Content)
    
    // 3. 同步处理 Chunks（生成 embedding、索引）
    s.processChunks(ctx, kb, knowledge, chunks)
    
    // 4. 更新状态为完成
    knowledge.ParseStatus = types.ParseStatusCompleted
    s.repo.UpdateKnowledge(ctx, knowledge)
    
    return knowledge, nil
}
```
# 解析模式
## FirstParser - 链式尝试模式
尝试多个解析器，直到第一个成功
```python
class FirstParser(BaseParser):
    _parser_cls: Tuple[Type["BaseParser"], ...] = ()
    
    def parse_into_text(self, content: bytes) -> Document:
        """顺序尝试每个解析器"""
        for p in self._parsers:
            try:
                document = p.parse_into_text(content)
                if document.is_valid():
                    return document  # ← 第一个成功就返回
            except Exception:
                continue  # ← 失败就继续下一个
        return Document()  # ← 都失败就返回空
```
## PipelineParser - 管道链式模式
多个解析器串联处理，前一个的输出是后一个的输入
```python
class PipelineParser(BaseParser):
    _parser_cls: Tuple[Type["BaseParser"], ...] = ()
    
    def parse_into_text(self, content: bytes) -> Document:
        """依次调用每个解析器，累积图片"""
        images: Dict[str, str] = {}
        document = Document()
        
        for p in self._parsers:
            document = p.parse_into_text(content)
            content = endecode.encode_bytes(document.content)  # ← 转换为下一个解析器的输入
            images.update(document.images)  # ← 累积图片
        
        document.images.update(images)
        return document
```
# 文档解析
## pdf 文档处理
```python
# 尝试顺序：
# 1. MinerUParser (主解析器)
# 2. MarkitdownParser (备选解析器)
class PDFParser(FirstParser):
    _parser_cls = (MinerUParser, MarkitdownParser)
# 管道流程：
# PDF → StdMinerUParser (调用 MinerU API)
#     → MarkdownTableFormatter (格式化表格)
#     → 最终 Document
class MinerUParser(PipelineParser):
    _parser_cls = (StdMinerUParser, MarkdownTableFormatter)
    

class MarkitdownParser(PipelineParser):
    """
    使用管道处理模式的 Markdown 解析器
    
    数据流：
    PDF 字节流
      ↓
    StdMarkitdownParser (第一阶段)
      ├─ 调用 MarkItDown 库解析
      ├─ 返回 Markdown 文本 + 内嵌数据 URI
      └─ Document(content, images={})
      ↓
    MarkdownParser (第二阶段)
      ├─ 处理 Markdown 内容
      ├─ 提取数据 URI 中的图片
      ├─ 上传到存储
      └─ 返回最终 Document(content, images)
    """
    
    _parser_cls = (StdMarkitdownParser, MarkdownParser)
    # 两个解析器按顺序管道处理
```
优先使用 MinerU 将文件转换成 markdown 格式，**这里需要用户自行配置 MinerU token 才能正常使用。**
降级采用微软开源框架 MarkItDown，将 pdf 转换成 markdown 格式，**MarkItDown 插件处理不了 pdf 扫描件。**
转换成 markdown 之后，对 markdown 进行初步格式化处理：
1. 对 markdown 文档的表格空行进行规范化处理，标准化列对齐
2. 将 markdown 文档中的 base64 图片转换为二进制上传到 COS，将 COS 链接替换 markdown 文档中的 base64 图片路径

## docx 文档处理
```python
# 尝试顺序：
# 1. MarkitdownParser (主解析器，与 pdf 备选方案一致)
# 2. DocxParser (备选解析器)
class Docx2Parser(FirstParser):
    _parser_cls = (MarkitdownParser, DocxParser) 
```
**降级策略 DocxParser 采用 python-docx**，最终输出 markdown 格式的文本内容。处理流程如下：
1. 通过 python-docx 库解析 docx 文件，提取纯文本内容，表格和图片。
2. 若开启了图片处理 enable_multimodal，则对提取的图片进行过滤，缩放等处理后上传至 COS，再将 COS 链接替换 markdown 文档中的图片路径
```python
# 过滤小图片（装饰元素）
if image.width < 50 or image.height < 50:
    return None

# 缩放大图片
if image.width > max_image_size:
    image = image.resize((new_width, new_height))
```
3. 将表格转换为 HTML 格式
```python
def _convert_table_to_html(self, table):
    html = "<table>"
    for r in table.rows:
        html += "<tr>"
        for c in r.cells:
            # 处理合并单元格
            html += f"<td colspan='{span}'>{c.text}</td>"
        html += "</tr>"
    html += "</table>"
    return html
```
4. 保持原文内容顺序进行 markdown 格式文档输出。
```python
@dataclass
class DocumentModel:
    content: str           # Markdown 格式文本（含图片链接）
    images: Dict[str, str] # {url: base64_data}
```

## doc 文档处理
doc 文档处理优先使用 _parse_with_docx 方法，**采用 LibreOffice 库将 .doc 文件转换为 .docx 格式**，再按照 Docx2Parser 流程解析文件。

若 _parse_with_docx 方法失败，将会使用降级方案 _parse_with_antiword，**采用 antiword 库直接提取 .doc 文本内容**。antiword 只能提取纯文本，不支持图片和表格的提取。

```python
class DocParser(Docx2Parser):
    """DOC document parser"""
    def parse_into_text(self, content: bytes) -> Document:
        logger.info(f"Parsing DOC document, content size: {len(content)} bytes")

        handle_chain = [
            # 1. Try to convert to docx format to extract images
            self._parse_with_docx,
            # 2. If image extraction is not needed or conversion failed,
            # try using antiword to extract text
            self._parse_with_antiword,
            # 3. If antiword extraction fails, use textract
            # NOTE: _parse_with_textract is disabled due to SSRF vulnerability
            # self._parse_with_textract,
        ]
```
## csv 文档处理
使用 pandas 库读取 csv 文件，将其转换为 DataFrame 格式，再将 DataFrame 转换为文本内容。会默认将 DataFrame 的第一行作为表头，将其他行作为数据行进行组合。
```python
 # Read CSV content into a pandas DataFrame, skipping malformed lines
df = pd.read_csv(BytesIO(content), on_bad_lines="skip")

# Process each row in the DataFrame
for i, (idx, row) in enumerate(df.iterrows()):
    # Format row as "column: value" pairs separated by commas
    content_row = (
        ",".join(
            f"{col.strip()}: {str(row[col]).strip()}" for col in df.columns
        )
        + "\n"
    )
    # Update end position for this chunk
    end += len(content_row)
    text.append(content_row)
    
    # Create a chunk for this row with position tracking
    chunks.append(Chunk(content=content_row, seq=i, start=start, end=end))
    # Update start position for next chunk
    start = end

return Document(
    content="".join(text),
    chunks=chunks,
)
```
最终输出格式示例如下：
```python
Document(
    content="姓名: 张三,年龄: 25,城市: 北京\n姓名: 李四,年龄: 30,城市: 上海\n...",
    chunks=[
        Chunk(content="姓名: 张三,年龄: 25,城市: 北京\n", seq=0, start=0, end=25),
        Chunk(content="姓名: 李四,年龄: 30,城市: 上海\n", seq=1, start=25, end=50),
        Chunk(content="姓名: 王五,年龄: 28,城市: 深圳\n", seq=2, start=50, end=75),
    ]
)
```
*Tips: 这种默认第一行为表头的策略能够最大程度的保持 csv 文档的结构。但对于第一行不是表头的情况，也会有混乱语义的危险。*

## excel 文档处理
使用 openpyxl 库读取 excel 文件，将每个工作表转换为文本内容。与 csv 处理策略相同，会默认将每个工作表的第一行作为表头，将其他行作为数据行进行组合。
```python
# Load Excel file from bytes into pandas ExcelFile object
excel_file = pd.ExcelFile(BytesIO(content))

# Process each sheet in the Excel file
for excel_sheet_name in excel_file.sheet_names:
    # Parse the sheet into a DataFrame
    df = excel_file.parse(sheet_name=excel_sheet_name)
    # Remove rows where all values are NaN (completely empty rows)
    df.dropna(how="all", inplace=True)

    # Process each row in the DataFrame
    for _, row in df.iterrows():
        page_content = []
        # Build key-value pairs for non-null values
        for k, v in row.items():
            if pd.notna(v):  # Skip NaN/null values
                page_content.append(f"{k}: {v}")
        
        # Skip rows with no valid content
        if not page_content:
            continue
        
        # Format row as comma-separated key-value pairs
        content_row = ",".join(page_content) + "\n"
        end += len(content_row)
        text.append(content_row)
        
        # Create a chunk for this row with position tracking
        chunks.append(
            Chunk(content=content_row, seq=len(chunks), start=start, end=end)
        )
        start = end

# Combine all text and return as Document
return Document(content="".join(text), chunks=chunks)
```
*Tips: pandas 处理 excel 文档，如果遇见多级表头，多个合并单元格的复杂文档，会丢失结构信息，导致语义混乱。*

## 图片处理
图片处理很简单，上传图片到存储服务, 将返回的图片 URL 转换为 Markdown 图片语法，并生成 base64 编码。
```python
def parse_into_text(self, content: bytes) -> Document:
    logger.info(f"Parsing image content, size: {len(content)} bytes")

    # Get file extension
    ext = os.path.splitext(self.file_name)[1].lower()

    # Upload image to storage
    image_url = self.storage.upload_bytes(content, file_ext=ext)
    logger.info(f"Successfully uploaded image, URL: {image_url[:50]}...")

    # Generate markdown text
    text = f"![{self.file_name}]({image_url})"
    images = {image_url: base64.b64encode(content).decode()}

    # Create image object and add to map
    return Document(content=text, images=images)
```

## 网页文档处理
```python
# 管道流程：
# URL → StdWebParser (获取网页内容)
#     → MarkdownParser (格式化 Markdown 内容)
#     → 最终 Document
class WebParser(PipelineParser):
    _parser_cls = (StdWebParser, MarkdownParser)
```
### 使用 Playwright 抓取页面内容
```python
async def scrape(self, url: str) -> str:
    async with async_playwright() as p:
        # 配置代理（可选）
        kwargs = {}
        if self.proxy:
            kwargs["proxy"] = {"server": self.proxy}
        
        # 启动 WebKit 浏览器
        browser = await p.webkit.launch(**kwargs)
        page = await browser.new_page()
        
        # 访问页面，30秒超时
        await page.goto(url, timeout=30000)
        
        # 获取完整 HTML
        content = await page.content()
        
        await browser.close()
        return content
```
### 使用 Trafilatura 提取正文
```python
def parse_into_text(self, content: bytes) -> Document:
    url = endecode.decode_bytes(content)
    
    # 抓取 HTML
    chtml = asyncio.run(self.scrape(url))
    
    # 提取正文，转为 Markdown
    md_text = extract(
        chtml,
        output_format="markdown",  # 输出格式
        with_metadata=True,        # 保留元数据
        include_images=True,       # 保留图片
        include_tables=True,       # 保留表格
        include_links=True,        # 保留链接
    )
    
    return Document(content=md_text)
```
通过 StdWebParser 解析器，输出 markdown 文档，包含元数据（如标题、作者、发布日期等）和图片、表格、链接等。进入后续 MarkdownParser 解析器，将 markdown 文档转换为 Document 格式。

## Markdown 文档处理（核心）
**以下是对文档处理的核心方案**，因为以上所有文档类型都是先转换成对应的 Markdown 文档结构，再进行统一处理。
### 核心架构

```
原始 Markdown 文本
  ↓
TextSplitter.split_text(text)
  ├─ Step 1: _split(text)
  │  └─ 递归分割，保证每个 split ≤ chunk_size
  │
  ├─ Step 2: _split_protected(text)
  │  └─ 提取所有受保护内容的位置和范围
  │
  ├─ Step 3: _join(splits, protect)
  │  └─ 合并 splits 和 protected，保证受保护内容完整
  │
  └─ Step 4: _merge(splits)
     ├─ 合并 splits 成为最终的 chunks
     ├─ 处理 overlap（重叠）
     └─ 返回 List[Tuple[start, end, text]]

返回：List[(start_pos, end_pos, chunk_text), ...]
```
### 主函数 split_text()
```python
def split_text(self, text: str) -> List[Tuple[int, int, str]]:
    if text == "":
        return []

    # Step 1: Split text by separators recursively
    splits = self._split(text)
    # Step 2: Extract protected content positions
    protect = self._split_protected(text)
    # Step 3: Merge splits with protected content to ensure integrity
    splits = self._join(splits, protect)

    # Verify that joining all splits reconstructs the original text
    assert "".join(splits) == text

    # Step 4: Merge splits into final chunks with overlap
    chunks = self._merge(splits)
    return chunks
```
### 关键文本保护
受保护的 Regex 模式（6 种），分别是**LaTeX 数学公式，Markdown 图片链接，Markdown 普通链接，Markdown 表格表头（Header + 分隔行），Markdown 表格内容，代码块头（带语言标识）**，对于这些场景不进行截断
```python
protected_regex: List[str] = [
    # math formula - LaTeX style formulas enclosed in $$
    r"\$\$[\s\S]*?\$\$",
    # image - Markdown image syntax ![alt](url)
    r"!\[.*?\]\(.*?\)",
    # link - Markdown link syntax [text](url)
    r"\[.*?\]\(.*?\)",
    # table header - Markdown table header with separator line
    r"(?:\|[^|\n]*)+\|[\r\n]+\s*(?:\|\s*:?-{3,}:?\s*)+\|[\r\n]+",
    # table body - Markdown table rows
    r"(?:\|[^|\n]*)+\|[\r\n]+",
    # code header - Code block start with language identifier
    r"```(?:\w+)[\r\n]+[^\r\n]*",
],
```
### 语义递归粗切 -- _split()

```python
# 语义规则
self._split_fns = [
    split_by_sep("\n"),      # [1] 换行符（优先级最高）
    split_by_sep("。"),      # [2] 句号
    split_by_sep(" "),       # [3] 空格
    split_by_char()          # [最后] 逐字分割（fallback）
]
```

```python
def _split(self, text: str) -> List[str]:
    """
    递归分割，确保每个 split ≤ chunk_size
    """
    
    # 基础情况：文本已足够小
    if len(text) <= chunk_size:
        return [text]
    
    # 递归情况：使用分隔符分割
    splits = []
    
    # 按优先级尝试各个分隔符
    for split_fn in self._split_fns:  # 5 个函数
        splits = split_fn(text)
        # 如果这个分隔符能分割出多个部分，就使用它
        if len(splits) > 1:
            break
    
    # 如果上面的都没有分割成功，最后用 split_by_char() 逐字分割
    # （这通常不会发生，除非整个文本没有任何分隔符）
    
    new_splits = []
    for split in splits:
        if len(split) <= chunk_size:
            new_splits.append(split)
        else:
            # 递归处理超大的分割块
            new_splits.extend(self._split(split))
    
    return new_splits
```
```
假设：chunk_size = 200，文本长度 = 1000

第1次尝试：用 "\n" 分割
  ├─ 如果能分割成多个部分，检查是否都 ≤ 200
  ├─ 如果有部分 > 200，对这些部分递归调用 _split()
  └─ 递归中再用 "\n"，如果还是有 > 200 的
      └─ 换成 "。" 试试
      
第2次尝试：用 "。" 分割
  ├─ 可能会分割得更细
  └─ 继续检查

第3次尝试：用 " " (空格) 分割
  ├─ 分割得更细

最后：split_by_char()
  ├─ 按单个字符分割
  └─ 确保没有分割块 > chunk_size
```

### 保护文本提取 -- _split_protected()
```python
def _split_protected(self, text: str) -> List[Tuple[int, str]]:
    """
    扫描所有 protected_regex，找到所有受保护内容
    返回：[(start_pos, protected_text), ...]
    """
    
    # 步骤1: 使用所有 protected 模式进行匹配
    matches = [
        (match.start(), match.end())
        for pattern in self._protected_fns  # 6 个 regex pattern
        for match in pattern.finditer(text)
    ]
    
    # 步骤2: 按开始位置排序，处理重叠
    matches.sort(key=lambda x: (x[0], -x[1]))
    # 排序规则：
    # - 按 start_pos 升序（从前到后）
    # - 如果 start_pos 相同，按 length 降序（长的优先）
    
    # 步骤3: 使用 accumulate 过滤重叠的匹配
    res = []
    initial = -1  # 上一个匹配的结束位置
    
    for current_start, current_end in matches:
        # 只处理不与前面匹配重叠的内容
        if current_start >= initial:
            # 只保留 < chunk_size 的受保护内容
            if current_end - current_start < chunk_size:
                res.append((current_start, text[current_start:current_end]))
            else:
                logger.warning(f"Protected text ignore: {text[current_start:current_end]}")
                # 如果受保护内容本身 > chunk_size，忽略（记录警告）
        
        # 更新 initial 为这个匹配的结束位置
        initial = max(initial, current_end)
    
    return res
```

```
1. 表格匹配 (protected_regex[4] + [5]):
   ├─ start=20, end=80
   ├─ text="| 列1 | 列2 |\n| :--- | ---: |\n| 数据1 | 数据2 |"
   └─ length=60 < chunk_size ✅ 保留

2. 数学公式匹配 (protected_regex[0]):
   ├─ start=100, end=120
   ├─ text="$$f(x) = x^2$$"
   └─ length=14 < chunk_size ✅ 保留

返回：[
    (20, "| 列1 | 列2 |\n| :--- | ---: |\n| 数据1 | 数据2 |"),
    (100, "$$f(x) = x^2$$")
]
```

### chunk 和保护文本合并 -- _join()
_join() 的目标是**重新整理 chunk list，通过位置计算，确保受保护内容（如完整表格）不会被分割**。如果原始 chunk 中的表格被断开了，它会合并这些断开的部分并插入完整的受保护内容。
```python
def _join(self, splits: List[str], protect: List[Tuple[int, str]]) -> List[str]:
    """
    关键目标：确保受保护内容保持完整
    
    问题场景：
    假设 _split() 返回：
      splits = [
        "普通文本 ABC。\n\n",
        "| 列1 | 列2 |\n|",      ← 表格被分割了！（不好）
        " :--- | ---: |\n| 数据1 | 数据2 |"
      ]
    
    但 protect 告诉我们：
      protect = [(20, "| 列1 | 列2 |\n| :--- | ---: |\n| 数据1 | 数据2 |")]
    
    _join() 的作用：重新组织 splits，使得表格作为一个完整单元出现
    """
    
    j = 0  # 受保护内容的索引
    point = 0  # 当前在原文中的位置
    start = 0  # 当前 split 的起始位置
    res = []  # 结果列表
    
    for split in splits:
        # 计算当前 split 在原文中的范围
        end = start + len(split)
        
        # 从 point 开始提取当前 split 的子串
        cur = split[point - start:]
        
        # 处理所有与当前 split 重叠的受保护内容
        while j < len(protect):
            p_start, p_content = protect[j]
            p_end = p_start + len(p_content)
            
            # 如果受保护内容在当前 split 之后，停止
            if end <= p_start:
                break
            
            # 添加受保护内容之前的部分
            if point < p_start:
                local_end = p_start - point
                res.append(cur[:local_end])
                cur = cur[local_end:]
                point = p_start
            
            # 添加整个受保护内容（作为一个完整的单元）
            res.append(p_content)
            j += 1
            
            # 跳过原 split 中受保护内容的部分
            if point < p_end:
                local_start = p_end - point
                cur = cur[local_start:]
                point = p_end
            
            # 如果当前 split 已处理完，跳出
            if not cur:
                break
        
        # 添加当前 split 的剩余部分
        if cur:
            res.append(cur)
            point = end
        
        # 移到下一个 split
        start = end
    
    return res
```
_join() 简述就是：
1. 遍历每个 chunk
2. 当遇到需要保护的内容 protect 时，移除 chunk 中的该内容
3. 插入完整的保护内容
4. 继续处理剩余的 chunk

### 合并最终 Chunks -- _merge()
处理 chunk 的 overlap 部分和解析处理 chunk 对应的多级 header 信息，构建最终的 chunk 内容。
```python
def _merge(self, splits: List[str]) -> List[Tuple[int, int, str]]:
    """
    合并 splits 成为最终的 chunks
    处理 overlap 和 header 追踪
    """
    
    chunks: List[Tuple[int, int, str]] = []
    cur_chunk: List[Tuple[int, int, str]] = []  # 当前 chunk 中的元素
    
    cur_headers = ""
    cur_len = 0
    cur_start, cur_end = 0, 0
    
    for split in splits:
        # 计算 split 在原文中的位置
        cur_end = cur_start + len(split)
        split_len = len(split)
        
        # 更新 header 信息
        self.header_hook.update(split)
        cur_headers = self.header_hook.get_headers()
        cur_headers_len = len(cur_headers)
        
        # 检查是否超过 chunk_size
        if cur_len + split_len + cur_headers_len > chunk_size:
            # 当前 chunk 已满，保存它
            if len(cur_chunk) > 0:
                chunks.append((
                    cur_chunk[0][0],              # 第一个元素的 start
                    cur_chunk[-1][1],             # 最后一个元素的 end
                    "".join([c[2] for c in cur_chunk])
                ))
            
            # 处理 overlap：从当前 chunk 的末尾向前选择
            while cur_chunk and (
                cur_len > chunk_overlap
                or cur_len + split_len + cur_headers_len > chunk_size
            ):
                # 移除第一个元素
                first = cur_chunk.pop(0)
                cur_len -= len(first[2])
            
            # 添加 headers 到新 chunk（如果有）
            if (
                cur_headers
                and split_len + cur_headers_len < chunk_size
                and cur_headers not in split
            ):
                next_start = cur_chunk[0][0] if cur_chunk else cur_start
                cur_chunk.insert(0, (0, 0, cur_headers))
                cur_len += cur_headers_len
        
        # 添加当前 split 到 chunk
        cur_chunk.append((cur_start, cur_end, split))
        cur_len += split_len
        cur_start = cur_end
    
    # 处理最后一个 chunk
    if cur_chunk:
        chunks.append((
            cur_chunk[0][0],
            cur_chunk[-1][1],
            "".join([c[2] for c in cur_chunk])
        ))
    
    return chunks
```
这里重点说下解析处理 chunk 对应的多级 header 信息
#### HeaderTracker
1. 解析标题 -- update(split)
```python

def update(self, split: str):
    """
    解析 split 中的所有 Markdown 标题
    """
    # 正则匹配：# 标题、## 子标题等
    for match in re.finditer(r"^(#+)\s+(.+)$", split, re.MULTILINE):
        level = len(match.group(1))  # # 的个数（1=一级，2=二级，...）
        title = match.group(2)       # 标题文本
        
        # 更新当前的标题栈
        # 例：读到 ## 时，更新二级标题
```
 ```
split = "## 第二季度\n\nQ2 销售额..."

匹配：
  level = 2  (因为是 ##)
  title = "第二季度"

更新结果：
  headers = {
    1: "2024 销售报告",      # 一级标题（前面设定的）
    2: "第二季度"            # 二级标题（刚更新的）
  }
```
2. 获取标题 -- get_headers()
```python
def get_headers(self) -> str:
    """
    返回当前的所有标题，按层级格式化
    """
    # 返回的格式类似：
    # # 2024 销售报告
    # ## 第二季度
```

**例子：**

```
当前标题栈：
{
    1: "2024 销售报告",
    2: "第二季度"
}

get_headers() 返回：
"# 2024 销售报告\n## 第二季度"
```

3. HeaderTracker 在 _merge() 中的使用，在处理每个 chunk 时，更新标题信息并检查是否需要创建新的 chunk。

```python
for split in splits:
    cur_end = cur_start + len(split)
    split_len = self.len_function(split)
    
    # 【1】更新标题信息
    self.header_hook.update(split)          # 解析当前 split 中的标题
    cur_headers = self.header_hook.get_headers()  # 获取当前标题链
    cur_headers_len = self.len_function(cur_headers)
    
    # 【2】检查 headers 是否太大
    if cur_headers_len > self.chunk_size:
        logger.error(f"Got headers of size {cur_headers_len}, ...")
        cur_headers, cur_headers_len = "", 0  # 舍弃太大的 headers
    
    # 【3】检查是否需要创建新 chunk
    if cur_len + split_len + cur_headers_len > self.chunk_size:
        # ...保存当前 chunk...
        # ...处理 overlap...
        
        # 【4】如果空间足够，添加 headers 到新 chunk
        if (
            cur_headers
            and split_len + cur_headers_len < self.chunk_size
            and cur_headers not in split
        ):
            # headers 不在 split 中才添加（避免重复）
            cur_chunk.insert(0, (header_start, header_end, cur_headers))
            cur_len += cur_headers_len
    
    # 【5】添加当前 split
    cur_chunk.append((cur_start, cur_end, split))
    cur_len += split_len
    cur_start = cur_end
```

### TextSplitter 设计亮点总结
- **文本保护机制**： 通过 regex 保护特殊内容。_split_protected() + _join() 使表格、代码块、公式、链接等强关联内容保持完整。
- **递归多层语义分割**： 多个分隔符逐级尝试，保留语义边界（段落 → 句子 → 词 → 字），递归调用，确保 chunk_size 不超过限制。
- **Overlap 处理**： Chunks 间有重叠，上下文连贯，增强 Chunks 间的连接性。
- **标题上下文保留**： HeaderTracker 获取多级标题，Chunk 包含标题上下文确保语义完整。
- **位置映射**： (start, end, text) 支持回溯原文，提供溯源高亮显示的能力。

### Chunk list 安全策略
直接根据最大 chunk 数（默认 1000）截断，超过部分直接丢弃。防止内存溢出 (OOM)，同时避免数据库性能问题。但这个策略同样存在信息完整性问题，可以考虑根据实际场景调整最大 chunk 数。
```python
# Limit the number of returned chunks
if len(chunks) > self.max_chunks:
    logger.warning(
        f"Limiting chunks from {len(chunks)} to maximum {self.max_chunks}"
    )
    chunks = chunks[: self.max_chunks]
```

## Chunk 图片内容处理
多模态图片处理流程：对每个 Chunk 中的图片进行提取、下载上传、OCR 文字识别和 VLM 图片描述生成。
### 多模态启用条件
```python
if self.enable_multimodal:
    # 支持的文件类型
    allowed_types = [
        # 文本文档
        ".pdf", ".md", ".markdown", ".doc", ".docx",
        # 纯图片文件
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp",
    ]
    
    if file_ext in allowed_types:
        chunks = self.process_chunks_images(chunks, document.images)
```

### 图片处理流程
对单个 Chunk 中的图片处理步骤：
1. **提取图片引用** - 从 `chunk.content` 中提取 Markdown 图片语法 `![alt](url)`
2. **下载/上传图片** - 将图片统一存储到对象存储 (COS/MinIO)
3. **OCR 文字识别** - 使用 Paddle OCR 提取图片中的文字
4. **VLM 图片描述** - 调用多模态 LLM 生成图片自然语言描述
5. **更新 chunk.images** - 将处理结果写入 Chunk

### 图片下载策略
```python
async def download_and_upload_image(self, img_url: str):
    """
    处理三种情况：
    1. 已在存储中 (COS/MinIO) → 直接使用
    2. 本地文件              → 上传到存储
    3. 远程 URL              → 下载后上传到存储
    """
```

### SSRF 安全防护
验证 URL 防止 SSRF 攻击：
```python
@staticmethod
def _is_safe_url(url: str) -> bool:
    """
    拒绝的 URL 类型：
    1. 非 HTTP/HTTPS 协议
    2. 私有 IP (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
    3. Loopback IP (127.0.0.1, ::1)
    4. 云服务元数据端点 (169.254.169.254)
    5. 本地主机名 (.local, localhost)
    """
```

### OCR & VLM 

| 处理方式 | 支持引擎 | 输入 | 输出 | 用途 |
|---------|------|------|------|------|
| **OCR** | Paddle OCR/Nanonets OCR | PIL Image | 图片中的文字 | 关键字匹配、结构化数据提取 |
| **VLM** | OpenAI/Ollama 自部署 | Base64 图片 | 自然语言描述 | 语义理解、RAG 问答增强 |

### OCR 实现详解 (PaddleOCR)

#### 初始化配置
```python
ocr_config = {
    "use_gpu": False,                    # 禁用 GPU，使用 CPU
    "text_det_limit_side_len": 960,      # 图片长边限制 960px
    "use_doc_orientation_classify": True, # 自动检测文档方向（0°/90°/180°/270°）
    "use_textline_orientation": True,     # 文本行方向检测
    
    # 模型选择（v4 最新版本）
    "text_recognition_model_name": "PP-OCRv4_server_rec",
    "text_detection_model_name": "PP-OCRv4_server_det",
    
    # 检测阈值
    "text_det_thresh": 0.3,              # 检测候选框阈值
    "text_det_box_thresh": 0.6,          # 文本框置信度阈值
    "text_det_unclip_ratio": 1.5,        # 文本框扩大比例
    
    # 高精度模式
    "use_dilation": True,                # 膨胀操作提高准确率
    "det_db_score_mode": "slow",         # 慢速但准确的评分模式
    "lang": "ch",                        # 识别中文
}
```

#### CPU 兼容性检测
PaddleOCR 使用 AVX 指令集加速，老旧 CPU 可能不支持：
```python
# 检测 CPU 是否支持 AVX
if platform.system() == "Linux":
    result = subprocess.run(
        ["grep", "-o", "avx", "/proc/cpuinfo"],
        capture_output=True, text=True, timeout=5
    )
    has_avx = "avx" in result.stdout.lower()
    
    if not has_avx:
        # 降级到兼容模式
        os.environ["FLAGS_use_avx2"] = "0"
        os.environ["FLAGS_use_avx"] = "1"
```

#### OCR 识别流程
```python
def _predict(self, image: Image.Image) -> str:
    # 1. 确保 RGB 格式
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # 2. 转换为 numpy 数组
    image_array = np.array(image)
    
    # 3. 执行 OCR
    ocr_result = self.ocr.ocr(image_array, cls=False)
    # 返回格式：[[[坐标框], ("文字", 置信度)], ...]
    
    # 4. 提取文字
    text = [line[1][0] for line in ocr_result[0] if line and line[1]]
    return " ".join(text)
```

### Chunk 图片信息结构
```go
chunk.images = [
    {
        # 原始信息
        "original_url": "https://example.com/chart.png",
        "start": 40,
        "end": 90,
        "alt_text": "销售图表",
        "match_text": "![销售图表](https://...)",
        
        # 存储信息
        "cos_url": "https://storage.local/abc123.png",
        
        # OCR 结果
        "ocr_text": "Q1销售额\n一月：100万\n二月：110万",
        
        # VLM 结果
        "caption": "这是一个柱状图，展示了Q1的月度销售数据..."
    }
]
```
生成最终的 chunk 结构后，将对 chunk 进行向量化，以及根据配置进行相应的 RAG 增强。

## 向量化
### 清除旧数据
清理旧的 chunks 和索引数据，避免重复数据，如果存在知识图谱数据也同样清除。
```go
// 删除旧的chunks
err := s.chunkService.DeleteChunksByKnowledgeID(ctx, knowledge.ID);
// 删除旧的索引数据
err := retrieveEngine.DeleteByKnowledgeIDList(ctx, []string{knowledge.ID}, embeddingModel.GetDimensions(), knowledge.Type);
// 删除知识图谱数据（如果存在）
err := s.graphEngine.DelGraph(ctx, []types.NameSpace{namespace});
```
### 构建 Chunk 对象
包括 chunk 的文本内容和图片 OCR 信息，以及图片 Caption 信息。
```go
for _, chunkData := range chunks {
    // 1. 创建主文本 Chunk
    textChunk := &types.Chunk{
        ID:              uuid.New().String(),
        TenantID:        knowledge.TenantID,
        KnowledgeID:     knowledge.ID,
        KnowledgeBaseID: knowledge.KnowledgeBaseID,
        Content:         chunkData.Content,
        ChunkIndex:      int(chunkData.Seq),
        ChunkType:       types.ChunkTypeText,  // "text"
        // ...
    }
    insertChunks = append(insertChunks, textChunk)

    // 2. 处理图片信息
    if len(chunkData.Images) > 0 {
        for i, img := range chunkData.Images {
            // 2.1 创建 OCR Chunk（如果有 OCR 文本）
            if img.OcrText != "" {
                ocrChunk := &types.Chunk{
                    ID:            uuid.New().String(),
                    Content:       img.OcrText,
                    ChunkType:     types.ChunkTypeImageOCR,  // "image_ocr"
                    ParentChunkID: textChunk.ID,             // 关联到父 Chunk
                    ImageInfo:     string(imageInfoJSON),
                    // ...
                }
                insertChunks = append(insertChunks, ocrChunk)
            }

            // 2.2 创建 Caption Chunk（如果有图片描述）
            if img.Caption != "" {
                captionChunk := &types.Chunk{
                    ID:            uuid.New().String(),
                    Content:       img.Caption,
                    ChunkType:     types.ChunkTypeImageCaption,  // "image_caption"
                    ParentChunkID: textChunk.ID,
                    ImageInfo:     string(imageInfoJSON),
                    // ...
                }
                insertChunks = append(insertChunks, captionChunk)
            }
        }
        // 将图片信息保存到文本 Chunk
        textChunk.ImageInfo = string(imageInfoJSON)
    }
}
```
### 设置 Chunk 关系
为**文本类型**的 Chunk 设置前后关系，构建 chunk 链表。为了支持检索时的上下文扩展。
```go
for i, chunk := range textChunks {
    if i > 0 {
        textChunks[i-1].NextChunkID = chunk.ID
    }
    if i < len(textChunks)-1 {
        textChunks[i+1].PreChunkID = chunk.ID
    }
}
```
### 构建向量数据索引信息
```go
indexInfoList := make([]*types.IndexInfo, 0, len(insertChunks))
for _, chunk := range insertChunks {
    indexInfoList = append(indexInfoList, &types.IndexInfo{
        Content:         chunk.Content,      // 用于生成 embedding
        SourceID:        chunk.ID,
        SourceType:      types.ChunkSourceType,
        ChunkID:         chunk.ID,
        KnowledgeID:     knowledge.ID,
        KnowledgeBaseID: knowledge.KnowledgeBaseID,
    })
}
```
### 检查存储配额检查
简单估算本次插入向量的存储大小，检查是否超过配额。存储大小 ≈ 索引条目数 × (向量维度 × 4 + 元数据大小)
```go
// 估算存储大小
totalStorageSize := retrieveEngine.EstimateStorageSize(ctx, embeddingModel, indexInfoList)

// 检查配额
if tenantInfo.StorageQuota > 0 {
    if tenantInfo.StorageUsed + totalStorageSize > tenantInfo.StorageQuota {
        knowledge.ParseStatus = types.ParseStatusFailed
        knowledge.ErrorMessage = "存储空间不足"
        s.repo.UpdateKnowledge(ctx, knowledge)
        return
    }
}
```
### 保存 Chunk 列表并向量化
先将 chunk 列表保存到数据库，然后批量向量化。
```go
// 保存 chunk 列表
err := s.chunkService.CreateChunks(ctx, insertChunks); 

// 批量向量化
err = retrieveEngine.BatchIndex(ctx, embeddingModel, indexInfoList)
```
## 知识图谱构建
### 开启知识图谱提取
```go
if kb.ExtractConfig != nil && kb.ExtractConfig.Enabled {
    for _, chunk := range textChunks {
        err := NewChunkExtractTask(ctx, s.task, chunk.TenantID, chunk.ID, kb.SummaryModelID)
        ...
    }
}
```
### 构建提取模板
构建知识图谱提取 prompt 模板，用于 LLM 提取知识图谱中的实体和关系。
```go
template := &types.PromptTemplateStructured{
    Description: s.template.Description,
    Tags:        kb.ExtractConfig.Tags,        // 实体类型定义
    Examples: []types.GraphData{
        {
            Text:     kb.ExtractConfig.Text,      // 示例文本
            Node:     kb.ExtractConfig.Nodes,     // 示例节点
            Relation: kb.ExtractConfig.Relations, // 示例关系
        },
    },
}
```
### LLM 提取实体和关系
调用 LLM 提取 chunk 中的实体和关系，根据模板生成结构化的知识图谱。在提取实体和关系时，若 LLM 提取出了 relation，但没有对应的 node，会触发自动补充节点功能，将缺失的节点添加到知识图谱中。
```go
extractor := chatpipline.NewExtractor(chatModel, template)
graph, err := extractor.Extract(ctx, chunk.Content)
```
### 关联 Chunk 写入图数据库
将提取到的知识图谱关联到对应的 Chunk 中，存储在数据库中。
```go
// 为每个节点关联来源 Chunk
for _, node := range graph.Node {
    node.Chunks = []string{chunk.ID}
}

// 写入图数据库
err = s.graphEngine.AddGraph(ctx,
    types.NameSpace{
        KnowledgeBase: chunk.KnowledgeBaseID,
        Knowledge:     chunk.KnowledgeID,
    },
    []*types.GraphData{graph},
)
```

## 关联问题生成
### 开启问题生成
```go
if options.EnableQuestionGeneration && len(textChunks) > 0 {
    questionCount := options.QuestionCount
    if questionCount <= 0 {
        questionCount = 3
    }
    if questionCount > 10 {
        questionCount = 10
    }
    s.enqueueQuestionGenerationTask(ctx, knowledge.KnowledgeBaseID, knowledge.ID, questionCount)
}
```
### 构建问题生成相关文本
先对文本 chunk 按照时间排序，在通过当前文本 chunk 的链表结构结合前后 chunk 内容构建问题生成的相关文本内容。
```go
// 按 StartAt 排序（用于获取上下文）
sort.Slice(textChunks, func(i, j int) bool {
    return textChunks[i].StartAt < textChunks[j].StartAt
})
// 获取当前文本 chunk 的前后 chunk 内容，用于构建问题生成的相关文本
for i, chunk := range textChunks {
    // 获取前后 chunk 作为上下文
    var prevContent, nextContent string
    if i > 0 {
        prevContent = textChunks[i-1].Content
        if len(prevContent) > 500 {
            prevContent = prevContent[len(prevContent)-500:]  // 取后500字符
        }
    }
    if i < len(textChunks)-1 {
        nextContent = textChunks[i+1].Content
        if len(nextContent) > 500 {
            nextContent = nextContent[:500]  // 取前500字符
        }
    }
    ...
}
```
### 生成 Chunk 相关问题
通过 LLM 结合上下文（前 chunk 内容、后 chunk 内容、知识标题）生成与当前 chunk 相关的问题，并保存至 metadata。
```go
// 4. 调用 LLM 生成问题
questions, _ := s.generateQuestionsWithContext(ctx, chatModel, 
    chunk.Content, prevContent, nextContent, knowledge.Title, questionCount)

// 5. 保存问题到 chunk 的 metadata
generatedQuestions := make([]types.GeneratedQuestion, len(questions))
for j, question := range questions {
    questionID := fmt.Sprintf("q%d", time.Now().UnixNano()+int64(j))
    generatedQuestions[j] = types.GeneratedQuestion{
        ID:       questionID,
        Question: question,
    }
}
```
### 问题向量化
将生成的问题向量化，用于检索。
```go
// 构建问题索引
for _, gq := range generatedQuestions {
    sourceID := fmt.Sprintf("%s-%s", chunk.ID, gq.ID)
    indexInfoList = append(indexInfoList, &types.IndexInfo{
        Content:         gq.Question,
        SourceID:        sourceID,
        ChunkID:         chunk.ID,  // 指向原始 chunk
        KnowledgeID:     knowledge.ID,
        KnowledgeBaseID: knowledge.KnowledgeBaseID,
    })
}
// 向量化
retrieveEngine.BatchIndex(ctx, embeddingModel, indexInfoList)
```

## 摘要生成
若存在文本 chunk，则会触发摘要生成任务。
```go
if len(textChunks) > 0 {
    s.enqueueSummaryGenerationTask(ctx, knowledge.KnowledgeBaseID, knowledge.ID)
}
```
### 排序 Chunk 内容
对文本 chunk 按照 ChunkIndex 排序，确保摘要内容按文档顺序拼接。
```go
// 按 ChunkIndex 排序
sort.Slice(textChunks, func(i, j int) bool {
    return textChunks[i].ChunkIndex < textChunks[j].ChunkIndex
})
```

### 处理摘要原始内容
对排序后的 chunk 内容进行拼接，限制总长度为 4096 字符，作为摘要的原始内容。同时处理原始内容中的图片描述和 OCR 文字。
```go
// 拼接 chunk 内容（按 StartAt 排序，限制 4096 字符）
chunkContents := ""
for _, chunk := range sortedChunks {
    if chunk.EndAt > 4096 {
        break  // 限制内容长度
    }
    chunkContents = string([]rune(chunkContents)[:chunk.StartAt]) + chunk.Content
    
    // 收集图片信息
    if chunk.ImageInfo != "" {
        var images []*types.ImageInfo
        json.Unmarshal([]byte(chunk.ImageInfo), &images)
        allImageInfos = append(allImageInfos, images...)
    }
}

// 移除 Markdown 图片语法
re := regexp.MustCompile(`!\[[^\]]*\]\([^)]+\)`)
chunkContents = re.ReplaceAllString(chunkContents, "")

// 添加图片描述和 OCR 文字
if len(allImageInfos) > 0 {
    var imageAnnotations string
    for _, img := range allImageInfos {
        if img.Caption != "" {
            imageAnnotations += fmt.Sprintf("\n[图片描述: %s]", img.Caption)
        }
        if img.OCRText != "" {
            imageAnnotations += fmt.Sprintf("\n[图片文字: %s]", img.OCRText)
        }
    }
    chunkContents = chunkContents + imageAnnotations
}

// 内容太短直接返回
if len(chunkContents) < 300 {
    return chunkContents, nil
}

// 添加文档元数据
metadataIntro := fmt.Sprintf("文档类型: %s\n文件名称: %s\n", knowledge.FileType, knowledge.FileName)
contentWithMetadata = metadataIntro + "\n内容:\n" + chunkContents
```
### 调用 LLM 生成摘要
```go
summary, err := summaryModel.Chat(ctx, []chat.Message{
    {Role: "system", Content: s.config.Conversation.GenerateSummaryPrompt},
    {Role: "user",   Content: contentWithMetadata},
}, &chat.ChatOptions{
    Temperature: 0.3,
    MaxTokens:   1024,
    Thinking:    &thinking,
})
```

### 保存摘要并向量化
```go
// 保存摘要 chunk
summaryChunk := &types.Chunk{
    ID:              uuid.New().String(),
    TenantID:        knowledge.TenantID,
    KnowledgeID:     knowledge.ID,
    KnowledgeBaseID: knowledge.KnowledgeBaseID,
    Content:         fmt.Sprintf("# 文档名称\n%s\n\n# 摘要\n%s", knowledge.FileName, summary),
    ChunkIndex:      maxChunkIndex + 1,
    IsEnabled:       true,
    ChunkType:       types.ChunkTypeSummary,
    ParentChunkID:   textChunks[0].ID,
}
s.chunkService.CreateChunks(ctx, []*types.Chunk{summaryChunk})
// 向量化摘要 chunk
indexInfo := []*types.IndexInfo{{
    Content:         summaryChunk.Content,
    SourceID:        summaryChunk.ID,
    SourceType:      types.ChunkSourceType,
    ChunkID:         summaryChunk.ID,
    KnowledgeID:     knowledge.ID,
    KnowledgeBaseID: knowledge.KnowledgeBaseID,
}}
retrieveEngine.BatchIndex(ctx, embeddingModel, indexInfo) 
```
# 尾言
本文从工程视角系统梳理了 WeKnora 在**知识构建阶段**的整体设计：  
从多种知识上传方式开始，经过异步解析与降级策略保障；再到以 Markdown 为核心中间表示的统一解析架构；随后通过具备语义感知、结构保护与标题追踪能力的 TextSplitter 生成高质量 Chunk；并进一步叠加图片 OCR、VLM 描述、多模态向量、知识图谱抽取、问题生成与摘要生成，最终形成一套**可检索、可追溯、可扩展的知识底座**。

可以看到，WeKnora 并没有将 RAG 简化为“切 chunk + 向量化”这样的轻量流程，而是非常认真地对待**文档结构、上下文完整性和信息损失控制**这些在生产环境中至关重要的问题。这也是它与许多示例型 RAG 项目最本质的区别。

在下一篇文章中，我们将把视角从「知识如何被构建」转向「知识如何被召回」，重点分析 WeKnora 的**检索体系设计**：
  
它是如何结合 **稀疏索引（关键词/倒排）**、**向量索引（语义检索）** 与 **知识图谱检索** 的？

在召回中是如何使用文档摘要， chunk 相关提问，图片 ocr 文本以及格式化表格等信息的？

不同召回通道在什么场景下生效，又是如何进行融合、裁剪与排序的？

这将直接决定 WeKnora 在真实问答场景中的“命中率”和“可解释性”，也是整套 RAG 系统中承上启下的关键一环。