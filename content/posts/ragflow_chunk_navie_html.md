---
title: "【解密源码】 RAGFlow 切分最佳实践- naive parser 语义切块（html 篇）"
date: 2025-10-26T11:10:00+08:00
draft: false
tags: ["源码","技术","RAG"]
categories: ["RAGFlow"]
---

# 引言

# 省流版

# 手撕版

## HTML 文档
1. 解析器初始化，调用类实例解析 html 文档
```python
sections = HtmlParser()(filename, binary, chunk_token_num)
```

### HtmlParser 类
推断正确编码进行解码后，解析 html 文件。

```python
class RAGFlowHtmlParser:
    def __call__(self, fnm, binary=None, chunk_token_num=512):
        if binary:
            encoding = find_codec(binary)
            txt = binary.decode(encoding, errors="ignore")
        else:
            with open(fnm, "r",encoding=get_encoding(fnm)) as f:
                txt = f.read()
        return self.parser_txt(txt, chunk_token_num)
```

#### parser_txt
`parser_txt` 是解析 html 文档的核心方法，主要分 4 个步骤：
1. 移除文档干扰信息
```python
# 删除 style 和 script 标签
for style_tag in soup.find_all(["style", "script"]):
    style_tag.decompose()
# 删除 div 中的 script 标签
for div_tag in soup.find_all("div"):
    for script_tag in div_tag.find_all("script"):
        script_tag.decompose()
# 删除内联样式
for tag in soup.find_all(True):
    if 'style' in tag.attrs:
        del tag.attrs['style']
# 删除 HTML 注释
for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
    comment.extract()
```
2. 递归提取文本
```python
cls.read_text_recursively(soup.body, temp_sections, chunk_token_num=chunk_token_num)

@classmethod
def read_text_recursively(cls, element, parser_result, chunk_token_num=512, parent_name=None, block_id=None):
    if isinstance(element, NavigableString): # 判断是否为文本节点 NavigableString
        ...
        if is_valid_html(content): # 判断是否为有效 html 节点
            soup = BeautifulSoup(content, "html.parser")
            # 递归获取子节点文本内容
            child_info = cls.read_text_recursively(soup, parser_result, chunk_token_num, element.name, block_id)
            parser_result.extend(child_info)
        else:
            # 提取文本
            info = {"content": element.strip(), "tag_name": "inner_text", "metadata": {"block_id": block_id}}
            if parent_name:
                info["tag_name"] = parent_name
            return_info.append(info)
        ...
    elif isinstance(element, Tag): # 判断是否为标签节点
        if str.lower(element.name) == "table": # 表格节点特殊处理
            table_info_list = []
            table_id = str(uuid.uuid1())
            table_list = [html.unescape(str(element))]
            for t in table_list:
                table_info_list.append({"content": t, "tag_name": "table",
                                        "metadata": {"table_id": table_id, "index": table_list.index(t)}})
            return table_info_list
        else: # 保底处理，识别所有 html 标签 BLOCK_TAGS 集合
            block_id = None
            if str.lower(element.name) in BLOCK_TAGS:
                block_id = str(uuid.uuid1())
            for child in element.children:
                child_info = cls.read_text_recursively(child, parser_result, chunk_token_num, element.name,
                                                        block_id)
                parser_result.extend(child_info)
    ...
```
>*表格节点特殊处理：保留表格原始样式标签，不会进行文本提取。如："content": “`<table border="1"><tr><th>姓名</th><th>年龄</th></tr><tr><td>张三</td><td>25</td></tr><tr><td>李四</td><td>30</td></tr></table>`”*,

>*提取文本返回结构中设计 block id 信息，用于后续合并文本*
3. 文本合并
```python
block_txt_list, table_list = cls.merge_block_text(temp_sections)

@classmethod
def merge_block_text(cls, parser_result):
    ...
    if block_id:
        if title_flag:
            content = f"{TITLE_TAGS[tag_name]} {content}" #标题添加 Markdown 格式
        if lask_block_id != block_id:
            if lask_block_id is not None:
                block_content.append(current_content)
            current_content = content
            lask_block_id = block_id
        else:
            current_content += (" " if current_content else "") + content
    else:
        if tag_name == "table":
            table_info_list.append(item)
        else:
            current_content += (" " if current_content else "" + content)
```
>*根据 block id 合并文本，对于表格仍保持原始结构完整。*

>*将标题转换成 Markdown 格式（TITLE_TAGS = {"h1": "#", "h2": "##", "h3": "###", "h4": "#####", "h5": "#####", "h6": "######"}）保留语义信息*
4. 文本切分
按 `chunk_token_num` 配置切分文本
```python
sections = cls.chunk_block(block_txt_list, chunk_token_num=chunk_token_num)

@classmethod
    def chunk_block(cls, block_txt_list, chunk_token_num=512):
        ...
```

## JSON 文档
1. 解析器初始化，调用类实例解析 JSON 文档
```python
sections = JsonParser(chunk_token_num)(binary)
```

### JsonParser 类
推断正确编码进行解码后，解析 JSON 文件，兼容普通 JSON 格式和 JSONL 格式。
```python
def __call__(self, binary):
    encoding = find_codec(binary)
    txt = binary.decode(encoding, errors="ignore")

    if self.is_jsonl_format(txt):
        sections = self._parse_jsonl(txt)
    else:
        sections = self._parse_json(txt)
    return sections
```
对于 JSONL 的处理是先将 JSONL 每行转换成 JSON 格式后调用 `split_json` 处理。
```python
 def _parse_jsonl(self, content: str) -> list[str]:
    lines = content.strip().splitlines()
    all_chunks = []
    for line in lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            chunks = self.split_json(data, convert_lists=True)
    ...
```

#### split_json
1. 数据结构转换，递归将 JSON 中列表值转换成字典。
```python
preprocessed_data = self._list_to_dict_preprocessing(json_data)

def _list_to_dict_preprocessing(self, data: Any) -> Any:
    if isinstance(data, dict):
        # Process each key-value pair in the dictionary
        return {k: self._list_to_dict_preprocessing(v) for k, v in data.items()}
    elif isinstance(data, list):
        # Convert the list to a dictionary with index-based keys
        return {str(i): self._list_to_dict_preprocessing(item) for i, item in enumerate(data)}
    else:
        # Base case: the item is neither a dict nor a list, so return it unchanged
        return data
```
递归转换示例：
```python
# 输入
{"a": [1, 2, 3], "b": {"c": ["x", "y"]}}

# 输出
{
  "a": {
    "0": 1,
    "1": 2, 
    "2": 3
  },
  "b": {
    "c": {
      "0": "x",
      "1": "y"
    }
  }
}
```
2. 按照长度配置对 JSON 进行切分，输出 chunk。
```python
chunks = self._json_split(preprocessed_data, None, None)

def _json_split(self, data, current_path: list[str] | None, chunks: list[dict] | None) -> list[dict]:
    """
    Split json into maximum size dictionaries while preserving structure.
    """
    ...
    for key, value in data.items():
        new_path = current_path + [key]
        chunk_size = self._json_size(chunks[-1])
        size = self._json_size({key: value})
        remaining = self.max_chunk_size - chunk_size

        if size < remaining:
            # Add item to current chunk
            self._set_nested_dict(chunks[-1], new_path, value)
        else:
            if chunk_size >= self.min_chunk_size:
                # Chunk is big enough, start a new chunk
                chunks.append({})

            # Iterate
            self._json_split(value, new_path, chunks)
    ...
```
切分示例：
```python
# 输入
{"a": 1, "b": "hello", "c": {"d": 2, "e": "world"}}

#输出（具体输出需要根据 max_chunk_size 与 min_chunk_size 配置）
[
  {
    "a": 1,
    "b": "hello"
  },
  {
    "c": {
      "d": 2,
      "e": "world"
    }
  }
]
```
>*如果多层嵌套数据，使用 new_path = current_path + [key] 来维护嵌套层级，优先对 JSON 数据进行深度遍历*
```python
@staticmethod
def _set_nested_dict(d: dict, path: list[str], value: Any) -> None:
    """Set a value in a nested dictionary based on the given path."""
    for key in path[:-1]:
        d = d.setdefault(key, {})
    d[path[-1]] = value
```

## DOC 文档
兼容旧版 word .doc 文档解析。使用 python Tika 库解析 doc 文档。
```python
 elif re.search(r"\.doc$", filename, re.IGNORECASE):
    doc_parsed = parser.from_buffer(binary)
    sections = doc_parsed['content'].split('\n')
    sections = [(_, "") for _ in sections if _]
```

HTML，JSON，DOC 文档经过切分得到 sections 后，还需要进行 sections 后处理，这部分可参考《naive parser 语义切块（pdf 篇）》中 **sections 后处理模块中的无图 sections 处理逻辑**，经过后处理后得到最终输出的 res。