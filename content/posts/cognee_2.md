---
title: "【解密源码】Cognee Pipeline 机制与工作流深度解析"
date: 2025-12-16T11:33:10+08:00
draft: false
tags: ["源码","技术","Agent"]
categories: ["Cognee"]
---

## 引言

### 为什么需要理解 Pipeline 机制？

在第一篇文章中，我们从**.add、.cognify、.search、.memify**各自的角度剖析了实现细节。但如果你问："这些功能是怎么执行的？任务怎么协调？为什么能支持后台处理？"——这些问题的答案都在 **Pipeline 机制**中。

Pipeline 是 Cognee 的执行骨架，它决定了：
- 如何组织和调度任务
- 如何处理任务间的数据依赖
- 如何在失败时恢复
- 如何监控执行状态

掌握 Pipeline 机制，你才能深入定制 Cognee、开发高级应用、甚至贡献源代码。

### 本篇的讲解策略

本篇采用**由浅入深**的方式：
1. **基础概念**：Task、Executor、Pipeline 的定义与关系
2. **工作流程**：从任务创建到执行完成的全过程
3. **高级话题**：错误恢复、监控、自定义扩展
4. **实战指南**：如何编写自定义任务和执行器

---

## 一、Pipeline 核心概念

### 1.1 三层架构：Task → Executor → Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                   Pipeline（流水线）                    │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Executor（执行器）                              │   │
│  │  ┌──────────────────────────────────────────┐   │   │
│  │  │ Task 1: classify_documents              │   │   │
│  │  │ Task 2: extract_chunks_from_documents  │   │   │
│  │  │ Task 3: extract_graph_from_data        │   │   │
│  │  │ Task 4: summarize_text                 │   │   │
│  │  └──────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

| 层级 | 职责 | 示例 |
|------|------|------|
| **Task** | 单个操作单元 | `extract_chunks_from_documents(documents, max_chunk_size)` |
| **Executor** | 任务的执行方式 | 顺序执行、并行执行、分布式执行 |
| **Pipeline** | 整体流程编排 | 四大功能（.add、.cognify、.search、.memify） |

### 1.2 设计理念

**关键特性**：
- **任务抽象**：每个操作都是一个 Task，支持参数化和复用
- **灵活调度**：Executor 负责如何执行，Pipeline 负责执行什么
- **异步优先**：所有操作都是异步的，支持大规模并发
- **可观测**：每个执行步骤都记录状态和结果

### 1.3 第一篇与本篇的关联

回顾第一篇中的 `.cognify` 七步骤：

```python
tasks = [
    Task(classify_documents),
    Task(check_permissions_on_dataset, user=user, permissions=["write"]),
    Task(extract_chunks_from_documents, ...),
    Task(extract_graph_from_data, ...),
    Task(summarize_text, ...),
    Task(add_data_points, ...),
]
```

这就是 Pipeline 机制的实际应用：
- 每个 `Task(func, **kwargs)` 封装一个操作
- Pipeline 将这些 Task 按序组织
- Executor 负责如何高效地执行它们

本篇将深入解释这个过程的**"如何"**和**"为什么"**。

---

## 二、Task：任务的最小单位

### 2.1 Task 的结构与定义

```python
class Task:
    """Cognee 的任务基类"""
    
    def __init__(
        self,
        func: Callable,  # 执行函数
        *args,
        **kwargs
    ):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
```

### 2.2 Task 的生命周期

```
创建 → 入队 → 就绪 → 执行中 → 完成/失败 → 结果存储
```

**状态转移详解**：

| 状态 | 含义 | 触发条件 | 后续状态 |
|------|------|---------|---------|
| **PENDING** | 待执行 | Task 被创建 | QUEUED |
| **QUEUED** | 已入队 | 加入 Executor 队列 | RUNNING |
| **RUNNING** | 执行中 | Executor 开始执行 | COMPLETED / FAILED |
| **COMPLETED** | 完成 | 执行成功 | （终态）|
| **FAILED** | 失败 | 执行异常或超时 | RETRYING / CANCELLED |
| **RETRYING** | 重试中 | 触发重试逻辑 | RUNNING / FAILED |
| **CANCELLED** | 已取消 | 用户或系统取消 | （终态）|

**示例：Task 执行的完整流程**

```python
# 第一篇中的实际例子
task = Task(
    extract_chunks_from_documents,
    documents=doc_list,
    max_chunk_size=2048,
    chunker=TextChunker
)

# 生命周期跟踪
print(task.status)  # TaskStatus.PENDING

await executor.execute(task)

# 执行完成后
print(task.status)   # TaskStatus.COMPLETED
print(task.result)   # [DocumentChunk, DocumentChunk, ...]
```

### 2.3 内置任务与自定义任务

**内置任务**（Cognee 提供）：

Cognee 在 `cognee/tasks/` 目录下提供了大量内置任务：

| 任务名 | 所属模块 | 功能 | 输入 | 输出 |
|--------|---------|------|------|------|
| `classify_documents` | cognify | 文档类型分类 | `List[Document]` | `List[Document]` （带类型标签）|
| `extract_chunks_from_documents` | cognify | 文本分块 | `List[Document]`, `max_chunk_size` | `AsyncGenerator[DocumentChunk]` |
| `extract_graph_from_data` | cognify | 知识图提取 | `List[DocumentChunk]`, `graph_model` | `List[Edge]` |
| `summarize_text` | cognify | 摘要生成 | `List[DocumentChunk]` | `List[TextSummary]` |
| `add_data_points` | graph | 数据存储 | `List[Node/Edge]` | `None` |
| `extract_subgraph_chunks` | memify | 子图提取 | `graph_id`, `node_type` | `List[Node]` |

**自定义任务**：

```python
# 定义自定义函数
async def my_custom_task(data: list, param: str) -> dict:
    """自定义任务示例"""
    result = {}
    for item in data:
        # 业务逻辑
        result[item.id] = process(item, param)
    return result

# 将其包装为 Task
task = Task(
    my_custom_task,
    data=input_data,
    param="some_value"
)

# 在 Pipeline 中使用
await pipeline.execute([task])
```

### 2.4 Task 的参数管理

**参数来源**：

```python
# 方式 1：直接传入参数
task = Task(
    extract_chunks_from_documents,
    documents=[doc1, doc2],
    max_chunk_size=2048
)

# 方式 2：从 Context 获取参数（动态）
task = Task(
    extract_graph_from_data,
    graph_model="${context.graph_model}",  # 从上下文替换
    config="${context.config}"
)

# 方式 3：参数插值与合并
task_config = {
    "batch_size": 10,
    "timeout": 300,
    "retry_count": 3
}

task = Task(
    process_chunks,
    chunks=chunk_list,
    **task_config  # 参数展开
)
```

**参数验证**：

```python
# Task 执行前的参数检查
class Task:
    def validate_params(self) -> bool:
        """验证参数有效性"""
        # 检查必需参数
        sig = inspect.signature(self.func)
        required_params = {
            param.name for param in sig.parameters.values()
            if param.default == inspect.Parameter.empty
        }
        provided_params = set(self.kwargs.keys()) | {
            f"arg_{i}" for i in range(len(self.args))
        }
        
        missing = required_params - provided_params
        if missing:
            raise ValueError(f"缺少必需参数: {missing}")
        
        return True
```

---

## 三、Executor：执行引擎设计

### 3.1 执行策略对比

| 执行器 | 机制 | 场景 | 性能 | 优点 | 缺点 |
|--------|------|------|------|------|------|
| **Sequential** | 顺序执行 | 单任务、调试 | 最慢 | 易调试、确定性高 | 无法并行 |
| **Parallel** | 多进程/线程 | 任务独立 | 中等 | 充分利用 CPU | GIL 限制、进程通信开销 |
| **AsyncIO** | 异步事件循环 | I/O 密集 | 最快 | 高并发、低开销 | 需要异步函数 |
| **Distributed** | 远程执行 | 超大规模 | 可扩展 | 无限扩展 | 网络开销、复杂部署 |

### 3.2 AsyncIO Executor 的实现

Cognee 默认使用 AsyncIO Executor，因为大多数操作都是 I/O 密集的（数据库查询、LLM API 调用、向量搜索）。

**核心实现**：

```python
class AsyncIOExecutor:
    """异步事件循环执行器"""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.running_tasks = {}
    
    async def execute_task(self, task: Task) -> Any:
        """执行单个任务"""
        async with self.semaphore:  # 控制并发数
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            try:
                # 调用任务函数
                result = await self._call_func(task.func, *task.args, **task.kwargs)
                task.result = result
                task.status = TaskStatus.COMPLETED
                return result
            
            except Exception as e:
                task.error = e
                task.status = TaskStatus.FAILED
                raise
            
            finally:
                task.completed_at = datetime.now()
    
    async def _call_func(self, func, *args, **kwargs):
        """调用函数，自动处理同步/异步"""
        if asyncio.iscoroutinefunction(func):
            # 异步函数：直接 await
            return await func(*args, **kwargs)
        else:
            # 同步函数：在线程池中运行，避免阻塞
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)
    
    async def execute_tasks(self, tasks: List[Task]) -> List[Any]:
        """并发执行多个任务"""
        results = []
        for task in tasks:
            # 创建并发任务
            coro = self.execute_task(task)
            results.append(coro)
        
        # 并发等待所有任务完成
        return await asyncio.gather(*results, return_exceptions=True)
```

**使用示例**：

```python
# 四大功能中的实际应用（来自第一篇）
executor = AsyncIOExecutor(max_concurrent=10)

# .cognify 的七个任务并发执行
tasks = [
    Task(classify_documents, documents=docs),
    Task(check_permissions_on_dataset, user=user, permissions=["write"]),
    Task(extract_chunks_from_documents, documents=docs, max_chunk_size=2048),
    Task(extract_graph_from_data, chunks=chunks, graph_model=KnowledgeGraph),
    Task(summarize_text, chunks=chunks),
    Task(add_data_points, nodes=nodes, edges=edges),
]

# 并发执行（实际上大部分任务有依赖，所以用 Pipeline 而非直接 gather）
results = await executor.execute_tasks(tasks)
```

### 3.3 错误隔离与重试

**单个任务的失败不影响其他任务**（错误隔离）：

```python
async def execute_tasks(self, tasks: List[Task]) -> List[Any]:
    """并发执行，错误隔离"""
    results = await asyncio.gather(
        *[self.execute_task(task) for task in tasks],
        return_exceptions=True  # 捕获异常，继续执行其他任务
    )
    
    # 处理结果
    for task, result in zip(tasks, results):
        if isinstance(result, Exception):
            logger.error(f"任务 {task.func.__name__} 失败: {result}")
            # 可以在这里触发重试逻辑
        else:
            logger.info(f"任务 {task.func.__name__} 成功")
    
    return results
```

**重试策略**：

```python
class RetryPolicy:
    """重试策略"""
    
    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.jitter = jitter
    
    async def execute_with_retry(self, func, *args, **kwargs):
        """带重试的执行"""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            
            except Exception as e:
                last_error = e
                
                # 最后一次尝试，直接抛出异常
                if attempt >= self.max_retries:
                    raise
                
                # 计算等待时间（指数退避 + 抖动）
                wait_time = (self.backoff_factor ** attempt)
                if self.jitter:
                    wait_time *= random.uniform(0.5, 1.5)
                
                logger.warning(
                    f"任务执行失败 (尝试 {attempt + 1}/{self.max_retries + 1}), "
                    f"将在 {wait_time:.1f}s 后重试: {e}"
                )
                
                await asyncio.sleep(wait_time)
        
        raise last_error
```

**在 Task 中应用重试**：

```python
task = Task(
    extract_graph_from_data,
    chunks=chunks,
    graph_model=KnowledgeGraph,
    retry_policy=RetryPolicy(max_retries=3)  # 最多重试 3 次
)
```

---

## 四、Pipeline：任务编排与协调

### 4.1 Pipeline 的定义

Pipeline 是任务的有序集合，负责：
1. **组织任务**：将任务按依赖关系组织成 DAG（有向无环图）
2. **调度执行**：决定任务执行的顺序和方式
3. **管理上下文**：在任务间传递数据和状态
4. **监控状态**：跟踪整体执行进度

**基本结构**：

```python
class Pipeline:
    """Pipeline 类"""
    
    def __init__(
        self,
        tasks: List[Task],
        executor: Executor = None,
        context: Context = None,
        name: str = "pipeline"
    ):
        self.tasks = tasks
        self.executor = executor or AsyncIOExecutor()
        self.context = context or Context()
        self.name = name
        self.status = PipelineStatus.PENDING
        self.results = {}
    
    async def execute(self) -> Dict[str, Any]:
        """执行 Pipeline"""
        self.status = PipelineStatus.RUNNING
        
        try:
            for task in self.tasks:
                # 每个任务执行前解析参数（可能来自上下文）
                task = self._resolve_task_params(task)
                
                # 执行任务
                result = await self.executor.execute_task(task)
                
                # 将结果保存到上下文，供后续任务使用
                self.context.set(task.func.__name__, result)
                self.results[task.func.__name__] = result
            
            self.status = PipelineStatus.COMPLETED
            return self.results
        
        except Exception as e:
            self.status = PipelineStatus.FAILED
            logger.error(f"Pipeline 执行失败: {e}")
            raise
    
    def _resolve_task_params(self, task: Task) -> Task:
        """解析任务参数，支持上下文替换"""
        resolved_kwargs = {}
        for key, value in task.kwargs.items():
            if isinstance(value, str) and value.startswith("${context."):
                # 从上下文获取参数
                param_name = value[10:-1]  # 提取参数名
                resolved_kwargs[key] = self.context.get(param_name)
            else:
                resolved_kwargs[key] = value
        
        task.kwargs = resolved_kwargs
        return task
```

### 4.2 DAG（有向无环图）与任务依赖

.cognify 的七个任务虽然写成列表，但实际上有严格的依赖关系：

```
Task 1: classify_documents
    ↓ (输出文档)
Task 2: check_permissions_on_dataset
    ↓
Task 3: extract_chunks_from_documents
    ↓ (输出文本块)
Task 4: extract_graph_from_data (并行)
Task 5: summarize_text           (并行) ← 这两个可并行，都需要 Task 3 的输出
    ↓
Task 6: add_data_points          ← 等待 Task 4 和 Task 5 完成
```

**DAG 表示**：

```python
# 显式定义依赖关系
dependencies = {
    "classify_documents": [],
    "check_permissions": ["classify_documents"],
    "extract_chunks": ["check_permissions"],
    "extract_graph": ["extract_chunks"],
    "summarize": ["extract_chunks"],  # 与 extract_graph 并行
    "add_data_points": ["extract_graph", "summarize"]  # 等待两个都完成
}

# Pipeline 根据依赖关系调度任务
class DependencyAwarePipeline(Pipeline):
    
    async def execute(self) -> Dict[str, Any]:
        """基于依赖关系的执行"""
        # 构建 DAG
        dag = self._build_dag()
        
        # 按拓扑序执行
        for level in self._topological_sort(dag):
            # 同一 level 的任务可并行执行
            tasks_to_run = [self.tasks[idx] for idx in level]
            results = await self.executor.execute_tasks(tasks_to_run)
            
            # 保存结果
            for task, result in zip(tasks_to_run, results):
                self.context.set(task.func.__name__, result)
        
        return self.results
```

### 4.3 数据在 Pipeline 中的流动

**数据传递的三种方式**：

```python
# 方式 1：直接参数传递
task1 = Task(extract_chunks, documents=docs)
task2 = Task(extract_graph, chunks=task1.result)  # ❌ 执行前 result 还是 None

# 方式 2：通过 Context（推荐）
# Task 执行后自动保存到 context
pipeline.context.set("chunks", task1.result)
# 后续任务从 context 获取
task2 = Task(
    extract_graph,
    chunks="${context.chunks}"  # 参数插值
)

# 方式 3：AsyncGenerator（流式处理）
async def extract_chunks(docs):
    for doc in docs:
        for chunk in split_document(doc):
            yield chunk  # 边生成边返回

async def extract_graph(chunks):
    async for chunk in chunks:
        # 流式处理每个块
        graph = await extract_from_chunk(chunk)
        yield graph
```

**Context 对象的工作原理**：

```python
class Context:
    """执行上下文，存储任务间的数据"""
    
    def __init__(self):
        self._data = {}
        self._locks = {}  # 防止竞态条件
    
    async def set(self, key: str, value: Any):
        """设置值"""
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        
        async with self._locks[key]:
            self._data[key] = value
            logger.debug(f"Context[{key}] = {type(value).__name__}")
    
    def get(self, key: str, default=None) -> Any:
        """获取值"""
        return self._data.get(key, default)
    
    def get_or_wait(self, key: str, timeout: float = 300) -> Any:
        """获取值，如果不存在则等待"""
        # 用于任务间的依赖等待
        start = time.time()
        while key not in self._data:
            if time.time() - start > timeout:
                raise TimeoutError(f"等待 context[{key}] 超时")
            asyncio.sleep(0.1)
        
        return self._data[key]
```

---

## 五、数据流与上下文管理

### 5.1 Context 对象的完整设计

Context 是 Pipeline 中的**共享状态容器**，所有任务可以读写：

```python
class Context:
    """Pipeline 执行上下文"""
    
    def __init__(self, initial_data: dict = None):
        self._data = initial_data or {}
        self._metadata = {}  # 记录每个值的元数据
        self._version = {}   # 版本号，用于追踪变化
    
    def set(self, key: str, value: Any, metadata: dict = None):
        """设置值并记录元数据"""
        self._data[key] = value
        self._version[key] = self._version.get(key, 0) + 1
        
        if metadata:
            self._metadata[key] = {
                **metadata,
                "timestamp": datetime.now(),
                "version": self._version[key]
            }
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取值"""
        return self._data.get(key, default)
    
    def list_keys(self) -> List[str]:
        """列出所有键"""
        return list(self._data.keys())
    
    def clear(self):
        """清空上下文"""
        self._data.clear()
        self._metadata.clear()
        self._version.clear()
```

**实际使用示例**（来自 .cognify）：

```python
# Pipeline 初始化时创建 Context
context = Context({
    "dataset_id": "abc-123",
    "user_id": "user-456",
    "graph_model": KnowledgeGraph,
    "config": {"max_chunk_size": 2048}
})

# Task 1: classify_documents
# 输入：documents（作为参数）
# 输出：保存到 context
docs_classified = await classify(documents)
context.set("documents", docs_classified)

# Task 2: extract_chunks_from_documents
# 输入：从 context 获取 documents
# 输出：保存到 context
documents = context.get("documents")
chunks = await extract_chunks(documents, max_chunk_size=2048)
context.set("chunks", chunks)

# Task 3: extract_graph_from_data (并行)
# 任务 4: summarize_text (并行)
# 都需要 chunks，都保存结果到 context
graphs = await extract_graph(chunks, graph_model=context.get("graph_model"))
context.set("graphs", graphs)

summaries = await summarize(chunks)
context.set("summaries", summaries)

# Task 5: add_data_points
# 输入：graphs 和 summaries
# 输出：持久化到数据库
graphs = context.get("graphs")
summaries = context.get("summaries")
await add_data_points(graphs, summaries)
```

### 5.2 上下文在任务间的传递

**显式传递**：

```python
# Task 定义时明确标记依赖
tasks = [
    Task(
        name="task1",
        func=extract_chunks,
        inputs={"documents": None},  # None 表示从外部输入
        outputs=["chunks"]
    ),
    Task(
        name="task2",
        func=extract_graph,
        inputs={"chunks": "task1.outputs.chunks"},  # 显式依赖
        outputs=["graphs"]
    ),
]
```

**隐式传递**（参数插值）：

```python
task1 = Task(extract_chunks, documents=docs)
task2 = Task(
    extract_graph,
    chunks="${context.chunks}",  # 参数插值，Pipeline 执行时替换
    graph_model="${context.graph_model}"
)
```

**最佳实践**：

```python
# ✅ 好的做法：清晰的数据流
pipeline = Pipeline(
    tasks=[
        Task(
            func=step1,
            inputs={"data": input_data},
            outputs={"result": "step1_result"}
        ),
        Task(
            func=step2,
            inputs={"data": "${context.step1_result}"},
            outputs={"result": "step2_result"}
        ),
    ]
)

# ❌ 不好的做法：隐式依赖
def workflow():
    global_result = None
    
    def step1():
        global global_result
        global_result = process1()
    
    def step2():
        global global_result
        return process2(global_result)
    
    # 难以追踪数据流，不利于并行化和调试
```

### 5.3 状态管理与幂等性

**幂等性保证**：多次执行同一 Pipeline 应该产生相同结果。

```python
class IdempotentPipeline(Pipeline):
    """支持幂等性的 Pipeline"""
    
    async def execute(self) -> Dict[str, Any]:
        """执行，支持断点恢复"""
        
        # 从检查点恢复（如果存在）
        checkpoint = await self._load_checkpoint()
        if checkpoint:
            self.context._data.update(checkpoint["context"])
            start_task_idx = checkpoint["last_completed_task"] + 1
        else:
            start_task_idx = 0
        
        try:
            for idx in range(start_task_idx, len(self.tasks)):
                task = self.tasks[idx]
                
                # 检查是否已执行过（根据输入哈希）
                task_hash = self._hash_task_inputs(task)
                cached_result = await self._check_cache(task_hash)
                
                if cached_result is not None:
                    logger.info(f"使用缓存结果，跳过任务 {task.func.__name__}")
                    result = cached_result
                else:
                    # 执行任务
                    result = await self.executor.execute_task(task)
                    # 缓存结果
                    await self._cache_result(task_hash, result)
                
                # 保存到上下文
                self.context.set(task.func.__name__, result)
                
                # 保存检查点
                await self._save_checkpoint(idx, self.context._data)
            
            return self.results
        
        finally:
            # 清理检查点
            await self._cleanup_checkpoint()
    
    def _hash_task_inputs(self, task: Task) -> str:
        """计算任务输入的哈希"""
        import hashlib
        import json
        
        inputs = {
            "func": task.func.__name__,
            "args": str(task.args),
            "kwargs": json.dumps(task.kwargs, default=str, sort_keys=True)
        }
        
        return hashlib.sha256(
            json.dumps(inputs, sort_keys=True).encode()
        ).hexdigest()
```

**实际应用**：

```python
# .add 函数中的幂等性
# 每次 add 相同的文件，不会重复处理
content_hash = hashlib.sha256(file_content).hexdigest()
data_id = uuid5(NAMESPACE_OID, f"{content_hash}:{user.id}")

# 检查是否已存在（第一篇中介绍过）
existing = await session.query(Data).filter(Data.id == data_id).first()
if existing:
    logger.info("数据已存在，跳过处理")
    return existing
```

---

## 六、错误处理与恢复机制

### 6.1 异常捕获与分类

```python
class TaskError(Exception):
    """任务执行异常基类"""
    pass

class TaskTimeoutError(TaskError):
    """任务超时"""
    pass

class TaskRetryableError(TaskError):
    """可重试的异常"""
    pass

class TaskFatalError(TaskError):
    """不可恢复的异常"""
    pass

# 异常分类逻辑
class ErrorClassifier:
    """异常分类器"""
    
    RETRYABLE_EXCEPTIONS = (
        ConnectionError,      # 网络错误
        TimeoutError,         # 超时
        OSError,              # 文件系统错误
    )
    
    FATAL_EXCEPTIONS = (
        ValueError,           # 参数错误
        KeyError,             # 键错误
        TypeError,            # 类型错误
    )
    
    @staticmethod
    def classify(error: Exception) -> str:
        """分类异常"""
        if isinstance(error, ErrorClassifier.RETRYABLE_EXCEPTIONS):
            return "RETRYABLE"
        elif isinstance(error, ErrorClassifier.FATAL_EXCEPTIONS):
            return "FATAL"
        else:
            return "UNKNOWN"
```

### 6.2 重试策略

**指数退避重试**（已在 3.3 中介绍）：

```python
# 加强版：支持条件重试
class SmartRetryPolicy:
    """智能重试策略"""
    
    async def execute_with_retry(
        self,
        func,
        *args,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        on_retry: Callable = None,  # 重试前的回调
        should_retry: Callable[[Exception], bool] = None,  # 判断是否应重试
        **kwargs
    ):
        """执行，支持条件重试"""
        
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            
            except Exception as e:
                last_error = e
                
                # 检查是否应该重试
                if should_retry and not should_retry(e):
                    raise  # 不应重试，直接抛出
                
                if attempt >= max_retries:
                    raise  # 达到重试次数上限
                
                # 计算退避时间
                wait_time = backoff_factor ** attempt
                
                # 执行重试前的回调
                if on_retry:
                    await on_retry(e, attempt, wait_time)
                
                logger.warning(
                    f"任务失败，{wait_time:.1f}s 后重试 "
                    f"(尝试 {attempt + 1}/{max_retries + 1}): {e}"
                )
                
                await asyncio.sleep(wait_time)
        
        raise last_error
```

**在 Pipeline 中使用**：

```python
# 只重试网络错误
def should_retry(error):
    return isinstance(error, (ConnectionError, TimeoutError))

task = Task(
    extract_graph_from_data,
    chunks=chunks,
    retry_policy=SmartRetryPolicy(
        max_retries=3,
        should_retry=should_retry
    )
)
```

### 6.3 检查点与断点恢复

**检查点保存**：

```python
class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, checkpoint_dir: str = ".cognee_checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    async def save_checkpoint(
        self,
        pipeline_id: str,
        task_index: int,
        context: Context,
        metadata: dict = None
    ):
        """保存检查点"""
        checkpoint = {
            "pipeline_id": pipeline_id,
            "task_index": task_index,
            "context": context._data,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"{pipeline_id}_checkpoint_{task_index}.json"
        )
        
        # 原子性写入（先写临时文件再重命名）
        temp_path = checkpoint_path + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(checkpoint, f)
        
        os.rename(temp_path, checkpoint_path)
        logger.info(f"检查点已保存: {checkpoint_path}")
    
    async def load_checkpoint(
        self,
        pipeline_id: str,
        latest: bool = True
    ) -> Optional[dict]:
        """加载检查点"""
        checkpoints = glob.glob(
            os.path.join(self.checkpoint_dir, f"{pipeline_id}_checkpoint_*.json")
        )
        
        if not checkpoints:
            return None
        
        # 加载最新的检查点
        if latest:
            checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            checkpoint_path = checkpoints[-1]
        else:
            checkpoint_path = checkpoints[0]
        
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        
        logger.info(f"检查点已加载: {checkpoint_path}")
        return checkpoint
```

**断点恢复**：

```python
class ResumablePipeline(Pipeline):
    """支持断点恢复的 Pipeline"""
    
    async def execute(self, resume_from_checkpoint: bool = True):
        """执行，支持从检查点恢复"""
        
        checkpoint_mgr = CheckpointManager()
        
        # 尝试加载检查点
        if resume_from_checkpoint:
            checkpoint = await checkpoint_mgr.load_checkpoint(self.name)
            if checkpoint:
                logger.info(f"从检查点恢复 (任务 {checkpoint['task_index']})")
                self.context._data.update(checkpoint["context"])
                start_idx = checkpoint["task_index"] + 1
            else:
                start_idx = 0
        else:
            start_idx = 0
        
        # 从 start_idx 继续执行
        for idx in range(start_idx, len(self.tasks)):
            task = self.tasks[idx]
            
            try:
                result = await self.executor.execute_task(task)
                self.context.set(task.func.__name__, result)
                
                # 每个任务完成后保存检查点
                await checkpoint_mgr.save_checkpoint(
                    self.name,
                    idx,
                    self.context
                )
            
            except Exception as e:
                logger.error(f"任务 {idx} 失败: {e}, 检查点已保存")
                logger.info(f"下次可通过 resume_from_checkpoint=True 恢复")
                raise
        
        return self.results
```

### 6.4 回滚机制

**事务性 Pipeline**（部分任务失败时回滚）：

```python
class TransactionalPipeline(Pipeline):
    """支持回滚的事务性 Pipeline"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.undo_stack = []  # 记录可撤销的操作
    
    async def execute(self):
        """执行，失败时自动回滚"""
        
        try:
            for task in self.tasks:
                # 执行任务
                result = await self.executor.execute_task(task)
                self.context.set(task.func.__name__, result)
                
                # 记录可撤销操作
                if hasattr(task, 'undo_func'):
                    self.undo_stack.append(task.undo_func)
            
            return self.results
        
        except Exception as e:
            logger.error(f"Pipeline 执行失败，开始回滚: {e}")
            await self._rollback()
            raise
    
    async def _rollback(self):
        """回滚所有已执行的操作"""
        # 反向执行 undo 操作
        while self.undo_stack:
            undo_func = self.undo_stack.pop()
            try:
                result = undo_func()
                if asyncio.iscoroutine(result):
                    await result
                logger.info(f"已撤销操作: {undo_func.__name__}")
            except Exception as e:
                logger.error(f"撤销失败: {e}, 可能需要手动干预")
```

**示例：带回滚的数据摄取**

```python
async def add_with_rollback(data_list):
    """数据摄取，失败时回滚"""
    
    added_data_ids = []  # 记录已添加的数据 ID
    
    async def undo_add():
        """撤销已添加的数据"""
        for data_id in added_data_ids:
            await db.delete(Data, Data.id == data_id)
        logger.info(f"已删除 {len(added_data_ids)} 条数据")
    
    tasks = []
    for data in data_list:
        task = Task(
            store_data,
            data=data,
            undo_func=undo_add  # 绑定回滚函数
        )
        tasks.append(task)
    
    pipeline = TransactionalPipeline(tasks=tasks, name="add_data")
    return await pipeline.execute()
```

---

## 七、可观测性与监控

### 7.1 执行日志与追踪

```python
class PipelineLogger:
    """Pipeline 执行日志记录"""
    
    def __init__(self, log_file: str = None):
        self.log_file = log_file
        self.logs = []
    
    def log_event(
        self,
        event_type: str,  # "task_start", "task_complete", "task_failed"
        task_name: str,
        duration: float = None,
        result: Any = None,
        error: Exception = None
    ):
        """记录事件"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "task_name": task_name,
            "duration": duration,
            "result_type": type(result).__name__ if result else None,
            "error": str(error) if error else None
        }
        
        self.logs.append(event)
        
        # 格式化输出
        msg = f"[{event['timestamp']}] {event_type}: {task_name}"
        if duration:
            msg += f" ({duration:.2f}s)"
        if error:
            msg += f" - ERROR: {error}"
        
        logger.info(msg)
        
        # 持久化
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(event) + "\n")
```

**执行追踪（Tracing）**：

```python
class PipelineTracer:
    """分布式追踪"""
    
    def __init__(self, trace_id: str = None):
        self.trace_id = trace_id or str(uuid4())
        self.spans = []  # Span：单个操作的时间跨度
    
    def start_span(self, operation: str, task_id: str) -> "Span":
        """开始记录一个操作"""
        span = Span(
            trace_id=self.trace_id,
            operation=operation,
            task_id=task_id,
            start_time=time.time()
        )
        self.spans.append(span)
        return span
    
    def export_trace(self) -> dict:
        """导出追踪数据（兼容 OpenTelemetry）"""
        return {
            "trace_id": self.trace_id,
            "spans": [
                {
                    "operation": span.operation,
                    "start_time": span.start_time,
                    "duration": span.duration,
                    "attributes": span.attributes
                }
                for span in self.spans
            ]
        }

class Span:
    """单个操作的追踪"""
    
    def __init__(self, trace_id: str, operation: str, task_id: str, start_time: float):
        self.trace_id = trace_id
        self.operation = operation
        self.task_id = task_id
        self.start_time = start_time
        self.end_time = None
        self.attributes = {}
    
    def set_attribute(self, key: str, value: Any):
        """设置属性"""
        self.attributes[key] = value
    
    def end(self):
        """结束记录"""
        self.end_time = time.time()
    
    @property
    def duration(self) -> float:
        """获取持续时间"""
        return (self.end_time or time.time()) - self.start_time
```

### 7.2 性能指标收集

```python
class MetricsCollector:
    """性能指标收集"""
    
    def __init__(self):
        self.metrics = {}
    
    def record_task_metric(
        self,
        task_name: str,
        duration: float,
        status: str,
        input_size: int = None,
        output_size: int = None
    ):
        """记录任务指标"""
        if task_name not in self.metrics:
            self.metrics[task_name] = {
                "count": 0,
                "total_duration": 0.0,
                "min_duration": float('inf'),
                "max_duration": 0.0,
                "errors": 0,
                "avg_input_size": 0,
                "avg_output_size": 0
            }
        
        m = self.metrics[task_name]
        m["count"] += 1
        m["total_duration"] += duration
        m["min_duration"] = min(m["min_duration"], duration)
        m["max_duration"] = max(m["max_duration"], duration)
        
        if status == "failed":
            m["errors"] += 1
        
        if input_size:
            m["avg_input_size"] = (m["avg_input_size"] * (m["count"] - 1) + input_size) / m["count"]
        if output_size:
            m["avg_output_size"] = (m["avg_output_size"] * (m["count"] - 1) + output_size) / m["count"]
    
    def get_summary(self) -> dict:
        """获取统计摘要"""
        summary = {}
        for task_name, m in self.metrics.items():
            summary[task_name] = {
                "executions": m["count"],
                "avg_duration": m["total_duration"] / m["count"],
                "min_duration": m["min_duration"],
                "max_duration": m["max_duration"],
                "error_rate": m["errors"] / m["count"],
                "throughput": m["count"] / m["total_duration"]  # 每秒任务数
            }
        return summary
```

### 7.3 可视化与调试工具

**Pipeline 执行时间线可视化**：

```python
def visualize_pipeline_timeline(tracer: PipelineTracer) -> str:
    """生成 Pipeline 执行时间线"""
    
    timeline = "Pipeline Execution Timeline:\n"
    timeline += "=" * 60 + "\n"
    
    # 按开始时间排序
    sorted_spans = sorted(tracer.spans, key=lambda s: s.start_time)
    
    min_time = sorted_spans[0].start_time
    
    for span in sorted_spans:
        relative_start = span.start_time - min_time
        duration = span.duration
        
        # ASCII 进度条
        bar_length = int(duration * 100)
        bar = "█" * bar_length
        
        timeline += f"{span.operation:30} |{bar:50}| {duration:.2f}s\n"
    
    total_duration = (sorted_spans[-1].end_time or time.time()) - min_time
    timeline += "=" * 60 + "\n"
    timeline += f"Total Duration: {total_duration:.2f}s\n"
    
    return timeline
```

**调试模式**：

```python
class DebugPipeline(Pipeline):
    """调试模式 Pipeline"""
    
    def __init__(self, *args, debug: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = debug
    
    async def execute(self):
        """执行，调试模式下显示详细信息"""
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
            logger.info("启用调试模式")
        
        for idx, task in enumerate(self.tasks):
            if self.debug:
                logger.debug(f"[Task {idx}] 准备执行: {task.func.__name__}")
                logger.debug(f"  参数: {task.kwargs}")
                logger.debug(f"  上下文: {self.context.list_keys()}")
            
            try:
                result = await self.executor.execute_task(task)
                self.context.set(task.func.__name__, result)
                
                if self.debug:
                    logger.debug(f"[Task {idx}] 执行成功")
                    logger.debug(f"  结果类型: {type(result).__name__}")
                    if isinstance(result, (list, dict)):
                        logger.debug(f"  结果大小: {len(result)}")
            
            except Exception as e:
                if self.debug:
                    logger.exception(f"[Task {idx}] 执行失败")
                raise
        
        return self.results
```

---

## 八、自定义与扩展指南

### 8.1 编写自定义 Task

**基础自定义 Task**：

```python
# 方式 1：函数直接转换为 Task
async def my_custom_processing(data: list, threshold: float) -> dict:
    """自定义处理函数"""
    results = {}
    for item in data:
        if item.value > threshold:
            results[item.id] = process(item)
    return results

# 使用
task = Task(
    my_custom_processing,
    data=input_data,
    threshold=0.5
)

# 方式 2：继承 Task 基类（支持更复杂的行为）
class CustomTransformTask(Task):
    """自定义转换任务"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform_type = kwargs.pop("transform_type", "default")
    
    async def execute(self, executor) -> Any:
        """任务执行逻辑"""
        logger.info(f"执行自定义转换: {self.transform_type}")
        
        # 预处理
        data = self.kwargs.get("data")
        data = self._preprocess(data)
        
        # 执行
        result = await self.func(data, **self.kwargs)
        
        # 后处理
        result = self._postprocess(result)
        
        return result
    
    def _preprocess(self, data: list) -> list:
        """数据预处理"""
        return [item for item in data if self._validate(item)]
    
    def _postprocess(self, result: dict) -> dict:
        """结果后处理"""
        return {k: v for k, v in result.items() if self._filter(v)}
    
    @staticmethod
    def _validate(item) -> bool:
        """项目验证"""
        return hasattr(item, 'id') and hasattr(item, 'value')
    
    @staticmethod
    def _filter(value) -> bool:
        """结果过滤"""
        return value is not None
```

**高级 Task：支持流式处理**：

```python
class StreamingTask(Task):
    """流式处理任务"""
    
    async def execute_stream(self, executor):
        """流式执行，逐项产生结果"""
        data = self.kwargs.get("data")
        
        async def stream_generator():
            for item in data:
                result = await self.func(item, **{
                    k: v for k, v in self.kwargs.items() if k != "data"
                })
                yield result
        
        return stream_generator()

# 使用示例
async def process_item(item, multiply: int = 1):
    """异步处理单项"""
    await asyncio.sleep(0.1)  # 模拟 I/O 操作
    return item.value * multiply

task = StreamingTask(
    process_item,
    data=large_dataset,
    multiply=2
)

# 流式消费结果
async for result in await task.execute_stream(executor):
    print(result)
```

### 8.2 编写自定义 Executor

**简单的 Executor：并发度控制**：

```python
class LimitedConcurrencyExecutor(Executor):
    """受限并发执行器"""
    
    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_tasks = []
    
    async def execute_task(self, task: Task) -> Any:
        """执行单个任务"""
        async with self.semaphore:
            logger.info(f"执行任务: {task.func.__name__} "
                       f"(活跃: {len(self.active_tasks) + 1}/{self.max_concurrent})")
            
            try:
                result = await self._run_task(task)
                return result
            
            except Exception as e:
                logger.error(f"任务失败: {e}")
                raise
    
    async def _run_task(self, task: Task) -> Any:
        """实际执行函数"""
        task.status = TaskStatus.RUNNING
        start = time.time()
        
        try:
            if asyncio.iscoroutinefunction(task.func):
                result = await task.func(*task.args, **task.kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, task.func, *task.args
                )
            
            task.status = TaskStatus.COMPLETED
            duration = time.time() - start
            logger.info(f"任务完成: {task.func.__name__} ({duration:.2f}s)")
            
            return result
        
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = e
            raise
```

**分布式 Executor（与 Ray 集成）**：

```python
import ray

class RayDistributedExecutor(Executor):
    """基于 Ray 的分布式执行器"""
    
    def __init__(self, num_actors: int = 4):
        if not ray.is_initialized():
            ray.init()
        
        self.num_actors = num_actors
        self.actor_pool = [
            ray.remote(TaskWorker).remote()
            for _ in range(num_actors)
        ]
    
    async def execute_task(self, task: Task) -> Any:
        """通过 Ray 远程执行"""
        # 将任务序列化
        task_data = {
            "func": task.func,
            "args": task.args,
            "kwargs": task.kwargs
        }
        
        # 选择一个 actor
        actor = self.actor_pool[hash(task) % self.num_actors]
        
        # 远程执行
        future = actor.execute.remote(task_data)
        result = ray.get(future)
        
        return result

@ray.remote
class TaskWorker:
    """Ray actor：任务工作进程"""
    
    async def execute(self, task_data: dict) -> Any:
        """执行任务"""
        func = task_data["func"]
        args = task_data["args"]
        kwargs = task_data["kwargs"]
        
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
```

### 8.3 集成第三方工作流引擎

**与 Apache Airflow 集成**：

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def cognee_to_airflow(pipeline: Pipeline) -> DAG:
    """将 Cognee Pipeline 转换为 Airflow DAG"""
    
    dag = DAG(
        pipeline.name,
        start_date=datetime(2025, 1, 1),
        schedule_interval=None
    )
    
    task_operators = {}
    
    # 为每个 Task 创建 Airflow Operator
    for idx, task in enumerate(pipeline.tasks):
        
        def run_task(task=task, **context):
            """Airflow 任务函数"""
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(pipeline.executor.execute_task(task))
        
        op = PythonOperator(
            task_id=task.func.__name__,
            python_callable=run_task,
            dag=dag
        )
        
        task_operators[task.func.__name__] = op
    
    # 设置任务依赖
    # （这里需要根据 pipeline 的依赖关系设置）
    
    return dag

# 使用
cognify_pipeline = Pipeline(
    tasks=[...],
    name="cognify_pipeline"
)

airflow_dag = cognee_to_airflow(cognify_pipeline)
```

### 8.4 最佳实践

**1. 任务颗粒度**：

```python
# ❌ 太粗：任务内部没有细粒度控制
task = Task(
    entire_cognify_pipeline,
    dataset=dataset_id
)

# ✅ 合适：每个任务单一职责
tasks = [
    Task(classify_documents, documents=docs),
    Task(extract_chunks_from_documents, documents=docs),
    Task(extract_graph_from_data, chunks=chunks),
    Task(summarize_text, chunks=chunks),
    Task(add_data_points, graphs=graphs, summaries=summaries)
]
```

**2. 错误处理**：

```python
# ❌ 无视异常
async def process():
    await task.execute()

# ✅ 处理异常
async def process():
    try:
        await task.execute()
    except TaskRetryableError as e:
        logger.warning(f"可重试错误: {e}")
        # 触发重试
    except TaskFatalError as e:
        logger.error(f"不可恢复错误: {e}")
        # 中止 Pipeline
        raise
```

**3. 参数管理**：

```python
# ❌ 硬编码参数
task = Task(process, max_size=2048, timeout=300)

# ✅ 从配置读取
config = load_config("cognee.yaml")
task = Task(
    process,
    max_size=config.get("max_chunk_size", 2048),
    timeout=config.get("task_timeout", 300)
)
```

**4. 可观测性**：

```python
# 使用日志记录所有重要操作
logger.info(f"Pipeline 开始: {pipeline.name}")
for task in pipeline.tasks:
    logger.debug(f"Task 参数: {task.kwargs}")

# 记录指标
metrics = MetricsCollector()
metrics.record_task_metric(task_name, duration, status)
```

---

## 尾言

Pipeline 机制是 Cognee 能够灵活、可靠、高效地处理复杂 AI 工作流的基础。通过理解和掌握这一机制，你可以：

- **高效开发**：快速原型化新功能
- **可靠运维**：构建健壮的生产系统
- **性能优化**：根据场景选择最优的执行策略
- **社区贡献**：开发高质量的扩展和插件

期待你在 Cognee 的生态中发挥更大的作用！