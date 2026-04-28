---
title: "Memory Engineering：如何把 Agent 的长期记忆真正做出来"
date: 2026-04-28T11:50:10+08:00
draft: true
tags: ["技术","Memory","Agent"]
categories: ["Agent"]
---

前三篇其实已经把“为什么”讲得差不多了。

第一篇《为什么 AI Agent 的长期记忆几乎都是错的？》讲的是现象：为什么很多 Agent 明明已经接了 memory，实际表现还是不稳定。
第二篇《Memory OS：AI Agent 不是缺记忆，而是缺一套记忆系统》讲的是系统边界：长期记忆不是“聊天记录 + 检索”，而是一条从写入、整合到重建的完整链路。
第三篇《为什么 Memory 不能只是 KV？——从结构化记忆到 HyperMem 的关键一步》继续往下拆，讨论了为什么 memory 不能只是 KV，为什么一旦进入多跳、时序、状态冲突场景，系统最终一定会走向结构化记忆。EverMemOS 把这条链路概括成 Episodic Trace Formation、Semantic Consolidation 和 Reconstructive Recollection；HyperMem 则进一步把记忆组织成 topic、event、fact 的层级结构。

但讲完这些，还只是把问题讲明白了。真正决定系统成败的，往往不是概念，而是工程实现。

也就是说，接下来真正值得回答的问题已经不是“长期记忆应不应该做”，而是：

> **如果真的要从 0 到 1 搭一套可用的 Agent Memory，工程上到底该怎么落？**

这篇文章就沿着这个问题往下写。而且这次不只讲论文思路，也会结合 EverMind 开源的 EverOS 仓库来看。因为这个仓库本身就是前几篇论文在工程层面的落地实践：README 直接把仓库定义成一个统一的平台，用来 build、evaluate、integrate long-term memory，并把 EverCore（long-term memory operating system）和 HyperMem（hypergraph memory architecture）并列放在 methods 下。

---

# 一、先别急着选技术，先把系统边界定清楚

很多长期记忆项目最后做崩，不是因为模型不够强，也不是因为数据库选错了，而是因为系统边界一开始就没定清。

一个真正的长期记忆系统，负责的应该是这几件事：

* 把原始交互写成可维护的记忆对象
* 把这些对象分层组织成事件、事实、状态、画像
* 在查询到来时，按当前问题重建“必要且足够”的上下文
* 在时间推移中持续更新、降权、覆盖、过期管理这些记忆对象

而它**不负责**替代大模型本身的推理能力，也不负责把所有历史无损压进上下文，更不负责解决全部世界知识问题。EverMemOS 论文的核心主张也是这个：它要解决的不是“把更多历史存起来”，而是把碎片化经验转成稳定语义结构，并在回答时只重建必要且充分的 grounded context。

如果没有这个边界意识，项目很容易滑向一种常见误区：把 memory 当成一个无所不能的“外挂数据库”，结果写入、检索、状态管理、上下文拼装全混在一起，最后越来越重，越来越难维护。

所以第四篇一开始最想强调的一句话是：

> **长期记忆系统不是大模型旁边再挂一个库，而是一个面向长期状态与上下文重建的专用运行时。**

---

# 二、先解释四个后面会反复出现的概念

在继续往下讲之前，先把几个会反复出现的术语说清楚。不然后面读起来会有一点跳。

## 1. MemCell

**MemCell** 可以理解为长期记忆系统里的基础记忆单元。
它不是原始聊天记录，而是从一段对话中整理出来、可被后续系统继续加工和维护的“记忆中间对象”。EverMemOS 论文把它定义为 `(Episode, Facts, Foresight, Metadata)` 组成的基础 primitive；EverOS 代码里也把 MemCell 放在独立的 memcell extractor 链路里，而不是直接把消息写入长期记忆。

可以把它理解成：**聊天记录的毛坯记忆。**

## 2. MemScene

**MemScene** 可以理解为更高一层的“记忆场景”或“主题化记忆包”。
如果说 MemCell 还是单个事件片段，那么 MemScene 更像把多个相关记忆单元整合之后形成的语义场景。EverMemOS 在论文中把这一步放在 Semantic Consolidation 阶段，强调要把 episodic traces 进一步组织成更稳定的 semantic structure。

可以把它理解成：**整理后的记忆场景。**

## 3. User Profile

**User Profile** 是用户画像层。
它不是在记录“发生过什么”，而是在沉淀“这是一个怎样的人”：偏好、习惯、长期目标、稳定状态，这些更适合落到 Profile 上，而不是落到事件日志里。EverOS 的主仓库 README 明确把 EverCore 描述成会“extract, structure, and retrieve long-term knowledge from conversations”，而代码里也把 PROFILE 当成独立 memory type 处理。

可以把它理解成：**长期稳定用户画像。**

## 4. Foresight

**Foresight** 可以理解为前瞻性记忆。
它保存的是“未来一段时间内仍然会影响判断”的计划、限制和潜在状态，比如“这两周在吃药”“最近在备考”“下个月准备跳槽”。EverMemOS 论文把它作为 MemCell 的组成部分之一，EverOS 代码里也有独立的 `foresight_extractor.py` 来生成这类对象。

可以把它理解成：**短中期有效状态。**

如果用更直白的话概括：

> **MemCell 像毛坯记忆，MemScene 像整理后的场景，Profile 像长期画像，Foresight 像短中期有效状态。**

---

# 三、真正的工程起点，不是检索，而是写入

很多团队做 memory 时，天然会把注意力放在 retrieval 上：向量召回怎么做、rerank 怎么调、top-k 取多少、混合检索怎么配。可从长期记忆的视角看，真正的起点其实不是 retrieval，而是 write path。

因为只要写入阶段错了，后面所有东西都会跟着错。

EverOS 代码里，这一点体现得非常明显。`memory_manager.py` 把系统主入口清楚地拆成两条路径：一条是 `memorize`，一条是 `retrieve_mem`。前者负责把原始数据异步写入 memory，后者才是读取。也就是说，它在架构上先承认了一件事：**长期记忆首先是一个“写入系统”，然后才是一个“检索系统”。** 

更关键的是，EverOS 并没有把原始消息直接当成长记忆存下来。`conv_memcell_extractor.py` 的文件注释和实现都表明，这个组件负责 boundary detection 和 basic MemCell creation，会先把历史消息与新消息合并，再做边界判断，然后决定哪里切成一个可被系统后续维护的单元。代码里还存在 `should_wait` 这类机制，说明它不是一来一条消息就硬切，而是在判断“当前这段内容是不是已经形成了一个可写入的记忆对象”。

这件事特别重要。因为它意味着工程上第一步不是“把所有对话扔进库”，而是：

> **先把对话写成系统能维护的对象。**

从 Memory Engineering 的角度看，这一步才是长期记忆系统真正的地基。
如果地基还是原始聊天记录，那么后面的 profile、constraint、graph、hypergraph，最后都会退化成“围绕日志做复杂检索”。

---

# 四、记忆对象一定要拆，不要幻想一个 memory bucket 解决全部问题

第二篇已经讲过，长期记忆系统不该只有一种记忆对象。到了工程层，这个原则反而更重要，因为不同 memory type 的生命周期、更新逻辑、检索逻辑根本不一样。

EverOS 的代码正好给了一个很具体的参照。

在论文里，EverMemOS 把核心 primitive 定成了 MemCell、MemScene、User Profile、Foresight。到了代码里，这套抽象被拆成了更可工程化的对象层：MemCell、EpisodeMemory、AtomicFact / EventLog、Profile、Foresight 等分别由不同 extractor 和 repository 管理。`episode_memory_extractor.py` 的职责就写得很明确：它负责从 MemCell 继续抽取 Episode memory，而不是顺手把一切都混在一起。

这个设计很像什么？很像数据库建模里的“先分实体，再谈查询”。
因为如果不先把 memory types 拆开，后面几乎所有工程动作都会失控：

* 检索时不知道是该先找 episode 还是先找 constraint
* 更新时不知道新信息是在覆盖旧状态，还是补充旧状态
* 生命周期管理时不知道哪些内容该长期保留，哪些内容应该过期降权

在 `memory_manager.py` 里，这种拆分已经进入了读取链路本身。代码不是把所有 memory types 一视同仁地检索，而是先把 `PROFILE` 单独拆出来，再把非 profile 类型走 keyword、vector、hybrid 或 agentic 的检索路径。也就是说，EverOS 在工程实现上已经接受了一个很重要的事实：**画像检索和事件检索，本来就不是同一个问题。**

如果把这件事翻译成更通俗的工程建议，就是：

> **不要做一个“大 memory 表”，而要做一组生命周期不同、读取策略不同的 memory objects。**

---

# 五、存储层天然是多存储协作问题，而不是单库问题

一旦 memory objects 拆开，存储层的现实问题就马上出来了：一种存储通常不够。

原因很简单。长期记忆系统同时需要几种完全不同的能力：

* 需要保存原始和结构化对象
* 需要做语义召回
* 需要做关键词/全文补充
* 需要缓存热状态
* 如果继续往结构层走，还需要表达对象关系

EverOS 虽然在 README 里没有直接画一张“多存储架构图”，但代码已经把这个事实暴露得很明显了：`memory_manager.py` 同时引入了 raw repository，也引入了 `EpisodicMemoryEsRepository`、`ForesightEsRepository`、`AtomicFactEsRepository` 这些全文检索仓库，以及对应的 `EpisodicMemoryMilvusRepository`、`ForesightMilvusRepository`、`AtomicFactMilvusRepository` 这些向量检索仓库。换句话说，这个系统的工程默认值就是：**对象存储、全文检索、向量召回不是一回事。**

这其实很有代表性，因为它说明长期记忆系统从一开始就不该被设计成“只选 Mongo 还是只选向量库”的单选题。更现实的做法通常是：

* 文档库负责 memory object 本体
* 向量库负责 semantic recall
* 全文索引负责 keyword / exact cue recall
* 缓存层负责热点状态和短期 session context
* 更强结构层再逐步往图关系或 hypergraph 演进

这也是为什么前几篇里一直强调 hybrid retrieval。不是因为 hybrid 很时髦，而是因为长期记忆里的 query 类型天生就很多样：有些更像语义召回，有些更像精确线索，有些则需要 profile / constraint 先行。

所以从工程落地角度说：

> **长期记忆系统天然是多存储协作问题，而不是单库问题。**

---

# 六、读取链路的难点，从来都不是“召回更多”，而是“重建得更准”

很多 memory 系统做到中期都会出现一个幻觉：好像只要把 recall 做得更多一点，就能把答案补全。

现实通常相反。长期记忆的主要问题不是召回不够，而是噪声太多、层次混乱、上下文拼装不对。

EverMemOS 论文里把这一阶段叫 Reconstructive Recollection，核心不再是 retrieve everything relevant，而是 compose only the necessary and sufficient context。论文还明确提到 query rewrite、scene match、episode rerank、sufficiency check 这类动作，说明它不是一次性 top-k，而是围绕“够不够回答当前问题”做闭环式重建。

EverOS 的代码也明显在往这个方向落。`MemoryManager.retrieve_mem` 会根据 `retrieve_method` 走 keyword、vector、hybrid、agentic 等不同路径；而导入和调用链路里又能看到 `check_sufficiency`、多 query 生成等能力，说明这个系统已经不满足于“给一个 query 只查一次”，而是在尝试把多轮召回、充分性判断、query expansion 纳入正式检索链路。

这背后真正重要的工程思想其实是：

* 先找状态
* 再找相关事件
* 必要时再下钻事实
* 最后把它们重建成回答当前问题所需的最小上下文

而不是一开始就把一堆文本全部塞回去。

如果把这一步做对，很多“模型不够聪明”的问题会立刻下降。
如果这一步没做对，再强的模型也只是在噪声里硬撑。

所以读取链路真正要优化的，不是“top-k 再往上调多少”，而是：

> **系统能不能把当前问题真正需要的上下文重建出来。**

---

# 七、状态更新才是长期记忆最难、也最容易被低估的部分

做过长期 memory 的人，最后几乎都会走到这个问题上：**新增一条记忆并不难，难的是新信息进来之后，旧状态怎么办。**

这是长期记忆系统和普通知识库最大的差别之一。
知识库更关心“有没有这条事实”；
长期记忆系统更关心“这条事实现在还算不算有效，它和旧状态是什么关系”。

EverMemOS 论文里最经典的例子就是 IPA 和抗生素：用户长期喜欢 IPA 是真的，但“最近在吃药，不适合喝酒”在当前时刻优先级更高。如果系统只会存 facts，不会维护当前状态，那么答案很容易看起来相关，却明显不对。

EverOS 代码之所以把 `PROFILE` 独立处理，又把 `ForesightExtractor` 做成专门组件，本质上就是在工程上承认：

* 长期偏好
* 当前约束
* 短期计划
* 未来一段时间内有效的状态

本来就不应该混成一层。`foresight_extractor.py` 的实现说明里也明确写着，它要生成对“用户未来生活与决策潜在影响”的预测，并把这类结果变成独立 memory 对象。

这件事对工程落地的启发很直接：

1. 不要轻易直接覆盖旧状态
2. 要保留状态变更轨迹和来源证据
3. 尽量给 constraint / foresight 增加时间边界
4. 能拆 profile 和 constraint，就不要把它们混存

因为长期记忆真正难的部分，不是“存下来”，而是“持续演化还不乱”。

---

# 八、结构层不用一步到位，但一定要给它留出进化空间

到这里，其实已经能看出一个比较现实的工程路线了。

很多人看完 HyperMem 会很兴奋，恨不得一上来就做 topic / episode / fact 全套结构、关系图、甚至 hypergraph。但从工程角度看，最稳妥的办法通常不是一步到位，而是分阶段演进。

EverOS 本身就是一个例子。README 里已经把 HyperMem 作为正式方法收编进去，同时又把 EverCore 作为 production-ready 的 long-term memory operating system 单独提供；也就是说，这个仓库本身就在表达一种很现实的路线：**先把 Memory OS 跑起来，再逐步把更强的结构方法纳入同一平台。** README 还明确写到，HyperMem 是 “A hypergraph-based hierarchical memory architecture that captures high-order associations through hyperedges”，并强调它把 memory 组织成 topic、event、fact 三层。

所以如果把第四篇压缩成一句最接地气的建议，大概会是这样：

* 第一阶段先把写入链路做对
* 第二阶段把 memory types 拆开，状态层独立
* 第三阶段把 hybrid / agentic retrieval 跑顺
* 第四阶段再逐步加强对象之间的关系表达，往 graph / hypergraph 走

这条路线看起来没那么“论文感”，但更像真正能活下来的工程方案。

因为长期记忆系统最怕的，不是做得不够先进，而是还没形成闭环，就先把自己做得过重、过杂、过难演进。

---

# 九、一个现实可用的 MVP，应该长什么样

如果现在就从 0 到 1 搭一个长期记忆系统，最现实的起点不是“直接复刻全部论文”，而是先做出一个最小闭环：

```text id="memory-mvp-flow"
Chat Input
→ Conversation Buffer
→ MemCell / Episode 抽取
→ Fact / Profile / Constraint 候选生成
→ 文档存储 + 向量索引 + 全文索引
→ Hybrid / Agentic Retrieval
→ Context Rebuilder
→ LLM Response
```

这套闭环里，最值得优先做好的不是花哨的 graph UI，也不是特别重的 multi-hop planner，而是三件基础但决定上限的事：

* **写对**：不要直接存原始聊天
* **分层**：不要把 profile、event、constraint 混成一锅
* **取对**：不要把 retrieval 简化成一次 top-k

EverOS 的主链路，基本就是按这个顺序在工程里落下来的：先有 `memorize`，再有 MemCell 边界检测，再有 Episode / Foresight 这些 extractor，再到多路检索和 profile 独立搜索。它还没有把“长期记忆的终局”一次做完，但已经把最关键的基础工程框架搭出来了。

这也是这篇文章最想传达的一个现实判断：

> **Memory Engineering 的起点，不是最先进，而是最小闭环。**

---

# 结尾：真正改变 Agent 形态的，不是一个 memory 模块，而是一整套运行方式

写到这里，再回头看前三篇，其实会发现一个很明显的变化：

一开始讨论的是“Agent 为什么会忘”。
后来讨论的是“Memory System 应该怎么设计”。
再往下，变成“结构为什么重要”。
到了这一篇，问题终于落到最现实的层面：**怎样把这些东西做成一个可运行、可迭代、可控成本的系统。**

这也是 Memory Engineering 真正有意思的地方。

它改变的不是一个外挂模块，也不是在大模型旁边多接一个向量库。它真正改变的，是 Agent 的整体运行方式：从一次性流程，变成一个能够持续形成状态、维护关系、累积用户模型并逐步演化的系统。EverOS 的 README 也正是用 “persistent memory you can actually see and feel” 来描述这种变化：它不是让 Agent 多记几句话，而是让 Agent 真的开始“带着记忆活着”。

没有长期记忆时，Agent 更像一个会说话的工具。
开始拥有长期记忆之后，Agent 才慢慢像一个“持续存在的系统”。

而这，才是长期记忆工程真正值得做的原因。

---

# 引用

[1]: https://arxiv.org/pdf/2601.02163 "EverMemOS: A Self-Organizing Memory Operating System for Structured Long-Horizon Reasoning"
[2]: https://github.com/EverMind-AI/EverOS "GitHub - EverMind-AI/EverOS: Build, evaluate, and integrate long-term memory for self-evolving agents. · GitHub"
[3]: https://github.com/EverMind-AI/EverOS/blob/main/methods/evermemos/src/agentic_layer/memory_manager.py "EverOS/methods/evermemos/src/agentic_layer/memory_manager.py at main · EverMind-AI/EverOS · GitHub"
[4]: https://github.com/EverMind-AI/EverOS/blob/main/methods/evermemos/src/memory_layer/memcell_extractor/conv_memcell_extractor.py "EverOS/methods/evermemos/src/memory_layer/memcell_extractor/conv_memcell_extractor.py at main · EverMind-AI/EverOS · GitHub"
[5]: https://github.com/EverMind-AI/EverOS/blob/main/methods/evermemos/src/memory_layer/memory_extractor/episode_memory_extractor.py "EverOS/methods/evermemos/src/memory_layer/memory_extractor/episode_memory_extractor.py at main · EverMind-AI/EverOS · GitHub"
[6]: https://github.com/EverMind-AI/EverOS/blob/main/methods/evermemos/src/memory_layer/memory_extractor/foresight_extractor.py "EverOS/methods/evermemos/src/memory_layer/memory_extractor/foresight_extractor.py at main · EverMind-AI/EverOS · GitHub"
[7]: https://arxiv.org/pdf/2604.08256 "HyperMem: Hypergraph Memory for Long-Term Conversations"
