---
title: "【解密源码】WeKnora RAG 检索与重排解析：生产级系统如何筛选可用 Chunk "
date: 2026-01-22T11:00:10+08:00
draft: false
tags: ["源码","技术","RAG"]
categories: ["RAG"]
---

# 引言

在 RAG 系统中，**Chunk 的质量只解决了一半问题**。

另一半更棘手的问题是：  
> **在一堆“看起来都相关”的 Chunk 里，究竟哪些才值得被送进 LLM？**

真实工程场景中，检索阶段面临的从来不是“有没有结果”，而是：
- 召回的 Chunk 数量过多，噪声极高  
- 不同来源（向量、关键词、FAQ、Web）的结果难以统一  
- rerank 一旦阈值设高，直接“全军覆没”  
- 阈值设低，又会把大量低价值 Chunk 一起塞进上下文  
- 多段 Chunk 来自同一文档，却彼此高度重复  

这些问题，并不能靠一个更大的模型解决。

在上一篇文章中，我们已经分析了 **WeKnora 如何构建结构完整、语义稳定的高质量 Chunk**。而本文关注的是**紧随其后的关键一步**：  
**当 Chunk 已经准备好之后，WeKnora 是如何在检索与重排阶段，筛选出“真正可用”的那一小部分？**

本文将基于源码，重点拆解 WeKnora 在 RAG 查询过程中围绕以下问题的工程实现：

- 多路召回（向量 / 关键词 / FAQ / Web）如何并行执行并统一结果
- rerank 在“选不出结果”时如何优雅降级
- FAQ、历史引用为何拥有更高优先级
- MMR 与位置先验是如何减少重复 Chunk 的
- Chunk 如何从“检索结果”一步步演化为最终上下文

本文会**专注于检索、重排与 Chunk 构建这一条最影响 RAG 实际效果的工程链路**。


# Pipline
WeKnora 检索召回采用 Pipline 的设计，在 WeKnora 中，一共包含以下 4 个预定义的 Pipline。
```go
var Pipline = map[string][]EventType{
	"chat": { // Simple chat without retrieval
		CHAT_COMPLETION,
	},
	"chat_stream": { // Streaming chat without retrieval (no history)
		CHAT_COMPLETION_STREAM,
		STREAM_FILTER,
	},
	"chat_history_stream": { // Streaming chat with conversation history
		LOAD_HISTORY,
		CHAT_COMPLETION_STREAM,
		STREAM_FILTER,
	},
	"rag": { // Retrieval Augmented Generation
		CHUNK_SEARCH,
		CHUNK_RERANK,
		CHUNK_MERGE,
		INTO_CHAT_MESSAGE,
		CHAT_COMPLETION,
	},
	"rag_stream": { // Streaming Retrieval Augmented Generation
		REWRITE_QUERY,
		CHUNK_SEARCH_PARALLEL, // Parallel: CHUNK_SEARCH + ENTITY_SEARCH
		CHUNK_RERANK,
		CHUNK_MERGE,
		FILTER_TOP_K,
		DATA_ANALYSIS,
		INTO_CHAT_MESSAGE,
		CHAT_COMPLETION_STREAM,
		STREAM_FILTER,
	},
}
```
以及一些自由组合的 Pipline，如：/knowledge-search 接口对应的实现纯检索的 Pipline 为：
```go
searchEvents := []types.EventType{
	types.CHUNK_SEARCH, // Vector search
	types.CHUNK_RERANK, // Rerank search results
	types.CHUNK_MERGE,  // Merge search results
	types.FILTER_TOP_K, // Filter top K results
}
```
通过预定义的 Pipline 可以看出 Weknora 将整个传统 RAG 中的 chat retrieval 过程拆分成了多个步骤。
- CHAT_COMPLETION 普通聊天
- CHAT_COMPLETION_STREAM 流式聊天
- STREAM_FILTER 流式过滤
- LOAD_HISTORY 加载历史记录
- REWRITE_QUERY 查询重写
- CHUNK_SEARCH_PARALLEL 多路检索（文档检索 + 实体检索）
- CHUNK_RERANK 检索重排
- CHUNK_MERGE 合并结果
- FILTER_TOP_K 过滤 top K 结果
- DATA_ANALYSIS 数据分析
- INTO_CHAT_MESSAGE 构建最终消息体

Chat 的相关步骤这里不做拆解，**重点拆解 rag_stream Pipline 中的所有步骤。**

# 查询重写 - REWRITE_QUERY

查询重写事件会触发两个插件：
- PluginRewrite 插件，对问题结合会话上下文进行重写
```go
func (p *PluginRewrite) ActivationEvents() []types.EventType {
	return []types.EventType{types.REWRITE_QUERY}
}
```
- PluginExtractEntity 插件，对问题中的实体进行提取
```go
func (p *PluginExtractEntity) ActivationEvents() []types.EventType {
	return []types.EventType{types.REWRITE_QUERY}
}
```
## PluginRewrite 插件
触发条件为 EnableRewrite 为 true, 否则直接跳过。默认配置在 `config.yaml` 中，默认值为 true。
```go
if !chatManage.EnableRewrite {
    pipelineInfo(ctx, "Rewrite", "skip", map[string]interface{}{
        "session_id": chatManage.SessionID,
        "reason":     "rewrite_disabled",
    })
    return next()
}
```
根据 seesion id 获取历史对话记录。
```go
history, err := p.messageService.GetRecentMessagesBySession(ctx, chatManage.SessionID, 20)
```
若该 session id 没有历史对话记录，则直接跳过查询重写。
```go
if len(historyList) == 0 {
    pipelineInfo(ctx, "Rewrite", "skip", map[string]interface{}{
        "session_id": chatManage.SessionID,
        "reason":     "empty_history",
    })
    return next()
}
```
若存在历史对话记录，则对历史对话记录做处理，如：消息格式化，清除无效对话，按时间排序，保留指定轮数对话记录等。
```go
// 格式化对话记录
for _, message := range history {
    history, ok := historyMap[message.RequestID]
    if !ok {
        history = &types.History{}
    }
    if message.Role == "user" {
        // User message as query
        history.Query = message.Content
        history.CreateAt = message.CreatedAt
    } else {
        // System message as answer, while removing thinking process
        history.Answer = reg.ReplaceAllString(message.Content, "")
        history.KnowledgeReferences = message.KnowledgeReferences
    }
    historyMap[message.RequestID] = history
}
// 清除无效对话记录
for _, history := range historyMap {
    if history.Answer != "" && history.Query != "" {
        historyList = append(historyList, history)
    }
}
// 按时间排序，保留最近的对话记录
sort.Slice(historyList, func(i, j int) bool {
    return historyList[i].CreateAt.After(historyList[j].CreateAt)
})
// 保留指定轮数对话记录
if len(historyList) > maxRounds {
    historyList = historyList[:maxRounds]
}
```
将对话记录加入到上下文中，rewrite prompt 中最重要的是基于历史对话记录对代词进行明确或补足，使用 LLM 对查询进行重写，对应 prompt 如下：
```yaml
rewrite_prompt_system: |
    你是一个专注于指代消解和省略补全的智能助手，你的任务是根据历史对话上下文，清晰识别用户问题中的代词并替换为明确的主语，同时补全省略的关键信息。

    ## 改写目标
    请根据历史对话，对当前用户问题进行改写，目标是：
    - 进行指代消解，将"它"、"这个"、"那个"、"他"、"她"、"它们"、"他们"、"她们"等代词替换为明确的主语
    - 补全省略的关键信息，确保问题语义完整
    - 保持问题的原始含义和表达方式不变
    - 改写后必须也是一个问题
    - 改写后的问题字数控制在30字以内
    - 仅输出改写后的问题，不要输出任何解释，更不要尝试回答该问题，后面有其他助手回去解答此问题

    ## Few-shot示例

    示例1:
    历史对话:
    用户: 微信支付有哪些功能？
    助手: 微信支付的主要功能包括转账、付款码、收款、信用卡还款等多种支付服务。

    用户问题: 它的安全性
    改写后: 微信支付的安全性

    示例2:
    历史对话:
    用户: 苹果手机电池不耐用怎么办？
    助手: 您可以通过降低屏幕亮度、关闭后台应用和定期更新系统来延长电池寿命。

    用户问题: 这样会影响使用体验吗？
    改写后: 降低屏幕亮度和关闭后台应用是否影响使用体验

    示例3:
    历史对话:
    用户: 如何制作红烧肉？
    助手: 红烧肉的制作需要先将肉块焯水，然后加入酱油、糖等调料慢炖。

    用户问题: 需要炖多久？
    改写后: 红烧肉需要炖多久

    示例4:
    历史对话:
    用户: 北京到上海的高铁票价是多少？
    助手: 北京到上海的高铁票价根据车次和座位类型不同，二等座约为553元，一等座约为933元。

    用户问题: 时间呢？
    改写后: 北京到上海的高铁时长

    示例5:
    历史对话:
    用户: 如何注册微信账号？
    助手: 注册微信账号需要下载微信APP，输入手机号，接收验证码，然后设置昵称和密码。

    用户问题: 国外手机号可以吗？
    改写后: 国外手机号是否可以注册微信账号
rewrite_prompt_user: |
    ## 历史对话背景
    {{conversation}}

    ## 需要改写的用户问题
    {{query}}

    ## 改写后的问题
```
## PluginExtractEntity 插件
通过 LLM 对查询进行实体提取，对应 prompt 如下：
```yaml
description: |
      请基于用户给的问题，按以下步骤处理关键信息提取任务：
      1. 梳理逻辑关联：首先完整分析文本内容，明确其核心逻辑关系，并简要标注该核心逻辑类型；
      2. 提取关键实体：围绕梳理出的逻辑关系，精准提取文本中的关键信息并归类为明确实体，确保不遗漏核心信息、不添加冗余内容；
      3. 排序实体优先级：按实体与文本核心主题的关联紧密程度排序，优先呈现对理解文本主旨最重要的实体；
examples:
    - text: "《红楼梦》，又名《石头记》，是清代作家曹雪芹创作的中国古典四大名著之一，被誉为中国封建社会的百科全书。"
    node:
        - name: "红楼梦"
        - name: "曹雪芹"
        - name: "中国古典四大名著"
```
解析 LLM 输出，将实体提取结果转换为结构体
```go
for _, group := range matchData {
    switch {
    case group[f.nodePrefix] != nil:
        attributes := make([]string, 0)
        attributesKey := f.nodePrefix + f.attributeSuffix
        if attr, ok := group[attributesKey].([]interface{}); ok {
            for _, v := range attr {
                attributes = append(attributes, fmt.Sprintf("%v", v))
            }
        }
        nodes = append(nodes, &types.GraphNode{
            Name:       fmt.Sprintf("%v", group[f.nodePrefix]),
            Attributes: attributes,
        })
    case group[f.relationSource] != nil && group[f.relationTarget] != nil:
        relations = append(relations, &types.GraphRelation{
            Node1: fmt.Sprintf("%v", group[f.relationSource]),
            Node2: fmt.Sprintf("%v", group[f.relationTarget]),
            Type:  fmt.Sprintf("%v", group[f.relationPrefix]),
        })
    default:
        logger.Warnf(ctx, "Unsupported graph group: %v", group)
        continue
    }
}
graph := &types.GraphData{
    Node:     nodes,
    Relation: relations,
}
```

# 多路检索 - CHUNK_SEARCH_PARALLEL
多路检索包含混合检索，知识图谱检索两部分，这两部分并行进行检索。混合检索包含 Web 搜索和知识库向量检索以及 BM25。
```go
// Goroutine 1: 混合检索
go func() {
    defer wg.Done()
    err := p.searchPlugin.OnEvent(ctx, types.CHUNK_SEARCH, &chunkChatManage, func() *PluginError {
        return nil
    })
    if err != nil && err != ErrSearchNothing {
        mu.Lock()
        chunkSearchErr = err
        mu.Unlock()
    }
    pipelineInfo(ctx, "SearchParallel", "chunk_search_done", map[string]interface{}{
        "result_count": len(chunkChatManage.SearchResult),
        "has_error":    err != nil && err != ErrSearchNothing,
    })
}()

// Goroutine 2: 知识图谱检索
go func() {
    defer wg.Done()
    if len(chatManage.Entity) == 0 {
        pipelineInfo(ctx, "SearchParallel", "entity_search_skip", map[string]interface{}{
            "reason": "no_entities",
        })
        return
    }
    err := p.searchEntityPlugin.OnEvent(ctx, types.ENTITY_SEARCH, &entityChatManage, func() *PluginError {
        return nil
    })
    if err != nil && err != ErrSearchNothing {
        mu.Lock()
        entitySearchErr = err
        mu.Unlock()
    }
    pipelineInfo(ctx, "SearchParallel", "entity_search_done", map[string]interface{}{
        "result_count": len(entityChatManage.SearchResult),
        "has_error":    err != nil && err != ErrSearchNothing,
    })
}()
```

## 混合检索 - CHUNK_SEARCH
### 前置检查
检查是否有搜索目标，如指定知识库，指定知识文档等，是否开启 Web 检索。
```go
hasKBTargets := len(chatManage.SearchTargets) > 0 || len(chatManage.KnowledgeBaseIDs) > 0 || len(chatManage.KnowledgeIDs) > 0
if !hasKBTargets && !chatManage.WebSearchEnabled {
    pipelineError(ctx, "Search", "kb_not_found", map[string]interface{}{
        "session_id": chatManage.SessionID,
    })
    return nil
}
```
若开启 Web 检索，则知识库检索和 Web 检索并行进行。
```go
// Goroutine 1: 知识库检索
go func() {
    defer wg.Done()
    kbResults := p.searchByTargets(ctx, chatManage)
    if len(kbResults) > 0 {
        mu.Lock()
        allResults = append(allResults, kbResults...)
        mu.Unlock()
    }
}()

// Goroutine 2: Web 检索
go func() {
    defer wg.Done()
    webResults := p.searchWebIfEnabled(ctx, chatManage)
    if len(webResults) > 0 {
        mu.Lock()
        allResults = append(allResults, webResults...)
        mu.Unlock()
    }
}()
```
### 知识库检索
```go
go func() {
    defer wg.Done()
    kbResults := p.searchByTargets(ctx, chatManage)
    if len(kbResults) > 0 {
        mu.Lock()
        allResults = append(allResults, kbResults...)
        mu.Unlock()
    }
}()
```
#### 小文件直接加载策略
   
遍历每个 knowledgeID 获取对应 chunks，**若累计 chunks 数量小于 50 则直接加入 allChunks，若当前 chunks 总数 + 新文件 chunks 数 > 50 则跳过此文件，并记录该文件 id 至 skippedIDs，** 最终输出 allChunks 和 skippedIDs。
```go
// 50 chunks * ~500 chars/chunk ~= 25k chars
const maxTotalChunks = 50
for _, kid := range knowledgeIDs {
    // Optimization: Check chunk count first if possible?
    chunks, err := p.chunkService.ListChunksByKnowledgeID(ctx, kid)
    if err != nil {
        logger.Warnf(ctx, "DirectLoad: Failed to list chunks for knowledge %s: %v", kid, err)
        skippedIDs = append(skippedIDs, kid)
        continue
    }

    if len(allChunks)+len(chunks) > maxTotalChunks {
        logger.Infof(ctx, "DirectLoad: Skipped knowledge %s due to size limit (%d + %d > %d)",
            kid, len(allChunks), len(chunks), maxTotalChunks)
        skippedIDs = append(skippedIDs, kid)
        continue
    }
    allChunks = append(allChunks, chunks...)
    loadedKnowledgeIDs[kid] = true
}
```
#### 向量+关键词混合检索
    
**WeKnora 支持文档切分后的文本块向量化以及用户手动输入的 FAQ 问答对向量化。**

- 对文本块内容进行向量检索 + 关键词检索。
- 对 FAQ 问答对进行向量检索。
```go
// 向量检索构建
vectorParams := types.RetrieveParams{
    Query:            params.QueryText,
    Embedding:        queryEmbedding,
    KnowledgeBaseIDs: []string{id},
    TopK:             matchCount,
    Threshold:        params.VectorThreshold,
    RetrieverType:    types.VectorRetrieverType,
    KnowledgeIDs:     params.KnowledgeIDs,
    TagIDs:           params.TagIDs,
}

// FAQ 检索构建
if kb.Type == types.KnowledgeBaseTypeFAQ {
    vectorParams.KnowledgeType = types.KnowledgeTypeFAQ
}
retrieveParams = append(retrieveParams, vectorParams)
// 关键词检索构建
retrieveParams = append(retrieveParams, types.RetrieveParams{
    Query:            params.QueryText,
    KnowledgeBaseIDs: []string{id},
    TopK:             matchCount,
    Threshold:        params.KeywordThreshold,
    RetrieverType:    types.KeywordsRetrieverType,
    KnowledgeIDs:     params.KnowledgeIDs,
    TagIDs:           params.TagIDs,
})
// 执行检索
retrieveResults, err := retrieveEngine.Retrieve(ctx, retrieveParams)
```
*Tips：这里实际的检索数量为指定检索数量的 3 倍（matchCount := params.MatchCount * 3），因为后续需要对检索结果进行去重和融合的操作*

#### 合并结果

对向量结果进行**去重 + 按分数排序**
```go
for _, info := range chunkInfoMap {
    deduplicatedChunks = append(deduplicatedChunks, info)
}
slices.SortFunc(deduplicatedChunks, func(a, b *types.IndexWithScore) int {
    if a.Score > b.Score {
        return -1
    } else if a.Score < b.Score {
        return 1
    }
    return 0
})
```

使用 RRF 算法对向量结果和关键词结果进行融合排序。RRF 算法简单来说就是：

对于每个检索器打的具体分数（因为标准不同），RRF 只关心文档在每个结果列表里的名次，然后把名次换算成分数，把多个列表的分数加起来重新排名，选出大家都认为靠前的好结果。

```go
const rrfK = 60
// 向量检索排名
vectorRanks[chunkID] = rank
// 关键词检索排名  
keywordRanks[chunkID] = rank
// 计算RRF分数
rrfScore = 1.0/(60+vectorRank) + 1.0/(60+keywordRank)
```

#### 迭代检索补充

若同时满足以下条件，则触发对检索结果的迭代检索补充，：
- 向量检索结果数量不足 params.MatchCount 个
- 检索类型为 FAQ，检索类型有 Document 和 FAQ。类型 Document 只对文本块进行检索，类型 FAQ 除了基础的文本块检索还支持对问答对进行检索。
- 向量检索结果数量达到最大检索数量
```go
needsIterativeRetrieval := len(deduplicatedChunks) < params.MatchCount &&
		kb.Type == types.KnowledgeBaseTypeFAQ && len(vectorResults) == matchCount
if needsIterativeRetrieval {
    logger.Info(ctx, "Not enough unique chunks, using iterative retrieval for FAQ")
    deduplicatedChunks = s.iterativeRetrieveWithDeduplication(
        ctx,
        retrieveEngine,
        retrieveParams,
        params.MatchCount,
        params.QueryText,
    )
}
```
迭代检索初始化检索数量依然是指定检索数量的 3 倍，`currentTopK := matchCount * 3`，默认进行 5 次迭代检索 `maxIterations := 5`，每次迭代检索数量为上一次的 2 倍 `currentTopK *= 2`。

当迭代检索满足以下条件时，会提前终止：
- 已获得足够 chunks
- 检索结果 < TopK 无更多结果

经过迭代检索补充后获取的 chunks list，会进行去重和排序操作，对其中 FAQ chunks 会进行负问题匹配过滤。**负问题在写入 FAQ 知识库时会被存储在 FAQ 的 meta.NegativeQuestions 字段中。** 例如：问题: "如何开通会员？" -> 负问题: ["如何取消会员", "如何退订会员"]。
```go
// Returns true if the query matches any negative question, false otherwise.
func (s *knowledgeBaseService) matchesNegativeQuestions(queryTextLower string, negativeQuestions []string) bool {
	if len(negativeQuestions) == 0 {
		return false
	}

	for _, negativeQ := range negativeQuestions {
		negativeQLower := strings.ToLower(strings.TrimSpace(negativeQ))
		if negativeQLower == "" {
			continue
		}
		// Check if query text is exactly the same as the negative question
		if queryTextLower == negativeQLower {
			return true
		}
	}
	return false
}
```

**Tips：对于 FAQ chunks 与 document chunks，在某些场景下，用户希望答案优先采用 FAQ chunks 中的内容，可以通过设置以下三个参数来提高 FAQ chunks 在检索结果中的权重。**
- `FAQPriorityEnabled`：是否开启 FAQ 优先级，默认值为 `false`。
- `FAQDirectAnswerThreshold`：FAQ 直接回答阈值。
- `FAQScoreBoost`：FAQ 分数提升值，大于等于 1.0。

### Web 检索
Web 检索支持 duckduckgo 搜索和 google 搜索两个方式，默认开启 duckduckgo 搜索。对经过重写的用户查询进行 Web 检索后，会对检索结果进行黑名单过滤。
#### DuckDuckGo 搜索
DuckDuckGo 搜索（简称 DDG），是一个基于隐私的搜索引擎，不存储用户搜索历史。DDG 搜索的结果是实时的，不会缓存搜索结果。常见于 搜索、RAG、Agent、隐私优先的 Web 查询场景。

DDG 支持两种搜索模式：
- HTML：获取网页版搜索结果页面，返回 HTML 格式信息。
- API：获取结构化的数据（如即时答案、摘要），返回 JSON 格式信息。

在 WeKnotra 中，DDG 优先使用 HTML 模式进行搜索，因为 HTML 模式返回的结果更丰富，包含了搜索结果的标题、摘要、链接等信息。降级使用 API 模式。
```go
// Try HTML scraping first (more reliable for general results)
htmlResults, err := p.searchHTML(ctx, query, maxResults)
if err == nil && len(htmlResults) > 0 {
    return htmlResults, nil
}
// Fallback to Instant Answer API
apiResults, apiErr := p.searchAPI(ctx, query, maxResults)
if apiErr == nil && len(apiResults) > 0 {
    return apiResults, nil
}
```

#### Google 搜索
Google 搜索需要配置 Google API Key 和 Google Custom Search Engine ID。

#### 黑名单过滤
对检索到的链接进行黑名单过滤，支持通配符模式（如 *://*.example.com/*）和正则（如 /example\.(net|org)/）两种方式。
```go
func (s *WebSearchService) matchesBlacklistRule(url, rule string) bool {
	// Check if it's a regex pattern (starts and ends with /)
	if strings.HasPrefix(rule, "/") && strings.HasSuffix(rule, "/") {
		pattern := rule[1 : len(rule)-1]
		matched, err := regexp.MatchString(pattern, url)
		if err != nil {
			logger.Warnf(context.Background(), "Invalid regex pattern in blacklist: %s, error: %v", rule, err)
			return false
		}
		return matched
	}

	// Pattern matching (e.g., *://*.example.com/*)
	pattern := strings.ReplaceAll(rule, "*", ".*")
	pattern = "^" + pattern + "$"
	matched, err := regexp.MatchString(pattern, url)
	if err != nil {
		logger.Warnf(context.Background(), "Invalid pattern in blacklist: %s, error: %v", rule, err)
		return false
	}
	return matched
}
```
#### 压缩 Web 检索结果
**先创建临时知识库，若该会话（session id）已存在临时知识库，则直接复用该知识库。**
```go
if createdKB == nil {
    kb := &types.KnowledgeBase{
        Name:             fmt.Sprintf("tmp-websearch-%d", time.Now().UnixNano()),
        Description:      "Ephemeral search compression KB",
        IsTemporary:      true,
        EmbeddingModelID: cfg.EmbeddingModelID,
    }
    createdKB, err = kbSvc.CreateKnowledgeBase(ctx, kb)
    if err != nil {
        return nil, tempKBID, seenURLs, knowledgeIDs, fmt.Errorf(
            "failed to create temporary knowledge base: %w",
            err,
        )
    }
    tempKBID = createdKB.ID
}
```
将 Web 检索结果中的**源链接、标题、摘要、正文内容信息**合并为一个 Passage，写入到临时知识库中。
```go
for _, r := range webSearchResults {
    sourceURL := r.URL
    title := strings.TrimSpace(r.Title)
    snippet := strings.TrimSpace(r.Snippet)
    body := strings.TrimSpace(r.Content)
    // skip if already ingested for this KB
    if sourceURL != "" && seenURLs[sourceURL] {
        continue
    }
    contentLines := make([]string, 0, 4)
    contentLines = append(contentLines, fmt.Sprintf("[sourceUrl]: %s", sourceURL))
    if title != "" {
        contentLines = append(contentLines, title)
    }
    if snippet != "" {
        contentLines = append(contentLines, snippet)
    }
    if body != "" {
        contentLines = append(contentLines, body)
    }
    knowledge, err := knowSvc.CreateKnowledgeFromPassageSync(ctx, createdKB.ID, contentLines)
    if err != nil {
        logger.Warnf(ctx, "failed to ingest passage into temp KB: %v", err)
        continue
    }
    if sourceURL != "" {
        seenURLs[sourceURL] = true
    }
    knowledgeIDs = append(knowledgeIDs, knowledge.ID)
}
```
*Tips：注意，`CreateKnowledgeFromPassageSync` 这里不仅仅是将 contentLines 写入到知识库存储中，还会对 contentLines 进行向量化，生成 Embedding 向量。分别对 contentLines 中的元素进行安全性校验，和归一化处理，再直接按照每个元素作为一个 chunk 的方式进行向量化。*

使用重写查询，在临时知识库中对 chunk 进行检索召回。
```go
params := types.SearchParams{
    QueryText:        q,
    VectorThreshold:  0.5,
    KeywordThreshold: 0.5,
    MatchCount:       matchCount,
}
results, err := kbSvc.HybridSearch(ctx, tempKBID, params)
```
使用轮询分配算法，从检索结果中公平选择引用，确保每个来源网站都有内容能够被引用。
```go
selected := s.selectReferencesRoundRobin(webSearchResults, allRefs, matchCount*len(webSearchResults))
```
将最终的 chunk list 按照各个来源分组后合并回原始的 WebSearchResult 结构。
```go
compressedResults := s.consolidateReferencesByURL(webSearchResults, selected)
```
最后将本次压缩后的 WebSearchResult 进行缓存与会话（session id）进行关联，后续检索时用于校验去重等操作。当会话清除时，临时知识库以及缓存中的数据也会被清除。

### 问题扩写
#### 触发条件
经过知识库检索和 Web 检索后，若检索到的结果数量不足预期数量的一半，且开启了问题扩写功能，则进行问题扩写。
```go
if chatManage.EnableQueryExpansion && len(chatManage.SearchResult) < max(1, chatManage.EmbeddingTopK/2) {
	expansions := p.expandQueries(ctx, chatManage)
    ...
}
```
#### 扩写策略
1. 对原始问题进行分词，提取有效关键词。
```go
keywords := extractKeywords(query)
// 预定义中英文停用词
var stopwords = map[string]struct{}{
	"的": {}, "是": {}, "在": {}, "了": {}, "和": {}, "与": {}, "或": {},
	"a": {}, "an": {}, "the": {}, "is": {}, "are": {}, "was": {}, "were": {},
	"be": {}, "been": {}, "being": {}, "have": {}, "has": {}, "had": {},
	"do": {}, "does": {}, "did": {}, "will": {}, "would": {}, "could": {},
	"should": {}, "may": {}, "might": {}, "must": {}, "can": {},
	"to": {}, "of": {}, "in": {}, "for": {}, "on": {}, "with": {}, "at": {},
	"by": {}, "from": {}, "as": {}, "into": {}, "through": {}, "about": {},
	"what": {}, "how": {}, "why": {}, "when": {}, "where": {}, "which": {},
	"who": {}, "whom": {}, "whose": {},
}
// 提取有效关键词
func extractKeywords(text string) []string {
    words := tokenize(text)  // 分词
    keywords := make([]string, 0, len(words))
    for _, w := range words {
        lower := strings.ToLower(w)
        // 过滤停用词和单字符
        if _, isStop := stopwords[lower]; !isStop && len(w) > 1 {
            keywords = append(keywords, w)
        }
    }
    return keywords
}
```
2. 按照规则提取有效短语
```go
phrases := extractPhrases(query)
// 提取短语（双引号、单引号、中文引号等）
func extractPhrases(text string) []string {
    var phrases []string
    // 匹配各种引号内的内容
    re := regexp.MustCompile(`["'"'「」『』]([^"'"'「」『』]+)["'"'「」『』]`)
    matches := re.FindAllStringSubmatch(text, -1)
    for _, m := range matches {
        if len(m) > 1 && len(m[1]) > 2 {  // 长度 > 2
            phrases = append(phrases, m[1])
        }
    }
    return phrases
}
```
3. 分隔符拆分问题
```go
segments := splitByDelimiters(query)
// 分隔符拆分问题（空格、逗号、句号等）
func splitByDelimiters(text string) []string {
    // 按标点符号和空白分割
    re := regexp.MustCompile(`[,，;；、。！？!?\s]+`)
    parts := re.Split(text, -1)
    var result []string
    for _, p := range parts {
        p = strings.TrimSpace(p)
        if p != "" {
            result = append(result, p)
        }
    }
    return result
}
```
4. 查询变体，移除疑问词
```go
cleaned := removeQuestionWords(query)
// 疑问词
var questionWords = regexp.MustCompile(
    `^(什么是|什么|如何|怎么|怎样|为什么|为何|哪个|哪些|谁|何时|何地|请问|请告诉我|帮我|我想知道|我想了解)`,
)
// 移除预定义疑问词（如：“什么是”、“如何”等）
func removeQuestionWords(text string) string {
    return strings.TrimSpace(questionWords.ReplaceAllString(text, ""))
}
```
通过以上策略扩写问题，每个策略都会生成 1 个或多个变体。将这些变体合并去重后，只取前 `expansions := make([]string, 0, 5)` 个变体作为最终的扩写结果。

最后，将扩写后的问题进行检索召回。
```go
res, err := p.knowledgeBaseService.HybridSearch(ctx, t.KnowledgeBaseID, paramsExp)
```

### 检索结果处理
对检索结果列表 SearchResult 进行上下文添加，去重等操作，**得到最终的混合检索结果。**
#### 获取历史引用
返回最近一轮有知识引用的结果，引用标记为 MatchTypeHistory，加入检索结果列表 SearchResult。
```go
// Add relevant results from chat history
historyResult := p.getSearchResultFromHistory(chatManage)
if historyResult != nil {
    chatManage.SearchResult = append(chatManage.SearchResult, historyResult...)
}
```
#### 检索结果去重
对检索结果列表 SearchResult 进行去重，移除重复的结果。这里去重分别对 chunk id 和 chunk content 进行双层去重。
```go
// Remove duplicate results
chatManage.SearchResult = removeDuplicateResults(chatManage.SearchResult)
```

## 知识图谱检索 - ENTITY_SEARCH
知识图谱检索相较于混合检索简单很多，和传统方案一样，根据问题中提取的实体进行实体搜索，返回相关 chunk。问题中的实体提取操作在问题重写事件中已经触发。知识图谱默认支持 Neo4j。
```go
graph, err := p.graphRepo.SearchNode(ctx, types.NameSpace{KnowledgeBase: knowledgeBaseID,Knowledge: knowledgeID}, entity)
```

对召回的 chunk 进行相关信息提取，数据结构转换，去重等操作后，加入检索结果列表 SearchResult。

# 检索重排 - CHUNK_RERANK
## Rerank 模型选择
WeKnora 支持 OpenAI（OpenAIReranker）、阿里云 DashScope（AliyunReranker）、智谱（ZhipuReranker），以及 Jina（JinaReranker）四种 rerank 方案，可以根据实际场景指定 RerankModelID 值来进行选择。
```go
switch providerName {
case provider.ProviderAliyun:
    return NewAliyunReranker(config)
case provider.ProviderZhipu:
    return NewZhipuReranker(config)
case provider.ProviderJina:
    return NewJinaReranker(config)
default:
    return NewOpenAIReranker(config)
}
```
**rerank 无默认值，若未设置 RerankModelID 则会跳过 rerank 过程。**
```go
if chatManage.RerankModelID == "" {
    return next()
}
```
## Passages 构建
先对检索结果列表 SearchResult 进行结果分类，在**混合检索中通过小文件加载策略**标记为 `types.MatchTypeDirectLoad` 的 directLoadResults 跳过 rerank，直接进入最终结果计算。
```go
if result.MatchType == types.MatchTypeDirectLoad {
    directLoadResults = append(directLoadResults, result)
    pipelineInfo(ctx, "Rerank", "direct_load_skip", map[string]interface{}{
        "chunk_id": result.ID,
    })
    continue
}
```
提取 result 中 chunk 相关的图片描述（Caption），图片 OCR 文本，相关问题（GeneratedQuestions）等信息加入 passages 构建，进行信息增强。
```go
// 合并Content和ImageInfo的文本内容
passage := getEnrichedPassage(ctx, result)

if result.ImageInfo != "" {
    var imageInfos []types.ImageInfo
    err := json.Unmarshal([]byte(result.ImageInfo), &imageInfos)
    if err != nil {
        pipelineWarn(ctx, "Rerank", "image_info_parse", map[string]interface{}{
            "error": err.Error(),
        })
    } else {
        // 提取所有图片的描述和OCR文本
        for _, img := range imageInfos {
            if img.Caption != "" {
                enrichments = append(enrichments, fmt.Sprintf("图片描述: %s", img.Caption))
            }
            if img.OCRText != "" {
                enrichments = append(enrichments, fmt.Sprintf("图片文本: %s", img.OCRText))
            }
        }
    }
}

// 解析ChunkMetadata中的GeneratedQuestions
if len(result.ChunkMetadata) > 0 {
    var docMeta types.DocumentChunkMetadata
    err := json.Unmarshal(result.ChunkMetadata, &docMeta)
    if err != nil {
        pipelineWarn(ctx, "Rerank", "chunk_metadata_parse", map[string]interface{}{
            "error": err.Error(),
        })
    } else if questionStrings := docMeta.GetQuestionStrings(); len(questionStrings) > 0 {
        enrichments = append(enrichments, fmt.Sprintf("相关问题: %s", strings.Join(questionStrings, "; ")))
    }
}

if len(enrichments) == 0 {
    return combinedText
}

// 组合内容和增强信息
if combinedText != "" {
    combinedText += "\n\n"
}
combinedText += strings.Join(enrichments, "\n")
```
## Rerank
将 passages 列表和问题传入 rerank 模型，获取 rerank 后的相关性分数。并使用默认 RerankThreshold 值（0.5）对 rerank 结果进行过滤。
```go
rerankResp, err := rerankModel.Rerank(ctx, query, passages)
...
// Filter results based on threshold with special handling for history matches
rankFilter := []rerank.RankResult{}
for _, result := range rerankResp {
    if result.Index >= len(candidates) {
        continue
    }
    th := chatManage.RerankThreshold
    matchType := candidates[result.Index].MatchType
    if matchType == types.MatchTypeHistory {
        th = math.Max(th-0.1, 0.5) // Lower threshold for history matches
    }
    if result.RelevanceScore > th {
        rankFilter = append(rankFilter, result)
    }
}
return rankFilter
```
*Tips：若引用片段标记为历史引用 MatchTypeHistory，将阈值匹配降低 0.1，最低限制不低于 0.5。因为“历史引用”这类结果通常是上轮已用过 / 用户刚提过，可能对当前问题仍然有价值，但 rerank 分数未必很高，所以稍微放宽过滤，提高留存概率。*

## 阈值降级策略
若 rerank 结果中无符合阈值的引用片段，且当前匹配阈值 > 0.3，则将阈值降级重新 rerank，新阈值 = 原阈值 × 0.7，最低不低于 0.3。
```go
if len(rerankResp) == 0 && originalThreshold > 0.3 {
    degradedThreshold := originalThreshold * 0.7
    if degradedThreshold < 0.3 {
        degradedThreshold = 0.3
    }
    chatManage.RerankThreshold = degradedThreshold
    rerankResp = p.rerank(ctx, chatManage, rerankModel, chatManage.RewriteQuery, passages, candidatesToRerank)
    // Restore original threshold
    chatManage.RerankThreshold = originalThreshold
}
```
## 计算最终评分
所有引用片段的计算最终评分。
- **包含 Passages 构建中跳过 rerank 的 directLoadResults 结果，假设高相关，即 modelScore 直接设为 1.0**
- **FAQ 类引用片段，如果有设置 FAQScoreBoost 来提高 FAQ 类引用片段的分数，会额外乘以 FAQScoreBoost 因子**

最终评分的计算公式为：
```go
composite = (0.6 × modelScore + 0.3 × baseScore + 0.1 × sourceWeight) × positionPrior
```
权重设计：
- 60%：rerank 模型分数（相关性）
- 30%：原始检索分数（baseScore）
- 10%：来源权重（web_search=0.95，其他=1.0）
```go
composite := 0.6*modelScore + 0.3*baseScore + 0.1*sourceWeight
```
positionPrior：位置先验因子，根据引用片段在原文中的位置（StartAt），位置越靠前越有优势，会有最多 +5% 的轻微加成；反之最多 -5%。**因为当分数接近时，倾向选择更“前置/摘要/定义”类片段（很多文档重要信息在开头）。**
```go
positionPrior := 1.0
if sr.StartAt >= 0 {
    positionPrior += searchutil.ClampFloat(1.0-float64(sr.StartAt)/float64(sr.EndAt+1), -0.05, 0.05)
}
composite *= positionPrior
```
## MMR 多样性选择
在保证高相关性的前提下，强制筛选出信息更全面、内容更不重复的文档集合，防止 LLM 收到一堆内容雷同的“废话”，最终生成更全面、更有洞见的答案。
```go
lambda := 0.7
mmr := lambda*relevance - (1.0-lambda)*redundancy
```
- Lambda 是调节相关性和冗余度的权重参数，取值范围为 [0,1]。值越大表示更重视相关性，值越小表示更重视多样性。这里设置为 0.7，表示更重视相关性。 
- Relevance 相关性即为上一步计算得出的 composite 分数。
- Redundancy 冗余度则是指文档集合中不同文档之间的相似度，使用 Jaccard 相似度计算。计算候选文档与每一个已选文档的相似度，然后取最大值。
```go
sim := searchutil.Jaccard(allTokenSets[i], selTokens)
if sim > redundancy {
    redundancy = sim
}
```
# 合并结果 - CHUNK_MERGE
对最终 Results 列表进行合并，优先使用 RerankResult，为空则降级到 SearchResult。
## Chunks 列表构建
按 KnowledgeID 分组，再按 ChunkType 细分，最终输出结构：map[KnowledgeID]map[ChunkType][]SearchResult
```go
knowledgeGroup := make(map[string]map[string][]*types.SearchResult)
for _, chunk := range searchResult {
    if _, ok := knowledgeGroup[chunk.KnowledgeID]; !ok {
        knowledgeGroup[chunk.KnowledgeID] = make(map[string][]*types.SearchResult)
    }
    knowledgeGroup[chunk.KnowledgeID][chunk.ChunkType] = append(knowledgeGroup[chunk.KnowledgeID][chunk.ChunkType], chunk)
}
```
按照 StartAt 位置进行升序排序，若 StartAt 相同，则按 EndAt 位置升序排序。
```go
sort.Slice(chunks, func(i, j int) bool {
    if chunks[i].StartAt == chunks[j].StartAt {
        return chunks[i].EndAt < chunks[j].EndAt
    }
    return chunks[i].StartAt < chunks[j].StartAt
})
```
## 合并重叠/相邻内容
根据位置信息合并重叠/相邻 Chunks。合并后取最高的 Score。
```go
knowledgeMergedChunks := []*types.SearchResult{chunks[0]}
for i := 1; i < len(chunks); i++ {
	lastChunk := knowledgeMergedChunks[len(knowledgeMergedChunks)-1]
	// If the current chunk starts after the last chunk ends, add it to the merged chunks
	if chunks[i].StartAt > lastChunk.EndAt {
		knowledgeMergedChunks = append(knowledgeMergedChunks, chunks[i])
		continue
	}
	// Merge overlapping chunks
	if chunks[i].EndAt > lastChunk.EndAt {
		lastChunk.Content = lastChunk.Content +
			string([]rune(chunks[i].Content)[lastChunk.EndAt-chunks[i].StartAt:])
		lastChunk.EndAt = chunks[i].EndAt
		lastChunk.SubChunkID = append(lastChunk.SubChunkID, chunks[i].ID)
        ...
	}
	if chunks[i].Score > lastChunk.Score {
		lastChunk.Score = chunks[i].Score
	}
}
```
以 URL 作为唯一标识，对 ImageInfo 进行合并。
```go
// 合并 ImageInfo
if err := mergeImageInfo(ctx, lastChunk, chunks[i]); err != nil {
    pipelineWarn(ctx, "Merge", "image_merge", map[string]interface{}{
        "knowledge_id": knowledgeID,
        "error":        err.Error(),
    })
}
```
## FAQ 内容填充
对召回的 FAQ 类引用片段，填充完整的 FAQ 内容。最终填充格式为：
```
Q: 标准问题
Answer:
- 答案1
- 答案2
```
## 短内容扩展
对普通的引用片段，若内容长度 < 350 字符的短内容，根据当前位置信息进行前后迭代扩展，迭代扩展先向前，再向后，直到达到 maxLen = 850 或无法继续。并更新扩展后的 metadata 信息。
```go
merged = mergeOrderedContent(prevContent, baseChunk.Content, nextContent, maxLen)
```
# 结果过滤 - FILTER_TOP_K
对 Results 列表进行过滤，保留前 topK 文本块。
```go
filterTopK := func(searchResult []*types.SearchResult, topK int) []*types.SearchResult {
    if topK > 0 && len(searchResult) > topK {
        pipelineInfo(ctx, "FilterTopK", "filter", map[string]interface{}{
            "before": len(searchResult),
            "after":  topK,
        })
        searchResult = searchResult[:topK]
    }
    return searchResult
}
```
对结果集进行降级策略，优先合并后结果 MergeResult，重排序结果次之 RerankResult，最后是原始的检索结果 SearchResult。
```go
if len(chatManage.MergeResult) > 0 {
    chatManage.MergeResult = filterTopK(chatManage.MergeResult, chatManage.RerankTopK)
} else if len(chatManage.RerankResult) > 0 {
    chatManage.RerankResult = filterTopK(chatManage.RerankResult, chatManage.RerankTopK)
} else if len(chatManage.SearchResult) > 0 {
    chatManage.SearchResult = filterTopK(chatManage.SearchResult, chatManage.RerankTopK)
} else {
    pipelineWarn(ctx, "FilterTopK", "skip", map[string]interface{}{
        "reason": "no_results",
    })
}
```
# 数据分析 - DATA_ANALYSIS
如果最终的 Results 列表中关联的文件包含**数据文件（如 .csv、.xlsx、.xls 等）**，则需要对数据进行处理，并通过 LLM 根据用户问题生成查询语句，对数据进行查询输出最终结果添加到 Results 列表中。
## 文档识别
识别 Results 列表中关联的文件是否包含数据文件（如 .csv、.xlsx、.xls 等）。
```go
for _, result := range chatManage.MergeResult {
	if isDataFile(result.KnowledgeFilename) {
		dataFiles = append(dataFiles, result)
	}
}
```
若存在数据文件，则移除检索结果 Results 列表中的数据文件相关表结构引用（如：ChunkTypeTableColumn 和 ChunkTypeTableSummary），避免重复引用。
```go
func filterOutTableChunks(results []*types.SearchResult) []*types.SearchResult {
	filtered := make([]*types.SearchResult, 0, len(results))
	filterList := []string{string(types.ChunkTypeTableColumn), string(types.ChunkTypeTableSummary)}
	for _, result := range results {
		if slices.Contains(filterList, result.ChunkType) {
			continue
		}
		filtered = append(filtered, result)
	}
	return filtered
}
``` 
## 加载数据
将数据加载到 DuckDB，根据 knowledgeID 生成表名 k_{knowledgeID}。
- .csv 文件使用 `read_csv_auto` 进行创建。
```go
func (t *DataAnalysisTool) LoadFromCSV(ctx context.Context, filename string, tableName string) (*TableSchema, error) {
	logger.Infof(ctx, "[Tool][DataAnalysis] Loading CSV file '%s' into table '%s' for session %s", filename, tableName, t.sessionID)

	// Record the created table for cleanup. If already exists, skip creation
	if t.recordCreatedTable(tableName) {
		// Create table from CSV using DuckDB's read_csv_auto function
		// Table will be created in the session schema
		createTableSQL := fmt.Sprintf("CREATE TABLE \"%s\" AS SELECT * FROM read_csv_auto('%s')", tableName, filename)

		_, err := t.db.ExecContext(ctx, createTableSQL)
		if err != nil {
			logger.Errorf(ctx, "[Tool][DataAnalysis] Failed to create table from CSV: %v", err)
			return nil, fmt.Errorf("failed to create table from CSV: %w", err)
		}

		logger.Infof(ctx, "[Tool][DataAnalysis] Successfully created table '%s' from CSV file in session %s", tableName, t.sessionID)
	}

	// Get and return the table schema
	return t.LoadFromTable(ctx, tableName)
}
```
- .xlsx、.xls 文件使用 `st_read` 进行创建。
```go
func (t *DataAnalysisTool) LoadFromExcel(ctx context.Context, filename string, tableName string) (*TableSchema, error) {
	logger.Infof(ctx, "[Tool][DataAnalysis] Loading Excel file '%s' into table '%s' for session %s", filename, tableName, t.sessionID)

	// Record the created table for cleanup. If already exists, skip creation
	if t.recordCreatedTable(tableName) {
		// Try to read Excel file using st_read (from spatial extension)
		// If spatial extension doesn't support Excel, we'll need to convert to CSV first
		createTableSQL := fmt.Sprintf("CREATE TABLE \"%s\" AS SELECT * FROM st_read('%s')", tableName, filename)

		_, err := t.db.ExecContext(ctx, createTableSQL)
		if err != nil {
			logger.Errorf(ctx, "[Tool][DataAnalysis] Failed to create table from Excel: %v", err)
			return nil, fmt.Errorf("failed to create table from Excel file. Consider converting to CSV first: %w", err)
		}

		logger.Infof(ctx, "[Tool][DataAnalysis] Successfully created table '%s' from Excel file in session %s", tableName, t.sessionID)
	}

	// Get and return the table schema
	return t.LoadFromTable(ctx, tableName)
}
```
## 生成 SQL
将用户问题，Knowledge ID，对应表结构（列名、类型、行数）加入 prompt，让 LLM 判断是否需要生成 SQL 查询语句。
```go
analysisPrompt := fmt.Sprintf(`
User Question: %s
Knowledge ID: %s
Table Schema: %s

Determine if the user's question requires data analysis (e.g., statistics, aggregation, filtering) on this table.
If YES, generate a DuckDB SQL query to answer the user's question and fill in the knowledge_id and sql fields.
If NO, leave the sql field empty.

Return your response in the specified JSON format.`, chatManage.Query, knowledge.ID, schema.Description())

response, err := chatModel.Chat(ctx, []chat.Message{
    {Role: "user", Content: analysisPrompt},
}, &chat.ChatOptions{
    Temperature: 0.1,
    Format:      formatSchema,
})
```
如果生成了 SQL 语句，需要对 SQL 语句进行校验，**仅允许只读查询：SELECT、SHOW、DESCRIBE、EXPLAIN、PRAGMA 安全操作**，执行 SQL 语句。
```go
toolResult, err := tool.Execute(ctx, json.RawMessage(response.Content))
if err != nil {
    logger.Errorf(ctx, "Failed to execute SQL: %v", err)
    return next()
}
```
最后将数据分析结果加入 Results 列表中。
# 构建最终消息体 - INTO_CHAT_MESSAGE
## 查询安全性校验
对用户的输入查询进行安全性校验。**个人认为安全性校验步骤应该在 Pipeline 最开始的步骤中（例如：rewrite）中进行是否更为合适，从源头对危险查询进行拦截，也减少资源浪费。**
```go
safeQuery, isValid := utils.ValidateInput(chatManage.Query)
```
主要是针对**控制字符，UTF-8 有效性，XSS 攻击**三方面安全性进行校验。
```go
func ValidateInput(input string) (string, bool) {
	if input == "" {
		return "", true
	}

	// 检查是否包含控制字符
	for _, r := range input {
		if r < 32 && r != 9 && r != 10 && r != 13 {
			return "", false
		}
	}

	// 检查 UTF-8 有效性
	if !utf8.ValidString(input) {
		return "", false
	}

	// 检查是否包含潜在的 XSS 攻击
	for _, pattern := range xssPatterns {
		if pattern.MatchString(input) {
			return "", false
		}
	}

	return strings.TrimSpace(input), true
}
```
## 引用信息上下文构建
### FAQ 分离（可选）
若开启了 FAQ 优先配置（FAQPriorityEnabled = true），则会将 results 分为 FAQ 和文档两类，并对 FAQ 进行高置信度检测 `Score >= FAQDirectAnswerThreshold`，且确保只返回一个高置信度的 FAQ。
```go
if chatManage.FAQPriorityEnabled {
    for _, result := range chatManage.MergeResult {
        if result.ChunkType == string(types.ChunkTypeFAQ) {
            faqResults = append(faqResults, result)
            // Check if this FAQ has high confidence (above direct answer threshold)
            if result.Score >= chatManage.FAQDirectAnswerThreshold && !hasHighConfidenceFAQ {
                hasHighConfidenceFAQ = true
            }
        } else {
            docResults = append(docResults, result)
        }
    }
}
```
### 构建引用上下文
上下文构建分为两种情况：
1. 开启 FAQ 优先配置，有高置信度的 FAQ 时，将高置信度的 FAQ 加入上下文。
```go
if chatManage.FAQPriorityEnabled && len(faqResults) > 0 {
	// Build structured context with FAQ prioritization
	contextsBuilder.WriteString("### 资料来源 1：标准问答库 (FAQ)\n")
	contextsBuilder.WriteString("【高置信度 - 请优先参考】\n")
	for i, result := range faqResults {
		passage := getEnrichedPassageForChat(ctx, result)
		if hasHighConfidenceFAQ && i == 0 {
			contextsBuilder.WriteString(fmt.Sprintf("[FAQ-%d] ⭐ 精准匹配: %s\n", i+1, passage))
		} else {
			contextsBuilder.WriteString(fmt.Sprintf("[FAQ-%d] %s\n", i+1, passage))
		}
	}

	if len(docResults) > 0 {
		contextsBuilder.WriteString("\n### 资料来源 2：参考文档\n")
		contextsBuilder.WriteString("【补充资料 - 仅在FAQ无法解答时参考】\n")
		for i, result := range docResults {
			passage := getEnrichedPassageForChat(ctx, result)
			contextsBuilder.WriteString(fmt.Sprintf("[DOC-%d] %s\n", i+1, passage))
		}
	}
}

// 输出示例
### 资料来源 1：标准问答库 (FAQ)
【高置信度 - 请优先参考】
[FAQ-1] ⭐ 精准匹配: [FAQ内容]
[FAQ-2] [FAQ内容]

### 资料来源 2：参考文档
【补充资料 - 仅在FAQ无法解答时参考】
[DOC-1] [文档内容]
[DOC-2] [文档内容]
```
2. 未开启 FAQ 优先配置或无高置信度的 FAQ 时，直接将所有文档加入上下文。
```go
passages := make([]string, len(chatManage.MergeResult))
for i, result := range chatManage.MergeResult {
    passages[i] = getEnrichedPassageForChat(ctx, result)
}
for i, passage := range passages {
    if i > 0 {
        contextsBuilder.WriteString("\n\n")
    }
    contextsBuilder.WriteString(fmt.Sprintf("[%d] %s", i+1, passage))
}
// 输出示例
[1] [内容1]

[2] [内容2]

[3] [内容3]
```
## 图片信息处理
若引用中包含图片信息 ImageInfo，会对图片信息进行解析后，根据 URL 的位置进行相应的内容（图片 OCR 和 图片描述 Caption）的插入。
```go
// 查找内容中的所有Markdown图片链接
matches := markdownImageRegex.FindAllStringSubmatch(content, -1)

// 替换每个图片链接，添加描述和OCR文本
for _, match := range matches {
    if len(match) < 3 {
        continue
    }

    // 提取图片URL，忽略alt文本
    imgURL := match[2]

    // 标记该URL已处理
    processedURLs[imgURL] = true

    // 查找匹配的图片信息
    imgInfo, found := imageInfoMap[imgURL]

    // 如果找到匹配的图片信息，添加描述和OCR文本
    if found && imgInfo != nil {
        replacement := match[0] + "\n"
        if imgInfo.Caption != "" {
            replacement += fmt.Sprintf("图片描述: %s\n", imgInfo.Caption)
        }
        if imgInfo.OCRText != "" {
            replacement += fmt.Sprintf("图片文本: %s\n", imgInfo.OCRText)
        }
        content = strings.Replace(content, match[0], replacement, 1)
    }
}

// 输出示例
原始内容: ![图表](https://example.com/chart.png)

增强后:
![图表](https://example.com/chart.png)
图片描述: 这是一张销售趋势图表
图片文本: 2024年Q1销售额: 100万
```

## 构建最终消息体
替换消息体中的对应变量后，输出最终的消息体 userContent。
- {{query}}：用户查询（已安全验证）
- {{contexts}}：构建的上下文内容
- {{current_time}}：当前时间（格式：2006-01-02 15:04:05）
- {{current_week}}：当前星期（中文：星期一、星期二等）
```go
// Replace placeholders in context template
userContent := chatManage.SummaryConfig.ContextTemplate
userContent = strings.ReplaceAll(userContent, "{{query}}", safeQuery)
userContent = strings.ReplaceAll(userContent, "{{contexts}}", contextsBuilder.String())
userContent = strings.ReplaceAll(userContent, "{{current_time}}", time.Now().Format("2006-01-02 15:04:05"))
userContent = strings.ReplaceAll(userContent, "{{current_week}}", weekdayName[time.Now().Weekday()])

// Set formatted content back to chat management
chatManage.UserContent = userContent
```

# 流式聊天 - CHAT_COMPLETION_STREAM
## 构建消息列表
消息列表中包含三个信息：
- System Message：系统提示词
```yaml
prompt: |
    你是一个专业的智能信息检索助手，名为WeKnora。你犹如专业的高级秘书，依据检索到的信息回答用户问题，不能利用任何先验知识。
    当用户提出问题时，助手会基于特定的信息进行解答。助手首先在心中思考推理过程，然后向用户提供答案。
    ## 回答问题规则
    - 仅根据检索到的信息中的事实进行回复，不得运用任何先验知识，保持回应的客观性和准确性。
    - 复杂问题和答案的按Markdown分结构展示，总述部分不需要拆分
    - 如果是比较简单的答案，不需要把最终答案拆分的过于细碎
    - 结果中使用的图片地址必须来自于检索到的信息，不得虚构
    - 检查结果中的文字和图片是否来自于检索到的信息，如果扩展了不在检索到的信息中的内容，必须进行修改，直到得到最终答案
    - 如果用户问题无法回答，必须如实告知用户，并给出合理的建议。

    ## 输出限制
    - 以Markdown图文格式输出你的最终结果
    - 输出内容要保证简短且全面，条理清晰，信息明确，不重复。
```
- History Messages：历史对话，User + Assistant 交替。**在 rag_stream 的 pipline 中 rewrite 阶段已经做了历史记录提取，不仅用于问题改写，还用于这个极端的消息列表构建。**
```go
chatManage.History = historyList
```
- Current User Message：当前用户查询，即为上一个步骤中构建的 userContent。
## 调用 LLM 进行流式回复
调用 LLM 进行流式回复，将回复内容逐步返回给用户。

# 流式过滤 - STREAM_FILTER
这里会对输出的内容进行前缀匹配过滤，若用户没有设置自定义的前缀匹配规则，则使用默认的前缀匹配规则。
- `<think>...</think>`：思考过程标签（会被移除）
- `NO_MATCH`：无匹配标记
若最终没有获取到有效内容，会触发降级策略，降级策略分为两种模式 fix 和 model，默认为 model。
- fix：输出固定回答
```yaml
fallback_response: "抱歉，我无法回答这个问题。"
```
- model：调用 LLM 进行回复，使用预定义的 fallback_prompt。
```yaml
fallback_prompt: |
    你是一个专业、友好的AI助手。请根据你的知识直接回答用户的问题。

    ## 回复要求
    - 直接回答用户的问题
    - 简洁清晰，言之有物
    - 如果涉及实时数据或个人隐私信息，诚实说明无法获取
    - 使用礼貌、专业的语气

    ## 用户的问题是:
    {{query}}
```

# 尾言

至此，围绕 WeKnora 的这一组源码解析也告一段落了。

在这个系列中，我们先从**文档解析与切分**入手，分析了 WeKnora 如何在复杂文档场景下构建结构稳定、语义完整的 Chunk；随后又聚焦 **RAG 的检索与重排阶段**，拆解了多路召回、降级策略、去重与筛选等一系列更贴近真实生产环境的工程实现。

如果一定要总结 WeKnora 的特点，它并不追求某一个环节“做到极致”，而是始终围绕一个目标展开：  
**在不稳定输入、不完美召回和有限上下文的前提下，尽可能稳定地为 LLM 提供可用信息。**

很多实现细节——例如对 FAQ 的优先处理、rerank 失败时的阈值回退、Chunk 合并与位置先验——本身并不复杂，但它们几乎都来自真实系统中的失败经验，而不是论文中的理想假设。这也正是 WeKnora 这套方案最有参考价值的地方。

RAG 从来不是“接个向量库就结束”的问题，而是一套需要不断权衡、取舍和修正的工程系统。希望这个系列对你理解生产级 RAG 的真实形态，能提供一些具体而可落地的参考。
