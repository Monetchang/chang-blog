---
title: "【解密源码】WeKnora 文档切分与 Chunk 构建解析：腾讯生产级 RAG 的底层设计 "
date: 2026-01-16T17:06:10+08:00
draft: false
tags: ["源码","技术","RAG"]
categories: ["RAG"]
---

# 引言

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
- LOAD_HISTORY 加载历史记录
- REWRITE_QUERY 查询重写
- CHUNK_SEARCH_PARALLEL 多路检索（文档检索 + 实体检索）
- CHUNK_RERANK 检索重排
- CHUNK_MERGE 合并结果
- FILTER_TOP_K 过滤 top K 结果
- DATA_ANALYSIS 数据分析
- INTO_CHAT_MESSAGE 将回答转换为聊天消息

Chat 的相关步骤这里不做拆解，重点拆解 rag_stream Pipline 中的所有步骤。

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


## 知识图谱检索 - ENTITY_SEARCH
# 检索重排 - CHUNK_RERANK
# 合并结果 - CHUNK_MERGE
# 结果过滤 - FILTER_TOP_K
# 数据分析 - DATA_ANALYSIS
# 答案输出 - INTO_CHAT_MESSAGE
