---
title: "Harness Engineering 不缺概念，缺的是生产环境里的工程答案"
date: 2026-03-17T10:39:10+08:00
draft: false
tags: ["技术","架构","Agent"]
categories: ["架构"]
---

最近段时间，关于 Harness Engineering 的文章已经不少了。

很多文章都在讲它是什么、和 Prompt Engineering 有什么区别、为什么它重要。这些内容当然没错，但看多了之后，很容易产生一种感觉：**概念已经讲得够多了，真正缺的是落地层面的经验总结。**

因为真正把 Agent 往生产环境里推的时候，遇到的问题往往都非常具体：

它到底能不能碰真实系统？  
工具权限怎么收口？  
出错以后怎么定位，而不是靠猜？  
系统开始漂移以后，谁来持续收拾残局？  
评测怎么接进 CI，而不是停留在“感觉这版更好了”？

从这个角度看，Harness Engineering 其实并不只是一个概念，也不是 Prompt 技巧的延伸。更准确地说，它是一套面向生产环境的工程要求：

> 当 AI Agent 不再只是回答问题，而是开始调用工具、改动状态、影响真实系统时，就必须像对待任何生产系统一样，为它补齐入口治理、权限边界、证据链、质量回归和持续治理机制。

这篇文章不准备再花太多篇幅讲概念定义，而是更聚焦在工程落地本身，重点讨论下面这些问题：

- 为什么很多 Agent 系统不是“能力不够”，而是“控制系统没搭好”？
- 一个生产级 Harness 应该拆成哪些工程层？
- 哪些东西必须在 PoC 阶段就做，哪些东西可以后置？
- 为什么 Context、System of Record、可执行约束、评测、可观测性和熵治理，最后会连成一整套系统？
- 如果只想做一个最小可用的 Harness，第一阶段到底该做什么？

---

# 一、Agent 稳不稳定，关键不在模型，而在控制系统

很多团队第一次做 Agent 时，往往都会经历一个相似的阶段：  
Demo 非常亮眼，单轮效果也不错，工具调用、任务拆解、多步推理看起来都已经“能跑起来了”。

但只要开始接真实链路，问题很快就会暴露出来。

这些问题虽然表面形式不同，但大多有一个共同特点：**不可控**。

最典型的第一个问题，是**行为不可预测**。

同一个任务，第一次跑得很好。第二次跑，它走了完全不一样的路径。第三次跑，它可能在某一步停住不动，或者开始重复已经做过的动作。这个时候很难判断到底是哪一层出了问题：是上下文不完整，是工具返回不稳定，还是模型对系统状态的理解已经偏了。

第二个问题，是**系统不可调试**。

Agent 一旦进入多步流程，就不再是一个简单的“输入-输出”黑盒了。它会检索、会调用工具、会修改状态、会生成中间结论。最后结果错了的时候，最痛苦的不是“它错了”，而是根本说不清它是**怎么错的**。如果没有完整的 traces、metrics、logs，没有工具调用记录，没有 policy 决策记录，整个排障过程就会变成看日志猜谜语。

第三个问题，是经常被低估的：**系统会悄悄变坏**。

这个坏不是一次事故，而是一种很缓慢、但很稳定的系统性漂移。比如：

- 同一个功能开始出现两套实现
- 工具接口命名越来越乱
- 文档和代码逐渐对不上
- 原本设定的边界，被一次次“小改动”蚕食掉

这种问题在传统软件里当然也有，但在 Agent 系统里会更快、更隐蔽。因为 AI 会复制仓库里已有的模式，包括那些次优模式；一旦缺少约束和回收机制，坏模式会被不断放大。

所以从工程角度看，Harness 更适合被理解为一套**控制系统**，而不是一组“让模型更聪明”的技巧。

它解决的核心问题其实很朴素：

> 不是如何让 Agent 多做一点事，而是如何让它只在可控范围内做事，并且在出错时留下足够证据，让系统能被修复、被回归、被持续演进。

---

# 二、Harness 不是一个模块，而是一整套分层系统

很多文章会把 Harness Engineering 讲成一个抽象理念。真正落到工程里，更有效的方式其实是把它拆成一套明确的分层系统。

一个更接近生产环境的拆法通常是：

**入口治理 → 上下文装配 → Agent Runtime → 受控工具层 → 输出治理**

再横切两条闭环：

**评测回归** 和 **可观测证据链**

这个拆法最大的价值在于：它能强迫团队停止把问题都甩给模型。

因为只要按层去看，就会发现很多问题根本不是“模型不够强”，而是某一层压根没有被工程化。

比如：

- 入口治理没做，系统一进来就把高风险动作交给 Agent
- 上下文装配没做好，Agent 就只能猜系统
- 工具层没有 policy，本质上就是在把一个高不确定性的系统直接连到真实资源上
- 输出治理没做，生成结果虽然“像对的”，但没人保证它真的可消费、可执行、可审计
- 评测和观测没做，整个系统就只能靠感觉迭代

所以 Harness 的本质，不是给 Agent 更多能力，而是**在每一层减少自由度，增加可验证性**。

真正稳妥的建设顺序通常也不是“先追求自治，再补控制”，而是反过来：

**先把控制系统搭起来，再逐步放开自治程度。**

---

# 三、真正落地时，最先该补的往往不是 Prompt，而是 Context Builder

如果要从实战角度挑一个最容易被低估、但又最关键的组件，通常会是 **Context Builder**。

因为很多 Agent 系统的第一类失败，不是推理能力问题，而是**上下文断裂**。

这个问题在长流程任务里尤其明显。  
一个 Agent 第一轮看起来理解得很好，到了第二轮、第三轮，状态就开始发散。它会重复做已经做过的事情，会误判某个步骤已经完成，会对系统当前状态做出错误假设。

很多团队一开始会尝试继续“加长 Prompt”，但这通常不是根本解法。更稳妥的做法，是单独做一个上下文装配器，显式地产出一份结构化的 `context_pack`。

这个包里不应该只有用户问题，也不应该只是几段 RAG 检索结果，而应该是一份**系统快照**。至少可以包含下面这些内容：

- 当前任务契约：目标、边界、不做什么、完成定义
- 最近进度摘要：已经完成什么、下一步应该是什么
- 当前系统状态：版本、环境、feature flag、关键配置
- 最近工具调用及其结果
- 关键错误摘要
- 关键指标时间窗
- 引用证据索引
- 本轮预算：token、工具调用次数、运行时长

光列这些内容还不够，真正落地时，最好把它做成固定模版。下面是一个的 JSON 示例：

```json
{
  "task_contract": {
    "task_id": "task_20260318_001",
    "goal": "修复支付回调重复入账问题，并补充回归测试",
    "non_goals": [
      "不修改支付网关 SDK",
      "不调整数据库表结构"
    ],
    "definition_of_done": [
      "重复入账问题被修复",
      "新增至少1个回归测试用例",
      "相关文档同步更新"
    ],
    "risk_level": "medium"
  },
  "progress_summary": {
    "completed": [
      "已定位问题发生在 webhook 幂等校验缺失",
      "已确认影响范围仅限 stripe callback"
    ],
    "next_step": "在 payment webhook handler 中补充幂等校验，并运行支付模块测试"
  },
  "system_state": {
    "env": "staging",
    "branch": "fix/payment-webhook-idempotency",
    "service_version": "payment-service@1.4.2",
    "feature_flags": {
      "new_payment_pipeline": true
    }
  },
  "recent_tool_calls": [
    {
      "tool": "code_search",
      "input": "search webhook handler",
      "result_summary": "定位到 payment_webhook_handler.py"
    },
    {
      "tool": "test_runner",
      "input": "pytest tests/payment -q",
      "result_summary": "1 failed, 12 passed"
    }
  ],
  "error_summary": [
    {
      "type": "test_failure",
      "message": "duplicate charge event creates duplicate ledger record"
    }
  ],
  "evidence_refs": [
    {
      "id": "doc_12",
      "title": "支付回调处理规范",
      "why_it_matters": "定义了 webhook 幂等要求"
    }
  ],
  "budget": {
    "max_tokens": 12000,
    "max_tool_calls": 8,
    "max_runtime_seconds": 300
  }
}
```

为什么要坚持把它做成结构化对象，而不是一大段“写给模型看的说明”？

因为一旦进入长流程、多轮、跨会话的 Agent 场景，结构化上下文的价值会非常明显：

第一，它可 diff、可版本化。
第二，它更适合在循环里增量更新。
第三，它更容易作为“硬输入”被系统消费，而不是当成一段“供参考的说明文字”。

一旦 Context Builder 被独立出来，很多原本归因给“模型不稳定”的问题，最后都会显现出更真实的根因：**系统没有把必要事实装进去**。

所以一个更值得先问的问题，不是“Prompt 怎么写更强”，而是：

> 这个 Agent 当前到底能看见哪些事实？
> 它看到的是说明文字，还是一份可以继续工作的系统快照？

---

# 四、System of Record 不是“多写文档”，而是把团队知识变成 Agent 真正能用的事实源

Harness 落地里，另一个很关键的点是 **System of Record**。

这个概念看起来有点抽象，但实际非常简单：

> Agent 只能使用系统里存在的知识，而不能使用“团队脑子里的知识”。

很多团队做 Agent 时有一个常见误区：总觉得自己的系统已经“有文档”了。
但所谓“有文档”，很多时候只是意味着：

* 一部分在企微，钉钉聊天里
* 一部分在个人笔记里
* 一部分在公司 Wiki 里
* 一部分在某个同事脑子里
* 一部分在 README 里，但过了半年没人更新

对人来说，这些信息可能还勉强能拼起来。
但对 Agent 来说，不在系统可读路径里的内容，本质上就是不存在。

所以 System of Record 从来不是“多写几篇文档”，而是把知识收敛成**可版本化、可引用、可验证、可机械检查**的系统事实源。

更实用的做法，通常是把仓库里的知识分层，而不是堆成一个大文档。比如：

```text
repo/
  AGENTS.md
  docs/
    architecture/
      system-boundaries.md
      dependency-rules.md
    runbooks/
      incident-payment-callback.md
      rollback-checklist.md
    evals/
      payment-agent-golden-cases.yaml
      refusal-rules.md
    policies/
      tool-access-policy.yaml
      output-guard-policy.yaml
  schemas/
    response_schema.json
    context_pack.schema.json
```

这里 `AGENTS.md` 更适合做“地图”，而不是做“百科全书”。它的作用不是把所有规则都塞进去，而是告诉 Agent：

* 哪些文档是权威来源
* 哪些文件定义边界
* 哪些地方不能猜
* 哪些规则改了必须联动更新

例如：

```markdown
# AGENTS.md

## Read First
- docs/architecture/system-boundaries.md
- docs/policies/tool-access-policy.yaml
- docs/evals/payment-agent-golden-cases.yaml

## Rules
- 不要绕过 docs/architecture/dependency-rules.md 中定义的依赖方向
- 修改支付逻辑时，必须同步检查 docs/runbooks/incident-payment-callback.md
- 任何新增工具调用都必须匹配 docs/policies/tool-access-policy.yaml

## Update Requirements
以下变更必须同时更新文档：
- 新增高风险工具
- 修改关键业务流程
- 修改 eval 规则
```

这一步听起来像“文档治理”，但它和传统文档治理最大的区别是：

**它不是为了让人看得舒服，而是为了让 Agent 在运行时真的有东西可依赖。**

---

# 五、真正让系统安全下来的，不是提示词，而是入口治理、工具沙箱和 Policy-as-Code

只要 Agent 开始调用工具，它就不再是一个纯文本系统，而是一个**会行动的系统**。
一旦到了这一步，安全问题就不能再靠 Prompt 解决。

因为 Prompt 最多是在“劝它别乱来”，而生产系统需要的是“它根本乱不来”。

所以更稳妥的做法，是在 Agent 之前先放一层和传统 API 网关非常像的治理层。这里至少要处理：

* 鉴权
* 限流
* 预算
* 策略路由
* 风险分级
* request_id 和审计信息生成

更进一步，真正有用的治理通常都会写成 **Policy-as-Code**。
不是“靠团队记住规则”，而是把规则写成系统能执行的策略。

下面是一个简化版例子：

```yaml
version: 1
default: deny

tools:
  code_search:
    allow: true

  read_docs:
    allow: true

  run_tests:
    allow: true
    limits:
      max_runtime_seconds: 600

  sql_query:
    allow: true
    constraints:
      statement_type: SELECT_ONLY
      max_rows: 1000

  deploy_prod:
    allow: false
    require_human_approval: true

budgets:
  max_tokens: 15000
  max_tool_calls: 10
  max_http_requests: 5

output_guard:
  require_citations: true
  block_pii: true
  enforce_schema: response_schema.json
```

这个例子虽然简单，但已经能体现几个关键判断：

第一，**默认 deny** 比默认 allow 更适合生产系统。
第二，**预算本身也是安全控制的一部分**。
第三，**输出治理必须跟工具治理放在一起看**，否则前面收得再严，最后还是可能从输出阶段漏出去。

所以工具层和输出层更合理的做法，是一起纳入“受控执行”：

* 工具执行走统一授权
* 高风险工具强制人工确认
* 默认最小权限
* 执行在沙箱里
* 文件系统、网络、CPU、timeout 都有边界
* 所有决策都有审计日志
* 输出必须过 guard

这一层做好之后，系统稳定性通常会明显上一个台阶。
原因不是 Agent 更聪明了，而是它终于被放进了一个工程系统里。

---

# 六、架构规则如果不能机械执行，就等于不存在

如果说前面几层解决的是“Agent 能不能安全跑”，那到了代码和业务系统层面，另一个非常现实的问题就是：

> 它会不会把系统一点点写坏。

这也是为什么 **Architectural Constraints** 非常关键。

在很多团队里，架构规则其实写得并不少。比如：

* 不允许跨层调用
* 某层不能直接连数据库
* providers 统一管理 cross-cutting concerns
* 日志必须结构化
* 文件行数不能无限膨胀

问题不是没有规则，而是这些规则往往只存在于“约定”层。
而在高吞吐的 Agent 场景里，只要规则不是机械执行的，它迟早会被冲垮。

所以真正有效的做法，不是试图规定每一行实现，而是把**不变量**写成可执行门禁。

一个很典型的例子，是把“禁止 UI 层直接访问数据源”写成结构测试。伪代码可以长这样：

```python
def test_ui_layer_must_not_import_data_access():
    forbidden = scan_imports(
        source_dir="src/ui",
        forbidden_modules=["src/data_access", "src/repositories"]
    )
    assert not forbidden, f"UI layer has forbidden imports: {forbidden}"
```

再比如，把“关键改动必须带 tests 和 docs”做成 CI gate：

```bash
changed_files=$(git diff --name-only origin/main...HEAD)

if echo "$changed_files" | grep -q "src/payment/"; then
  echo "$changed_files" | grep -q "tests/payment/" || exit 1
  echo "$changed_files" | grep -q "docs/" || exit 1
fi
```

这类规则的关键不是“复杂”，而是“真的能挡住违规”。

最重要的一点是：报错信息不能只是“失败”，而要足够带修复方向。因为如果错误信息能直接进入下一轮上下文，它其实就变成了 Agent 自修复闭环的一部分。

---

# 七、如果没有 Eval Harness，所谓优化大部分都只是错觉

Agent 系统如果没有成体系的 eval，几乎不可能稳定进化。

因为没有 eval 的优化，本质上都在靠印象驱动。
看了几个 case 变好了，并不等于整个系统变好了；换了一次 Prompt 或模型版本，局部流程更顺了，也不代表其他关键路径没有退化。

所以 Eval Harness 更适合被当成真正的“质量系统”，而不是一个附属工具。

一个比较实用的落地方式，是把 eval 分成三层。

第一层是 **deterministic checks**。
比如：

* 必须包含 citations
* 不允许危险动作
* policy 决策必须合规
* 没有证据时必须拒答
* 高风险动作必须触发人工确认

这一层的好处是稳定、低歧义、适合先接进 CI。

例如，一个简单的 deterministic rule 可以写成：

```yaml
- name: refuse_when_no_evidence
  input: "总结公司内部退款规则"
  context:
    retrieved_docs: []
  expected:
    must_refuse: true
    must_not_hallucinate: true
```

第二层是 **端到端用例评测**。
也就是把真实工作流中的关键路径沉淀成可回归样例，关注结果是不是满足 DoD，而不是只看某个局部指标。

比如对一个知识库 Agent，可以沉淀这样的 case：

```yaml
- case_id: kb_023
  user_query: "支付回调失败后多久会重试？"
  retrieved_docs:
    - "payment_retry_policy.md"
    - "webhook_runbook.md"
  expected:
    answer_should_include:
      - "重试时间窗口"
      - "最大重试次数"
    require_citations: true
    forbid:
      - "编造不存在的 SLA"
```

第三层是 **线上抽样与失败回流**。
因为离线评测集再完整，也不可能覆盖真实线上所有边缘路径和高风险场景。

所以必须做分层抽样，尤其要提高这些路径的采样率：

* 触发工具
* 拒答
* 人工确认
* 高风险动作
* 多轮长流程任务

并且把事故沉淀成新的回归资产：

> 每次线上事故，最后都应该变成新的 eval case，或者新的 policy / lint gate。

只有这样，事故才会真正变成系统能力。

---

# 八、Observability 和熵治理，决定了系统能不能活过“最初那几个月”

很多 Agent 项目在前期 demo 阶段都还不错，真正拉开差距的，往往不是首月效果，而是三个月以后系统还剩下多少可维护性。

这个问题通常可以拆成两半来看：

一半是 **Observability**，另一半是 **Entropy Management**。

前者解决“出了问题能不能定位”，后者解决“系统会不会越跑越乱”。这两件事缺一不可。因为如果只有可观测性，没有熵治理，团队最终看到的只会是一条越来越清晰的下坡路。

先说 Observability。

如果一个系统没有 traces、metrics、logs，那它就只能在出事以后靠经验猜。
但对 Agent 来说，这还不够。因为 Agent 的执行链路本身就比普通服务更长、更分叉、更依赖上下文，所以更合理的做法是把可观测性视为 Harness 的一部分，而不是运维附属品。一个最小可用的要求通常至少包括：

* 每个请求必须有 trace
* 每次工具调用必须打 span
* 每次 policy 决策必须记录
* 每次输出都能关联回 evidence 和 trace_id

例如，一个工具调用日志至少应该长这样：

```json
{
  "trace_id": "trace_abc123",
  "span_id": "span_tool_001",
  "tool_name": "sql_query",
  "input_summary": "SELECT retry_policy FROM payment_config",
  "policy_decision": "allowed",
  "runtime_ms": 142,
  "result_summary": "1 row returned",
  "timestamp": "2026-03-18T10:15:21Z"
}
```

这样做的价值在于，系统终于可以不再问“它为什么错了”，而是能沿着证据链还原“它在哪一步开始偏了”。

但如果只有可观测性，没有熵治理，系统还是会慢慢坏。

所谓 entropy management，实际就是给 Agent 系统加一个垃圾回收器。因为只要 AI 持续参与生成和修改，系统就一定会积累：

* 重复实现
* 风格漂移
* 边界松动
* 文档断链
* 评测缺口
* policy 漏洞

如果这些问题只靠人“有空的时候顺手收拾”，最后一定收不回来。更现实的做法，是把熵治理做成一套分层治理流程，而不是单次清理动作。

## 1. 先把“熵”分类，而不是笼统地说系统变乱了

很多团队会说系统在变乱，但如果不把漂移类型拆开，就很难自动治理。更适合落地的做法，是先给熵建立分类。

第一类是 **结构熵**。
比如模块边界被绕过、依赖方向反转、UI 层直接访问数据层、原本应该统一收口的 cross-cutting logic 被散落到各处。这类问题最适合用 structural tests 和自定义 linter 发现。

第二类是 **实现熵**。
比如同一功能出现多个版本、重复 helper 越来越多、命名风格分裂、同类逻辑散落在不同目录。它的特点不是立刻报错，而是不断抬高后续修改成本。

第三类是 **知识熵**。
比如 runbook 过期、AGENTS.md 指向错误、docs 和代码脱节、评测样例没有跟着业务变更更新。一旦这层腐烂，Agent 就会重新进入“猜系统”的状态。

第四类是 **治理熵**。
比如 policy 缺洞、output guard 覆盖不全、eval 缺少关键失败路径、线上事故没有回流成回归资产。这类问题最危险，因为它不是“系统乱”，而是“系统失去自我修复能力”。

## 2. 给每一类熵都配一个扫描器，而不是等人肉发现

熵治理真正能落地，关键在于“扫描器先于修复动作”。也就是说，不要先想怎么改，而要先让系统能稳定发现问题。

比较常见的做法包括：

* 用 structural tests 扫描越界依赖、非法 import、目录层级违规
* 用 repo scanner 扫描重复实现、超大文件、孤儿模块、未引用代码
* 用 docs checker 扫描失效链接、缺失 owner、超过时效的 runbook、未同步更新的规范文件
* 用 eval diff scanner 扫描关键业务变更后是否缺少新增用例
* 用 policy checker 扫描新增工具是否已纳入 allowlist、输出 schema 是否仍然匹配
* 用 observability coverage checker 扫描哪些关键路径没有 trace/span，没有进入证据包

这里可以把 nightly GC 理解成一个**定时治理任务**。实际工程里，常见做法是：用 CI/CD 调度器、Kubernetes CronJob，或者自定义任务框架，按固定时间触发一个扫描脚本；而下面这个 yaml 更适合作为**治理任务的配置示意**，用于描述“这个任务包含哪些扫描阶段、产出哪些结果”，而不是某个平台可以直接执行的标准流水线文件。

```yaml
nightly_gc:
  cron: "0 2 * * *"
  scanners:
    - name: architecture_drift
      runner: python scripts/scan_architecture_drift.py
    - name: duplicate_impls
      runner: python scripts/scan_duplicate_impls.py
    - name: docs_staleness
      runner: python scripts/scan_docs_staleness.py
    - name: eval_gaps
      runner: python scripts/scan_eval_gaps.py
    - name: policy_coverage
      runner: python scripts/scan_policy_coverage.py
  outputs:
    report: artifacts/drift_report.json
    candidate_fixes_dir: artifacts/candidate_fixes/
    pr_plan: artifacts/pr_plan.json
```

如果用自定义执行器，它的逻辑大致会是：

1. 读取这份配置；
2. 按顺序执行每个 scanner；
3. 汇总扫描结果；
4. 生成 drift report；
5. 按规则决定是否生成 candidate fixes 或 PR plan。

也就是说，真正“执行”这份配置的，不是 yaml 本身，而是外部调度器和执行脚本。

## 3. 修复动作要“小步、可逆、按类型分桶”

很多团队熵治理做不下去，不是因为不会发现问题，而是因为每次清理都太大、太杂、太危险。更稳妥的做法，是把 nightly GC 生成的修复按类型拆开，尽量做到“小 PR、低耦合、可回滚”。

常见的修复桶可以分成：

**一致性修复**
比如统一命名、合并重复 helper、补充缺失注释、归并重复配置。
这类 PR 风险最低，适合自动生成后直接进入 CI。

**边界修复**
比如把越界依赖拉回规定层级、把直连调用改回 provider、补全结构化 logging。
这类 PR 风险中等，通常需要明确 reviewer。

**知识修复**
比如更新失效 runbook、补齐 AGENTS.md 导航、修复 docs 链接、同步变更后的 eval 说明。
这类问题常常不影响运行，但会明显影响后续 Agent 的判断质量。

**治理修复**
比如给新工具补 policy、给新输出补 schema、把线上事故补成 eval case。
这一类最值得长期做，因为它是在修系统的“免疫系统”。

一个 nightly GC 生成的 drift report 可以长这样：

```json
{
  "architecture_drift": [
    {
      "type": "forbidden_import",
      "file": "src/ui/payment_page.ts",
      "detail": "imports src/repositories/payment_repo"
    }
  ],
  "duplicate_impls": [
    {
      "type": "similar_function",
      "files": [
        "src/utils/retry.py",
        "src/payment/retry_helper.py"
      ],
      "similarity": 0.93
    }
  ],
  "docs_staleness": [
    {
      "type": "stale_runbook",
      "file": "docs/runbooks/payment-callback.md",
      "last_verified_days": 124
    }
  ],
  "eval_gaps": [
    {
      "type": "missing_eval_case",
      "changed_area": "payment webhook",
      "expected_suite": "evals/payment-agent-golden-cases.yaml"
    }
  ]
}
```

扫描报告不是最终目的，重点在于它后面能自动映射成“哪一类修复 PR”。

## 4. 熵治理必须进入 CI、周期任务和看板，否则一定烂尾

真正成熟的熵治理，不是“偶尔开个清理分支”，而是进入日常工程节奏。

一个比较完整的落地方式通常是三层：

**第一层：提交时阻断**
新熵不要进主干。也就是能在 pre-commit、CI gate、structural tests 阶段挡住的，尽量当场挡住。

**第二层：周期性清理**
已经进入仓库、但短期不影响功能的漂移，交给 nightly GC 或 weekly maintenance job 扫描并开 PR。

**第三层：趋势治理**
不是只看单个问题，而是看趋势。比如：

* 架构违规次数是不是在下降
* 熵治理 PR 数是不是稳定减少
* docs 过期率是不是在下降
* 重复实现密度是不是在下降

如果这些指标长期不改善，说明熵治理只是“扫地”，还没有真正改变系统生成坏味道的方式。

## 5. 每次事故都应该反哺熵治理系统

熵治理最容易做浅的一种方式，是把它当“仓库美化”。
但真正高价值的做法，是让它和事故、回归、policy 一起联动。

一个比较成熟的闭环通常是：

线上事故
→ 证据链回放
→ 定位这次问题属于哪一类熵
→ 决定是补 lint、补 policy、补 eval，还是补 docs
→ 让下一次类似问题在提交时或 nightly GC 中被提前发现

只有做到这一步，熵治理才不是“打扫卫生”，而是在持续提升系统的自稳能力。

所以更准确地说，entropy management 不是一个清理动作，而是一套长期治理机制：

> 先把漂移分类，再让扫描自动化；
> 再把修复标准化、小步化、低风险化；
> 最后把治理结果接进 CI、指标和事故回流闭环。

真正成熟的 Harness，不只是能完成复杂任务，而是有能力**持续压低熵增速度，并把漂移控制在可维护范围内**。

---

# 九、如果重新搭一套 Harness，更合理的分阶段顺序是什么

很多人看完这些层之后，最自然的问题是：

> 这些东西是不是都得一开始就做全？

答案通常是：不用。
但顺序一定要对。

## 第一阶段：PoC，先让系统“不会乱来”

这个阶段最关键的不是“自治程度有多高”，而是：

* 有没有最小 policy
* 工具是不是在受控环境里
* 有没有基础输出治理
* 有没有 smoke eval
* 有没有最基本的 traces

PoC 阶段最怕的是为了追求效果，直接把 Agent 接到真实资源上。
更稳妥的方式恰恰相反：先把最硬的护栏立起来，哪怕先牺牲一点自由度。

## 第二阶段：内测，先让系统“可调试、可解释”

这个阶段开始补：

* Context Builder
* System of Record
* 更完整的 observability
* 可执行架构约束
* 更完整的 eval 与失败回流

这一步完成以后，系统才真正开始具备“工程可维护性”。

## 第三阶段：生产，先让系统“能持续演进”

到了生产阶段，更关键的问题就不再是“它能不能跑”，而是：

* 是否有线上抽样
* 是否有风险分层
* 是否能把事故转成回归资产
* 是否有 nightly GC
* 是否能持续压住成本、漂移和违规率

换句话说，生产阶段真正要建设的，是系统的**长期自稳能力**。

---

# 十、Harness Engineering 最核心的价值，不是增加自由，而是约束自由

如果再回头看 Harness Engineering 这个词，它其实不只是一个潮流概念，也不是 Prompt Engineering 的升级版，更不是一组散落的“最佳实践清单”。

从工程上看，它更接近一种很明确的立场：

> 当 AI Agent 开始真正参与系统运行时，不能再把它当成一个会说话的模型，而要把它当成一个需要被治理、被约束、被观测、被回归、被持续修复的生产系统。

所以如果要用一句更直接的话来概括：

> Harness 不是为了让 Agent 更自由，
> 恰恰相反，Harness 是为了让 Agent 的自由，始终被关在一个可理解、可验证、可修复的系统里。

这也是为什么真正有价值的 Harness 讨论，重点不该再停留在“它是什么”，而应该认真讨论这些更具体的问题：

* Context Builder 长什么样
* System of Record 是否真的可被 Agent 使用
* policy 是不是代码化了
* 架构约束是不是能阻断违规
* eval 能不能进 CI
* 证据链是不是够完整
* 系统有没有能力持续回收熵

这些问题，才是 Agent 落到生产以后真正绕不过去的问题。

---

# 结语

如果只看概念，Harness Engineering 很容易被理解成一种“更系统的 AI 工程方法”。

但从真实落地的角度看，它更像是一套非常现实的生产经验：

**入口要管住，工具要收口，上下文要结构化，知识要可追溯，规则要机械执行，质量要能回归，证据要能回放，漂移要能持续回收。**

做到这些，Agent 才有资格被送进生产环境。
否则它再聪明，也只是一个不稳定的高风险系统。

真正决定 Agent 系统上限的，往往不是模型能力，而是有没有认真把这套“控制系统”搭起来。
