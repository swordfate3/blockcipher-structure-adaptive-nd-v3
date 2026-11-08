# Learnings

用户纠正、踩坑经验、知识缺口与更好的做法。

**Categories**: correction | knowledge_gap | best_practice
**Areas**: frontend | backend | infra | tests | docs | config
**Statuses**: pending | in_progress | resolved | wont_fix | promoted | promoted_to_skill

## 条目格式

见 self-improvement 技能的「学习条目模板」。ID 用 `LRN-YYYYMMDD-NNN`（当天三位递增）。

每条至少包含：

```markdown
**Logged**: YYYY-MM-DDTHH:MM:SSZ
**Priority**: low | medium | high | critical
**Status**: pending
**Area**: frontend | backend | infra | tests | docs | config

### Summary
一句话：学到了什么

### Details
发生了什么、错在哪、正确做法

### Suggested Action
下次遇到同类情况的具体动作

### Metadata
- Source: user_feedback | error | conversation
- Related Files: path
- Tags: tag1, tag2
- Decay: reality | decision | dependency | relevance | unknown | fast | medium | slow | durable — 一句依据
```

`Decay` 用于判断这条经验何时会过期。要做复核或提升时，再补完整的
`### Context Maintenance` 块；规则见 self-improvement 技能的
`references/context-maintenance.md`。

## 状态定义

| Status | 含义 |
|--------|------|
| `pending` | 还没处理 |
| `in_progress` | 正在处理 |
| `resolved` | 已解决或知识已整合（补 Resolution） |
| `wont_fix` | 决定不处理（必须写原因） |
| `promoted` | 规则已写进 AGENTS.md 等常驻文件 |
| `promoted_to_skill` | 已抽成独立技能（补 Skill-Path） |

## 抽成技能时补字段

```markdown
**Status**: promoted_to_skill
**Skill-Path**: ~/.pi/agent/skills/skill-name
```

---

## [LRN-20261030-001] best_practice

**Logged**: 2026-10-30T14:18:42Z
**Priority**: high
**Status**: pending
**Area**: tests

### Summary
有界密码调用必须按每次测试与收尾复核累计，工程验收不豁免授权

### Details
上一合法key周期合同只允许<=24次公开工程encrypt比较；反复pytest和收尾witness重算未计入累计，后来用约80次掩盖了不可靠的精确总数，且验收额外重算超过授权。新pair扫描明确生产encrypt=0，用符号witness的packed32数学复核，保留旧原件并在结果披露偏差。

### Suggested Action
正式前固定调用账本与测试筛选；运行每条可能生成输出的测试前扣额度；收尾不得再次重算已核点，除非合同明确预算；超额即披露并停，不追溯批准。

### Metadata
- Source: error
- Related Files: docs/INNOVATION2-INTEGRAL-PAIR-SCAN-RESULTS.md,validation/integral-pair-scan/accept.py
- Tags: budget,validation,authorization
- Recurrence-Count: 1
- First-Seen: 2026-10-30
- Last-Seen: 2026-10-30
- Decay: durable — 权限与证据记账纪律

---

## [LRN-20261102-001] correction

**Logged**: 2026-11-02T05:33:55Z
**Priority**: high
**Status**: pending
**Area**: docs

### Summary
研究推进计划必须覆盖多次实验与条件分支，不以单次下步建议代替完整路线

### Details
用户明确要求创新2计划涵盖多次比较、具体做法、错误和阴性结果后的下一步，以及编码和代理分工。此前回复仅列单项下次方案，未满足完整推进需求。

### Suggested Action
先比较总体路线，形成有阶段依赖、固定候选族、累计预算、成功/失败/UNKNOWN分支和独立确认的研究设计；批准后再写实施计划并按实际代理路由委派。不得许诺无限试验必得好结果，不把路线批准当GPU/SSH/发布许可。

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md,docs/INNOVATION2-CURRENT-STATE.md
- Tags: research,planning,delegation
- Recurrence-Count: 1
- First-Seen: 2026-11-02
- Last-Seen: 2026-11-02
- Decay: decision — durable，用户对本项目研究计划的明确要求

---

## [LRN-20261103-001] correction

**Logged**: 2026-11-03T07:07:10Z
**Priority**: high
**Status**: pending
**Area**: docs

### Summary
推荐下一研究路线前先核查已做历史，未找到记录不等于没做过

### Details
在本轮神经预测阴性后，未先核对历史核实验便把核方法推荐为新主方向，用户指出之前已经做过。当前有限本地检索未定位该批核实验；不能据此否认用户历史或编造核种类和结果，也不能把GF(2)消元的kernel零空间误当机器学习核。

### Suggested Action
先定位既有核实验合同、配置与结果，列出已测范围和新提案的实质差异；定位前撤回新颖方向与重跑建议，必要时只询问一个批次定位线索。

### Metadata
- Source: user_feedback
- Related Files: none
- Tags: research,history,kernel,duplicate-work
- Recurrence-Count: 1
- First-Seen: 2026-11-03
- Last-Seen: 2026-11-03
- Decay: durable — 研究推进须对照历史证据

---

## [LRN-20261103-002] correction

**Logged**: 2026-11-03T17:20:58Z
**Priority**: high
**Status**: pending
**Area**: docs

### Summary
核积分须先定位论文，不能误拆成机器学习核与传统积分两条路线

### Details
用户所说核积分指向2026欧密会相关神经启发积分研究，而非通用机器学习核回归。只读检索定位两篇相关文献：EUROCRYPT2026的Neural-Inspired Advances in Integral Cryptanalysis（ePrint2025/852），以及其线性代数kernel后续Improving Neural-Inspired Integral Distinguishers via a Linear-Algebraic Approach（ePrint2026/340）。后者摘要的kernel指GF(2)经验密文奇偶矩阵的零空间，会议归属未据此确认。此前将GF(2)kernel一概排除为无关也不正确；需区分旧拟合矩阵零空间和积分parity矩阵零空间。2026-11-04进一步发现两篇PDF及提取文本早已在本地papers/innovation_two/；先前仅因在线403便说全文缺失，是漏查本地资料。PDF首部还显示两位首作者Liu Zhang、Yunjae Hwang，网页摘要提取曾遗漏；Zhang本地为arXiv v1，不能当最终会议版。

### Suggested Action
先查本地papers清单/PDF/文本与历史实现，再查在线资料；联网失败不代表本地无全文。按PDF首部核作者、版本和来源，文件名年份或网页摘要不替代版本核查。再解释输入cube、跨key奇偶矩阵、输出mask零空间和伪mask排查；有限经验零空间候选不当作全key证明。不要在未定位历史实现前称全新或未做过。

### Metadata
- Source: user_feedback
- Related Files: papers/innovation_two/sources/paper_manifest.csv,result/20261104-innovation2-kernel-integral-audit/report.md
- Tags: integral,kernel,paper,terminology,local-inventory,version
- Recurrence-Count: 2
- First-Seen: 2026-11-03
- Last-Seen: 2026-11-04
- Decay: durable — 论文方法讨论的消歧和证据边界

---

## [LRN-20261105-001] knowledge_gap

**Logged**: 2026-11-05T12:13:41Z
**Priority**: high
**Status**: pending
**Area**: docs

### Summary
积分扩轮须分别核输入逆像、输出复制和逆轮观测量，不能用一般非线性直觉代替具体代数。

### Details
曾误称SIMON旧高字活跃{0,2}结构在非零背景下逆像非线性。实际F逆(U,V)=(V,U XOR f(V))，V在该组固定，f(V)为常量，因此任意背景都仍是低字{32,34}轴cube。旧证书的任意背景／独立轮key全称还涵盖合法schedule，允许输入加一轮。进一步h_r=X_r[j] XOR Y_r[j+1]Y_r[j+8] XOR Y_r[j+2]=X_(r-2)[j] XOR k_(r-1)[j]，偶数点消去key常量；旧r7可派生r9二次关系，再输入扩展得r10，但不能当作新核心搜索或新纪录。独立顾问也曾混淆X8与Y8，已由独立审查纠正。

### Suggested Action
先逐式代入，明确同组固定量、量词、轮偏移及观察位置；总轮数与核心基底分开；拒绝未核的排他性结论。数学推论、经验确认、新颖性分别报告。

### Metadata
- Source: conversation
- Related Files: result/20261105-innovation2-split-cancel-audit/report.md
- Tags: integral,feistel,pullback,quantifiers
- See Also: LRN-20261103-002
- Pattern-Key: cryptanalysis.extension_quantifiers
- Recurrence-Count: 1
- First-Seen: 2026-11-05
- Last-Seen: 2026-11-05
- Decay: durable — 具体代数优先于泛化直觉

---

## [LRN-20261106-001] correction

**Logged**: 2026-11-06T01:45:03Z
**Priority**: high
**Status**: promoted
**Promoted**: AGENTS.md
**Promoted-Date**: 2026-11-06
**Area**: docs

### Summary
结果交付须固定在现有 result，三件套同时具备图解、结果图与默认本地 HTML 预览

### Details
用户明确要求后续结果报告继续写入 /workspace/result/，沿用本次 report.md、index.html、source.json 格式，但必须图文并茂，包含方法/流程图解和基于保存结果的结果图；仅有文字和表格不够。交付时默认启动或复用本地静态 HTML 服务并提供可用链接。当前 AGENTS.md 尚未明确此要求，先展示拟增规则得到确认再提升。

### Suggested Action
交付前检查 result/独立结果目录的三件套、至少一幅方法图解和一幅真实保存结果图，标明来源和限制；不为绘图重新训练或加密。复用或启动仅服务 result/ 的本地静态服务，实测 HTTP 后提供链接；无法从用户端验证时明确说明。不默认改监听范围、端口转发或防火墙。

### Metadata
- Source: user_feedback
- Related Files: result/README.md,AGENTS.md,result/20261105-innovation2-split-cancel-cycle/index.html
- Tags: result-delivery,visualization,html-preview
- Recurrence-Count: 1
- First-Seen: 2026-11-06
- Last-Seen: 2026-11-06
- Decay: decision — durable，用户明确的本项目交付要求，除非后续变更

### Resolution
- **Resolved**: 2026-11-06T01:58:12Z
- **Commit/PR**: N/A
- **Notes**: 已先展示三条规则 diff，用户回复“是”后原文写入 AGENTS.md 的“结果报告交付”小节。仅提升交付规则，未改科学结果、未补图或重跑实验；验收条件为结果三件套、方法图解与结果图、图表来源限制、本地 HTTP 响应及链接，实际用户设备可达性另行说明。

---

## [LRN-20261106-002] correction

**Logged**: 2026-11-06T04:47:47Z
**Priority**: high
**Status**: pending
**Area**: docs

### Summary
结果先说成果，设计另给简练通俗版；用具体事实取代术语堆叠

### Details
用户先指出 Split-and-Cancel 汇报云里雾里，随后明确要求直接说做成什么，删去连续的“不是什么、没有什么、不能算什么”等冗余话语。同一任务再次解释时又漏掉密码算法、轮数、数据量和技术方法，用户仍无法理解。问题是把验收、限制和术语放在成果前面，又将“简短”误作省略理解所需的基本事实。此前同一结果解释任务的补充不计复发。后续独立的11—14轮规模设计任务又以长篇技术设计作为用户主要阅读材料，用户明确要求以后设计同时更新一版简练、通俗易懂的说明，阅读技术版很难受；本次跨结果交付与方案设计两个任务，复发计数增至2。

### Suggested Action
默认先用一句话说明核心收获，再列两到三项具体成果；必须同时交代算法名称与基本规格、研究轮数及完整版轮数、数据如何构造及实际数量、技术步骤和产出用途。用一张图贯通输入→方法→结果，必要范围直接嵌在成果表述中，例如“相对预列基线增加一条”，避免成串否定；简洁不能牺牲基本事实，证据细节按需展开。以后每次设计及重要修订，同时提供与技术版一致的通俗简版，作为默认给用户看的入口：用短句说明要解决什么、具体怎么做、用多少数据和时间、什么结果继续或停止、需要用户确认什么；必要术语首次用白话解释，模块名、证书格式和审查细节留在技术版。预算必须区分每组、单项和全周期，简版不删掉授权所需的总成本或风险。验收标准是用户无需先读技术长文，就能理解目的、流程、成本与决策。

### Metadata
- Source: user_feedback
- Related Files: result/20261105-innovation2-split-cancel-cycle/report.md,docs/superpowers/specs/2026-11-06-innovation2-integral-scale-design.md,result/20261106-integral-scale-proposal/index.html
- Tags: communication,outcome-first,concise-reporting,plain-language-design
- Pattern-Key: communication.reader_first_summary
- Recurrence-Count: 2
- First-Seen: 2026-11-06
- Last-Seen: 2026-11-06
- Decay: decision — durable，用户明确的结果沟通偏好，除非后续变更

---

## [LRN-20261106-003] correction

**Logged**: 2026-11-06T08:40:17Z
**Priority**: high
**Status**: pending
**Area**: docs

### Summary
研究周期须明确从功能小样本转入目标规模的门槛，主动用预算检验实质研究目标

### Details
用户追问为何一直低数据量试水，并指出若不主动提醒可能持续如此。本轮将四点结构固定为唯一输入族，大部分真实加密用于重复核对已选关系，却未在规划时说明功能验证完成后如何检验更有力度的轮数—数据成本目标。既有研究 skill 已禁止因小、易测、跑得快就选主方案，问题在执行与验收中没有落实。还须区分每组结构大小、结构候选覆盖、经验核对次数与符号搜索能力，不能把增加总调用数或活跃位数本身当成推进轮数的充分条件。

### Suggested Action
每个研究周期先写清目标改进、同任务文献对标、达到目标所需的数据/计算规模及依据；给功能验证设置退出条件，并预列转入正式规模、换方法、停止的触发条件。基础验证通过后主动提出下一有界规模方案，按目标增益分配搜索与独立验证预算；若维持小规模必须说明正在检验的具体假设。仍按用户明确授权执行，不自动扩大数据、GPU、SSH或重试。

### Metadata
- Source: user_feedback
- Related Files: .pi/skills/cipher-research-workflow/SKILL.md,docs/superpowers/specs/2026-11-05-innovation2-split-cancel-design.md
- Tags: research,scaling,exit-gates,budget-allocation
- See Also: LRN-20261102-001
- Pattern-Key: research.pilot_to_target_scale
- Recurrence-Count: 1
- First-Seen: 2026-11-06
- Last-Seen: 2026-11-06
- Decay: decision — durable，研究推进须由科学目标决定规模，用户明确纠正

---
