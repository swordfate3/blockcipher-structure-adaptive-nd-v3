# Cube 固定数据50epoch早停 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development，TDD/独立审查。主控独占真实运行和Git提交发布；代理不再派代理。

**Goal:** 在同一百万对数据和原网络上，以最多50epoch/独立DEV早停验证更长训练，不重新加密、换key或加载旧best续训。

**Architecture:** 从af784fcb独立源码副本对具名fit加入可选core早停；cube_compare新增严格延迟读取provider，通过原同一双臂训练入口装配。保留旧5epoch模式；CLI按固定配方区分新数据复用模式，脚本只修原生退出捕获，不增加调度器。

**Tech Stack:** 既有Python3.12本地、Windows Python3.13.3/Torch2.7.1+cu118/NumPy2.4.0，原RunStore/Meter/core EarlyStopping、pytest/Ruff/mypy；不安装。

**Spec:** `docs/superpowers/specs/2026-11-23-innovation2-cube-50epoch-design.md`，APPROVED（用户已确认简短设计的50epoch/早停/3h/发布运行范围）。

## Global Constraints

- 新科学运行2fit，max_epochs50/warmup10/patience8/min_delta1e-5，独立早停，batch256、原FP32/Adam/lr/seed/结构卡/128BCE训练/active8DEV和TEST不变。
- 从原-02的3个NPZ只读复用：源run SHA和逐split SHA/bytes见spec；只TRAIN/DEV先读取，两新best持久重载成功后才读取TEST。新cipher0/秘密熵0/数值预检0，不重新划分。
- cap409600updates/111542272forward/null0，成功按实际两臂完整epoch核账；wall10800/CPU43200/RSS8GiB/output512MiB/ckpt16MiB；一次运行，retry0。
- 开发只人工tensor结构/fakeoracle/runtime/clock/FFI；不得真实cipher/秘密熵/NN数值前后向/Adam/RSS/GPU/SSH/安装/Git写入或读旧权重；旧真实数据仅主控核SHA、生产只读加载。
- 仅限定fit/cube_compare/config新配方/必要脚本及tests/docs；core、模型架构与原5epoch行为不改。主main/index不动，源基于af784fcb，所有模块纳入source指纹。
- 新tests全局唯一命名，pytest新owner显式目录，forcedRuff no-respect-gitignore；不能把默认测试没收集新owner说成全覆盖。

## Review Focus

1. 源身份/NPZ坏schema/重复成员/对象dtype/超大声明/族跨split或缺变体：失败闭合，不重采样；SHA前校验，标签不提前读TEST。
2. min_delta与best选择分离，warmup off-by-one/累计小改善/各臂独立/末轮上限分类；旧5epoch默认不受影响。
3. 两实际完整epoch数对应精确attempt/completed计量；maxcap不是成功必需数，早停不是资源故障豁免。
4. 第二best读回失败或第二TEST失败不持久化partial结果；复用旧数据不能扫描旧模型/旧TEST成绩，也不能无意调用旧preflight/cipher/秘密key。
5. Windows原生退出0/7稳定捕获、不以stdout补证；保持SSH前台、未知则未知、不新增服务调度器或自动重试。

## Task 1: 只读数据复用＋独立早停＋CLI/计量/脚本最小接入

**状态：COMPLETE。** implement-deep38324113；独立审查0b177113经Fix1放行；整体review bf57cf1c Approved。主控fresh173人工+早停/layout、forcedRuff全树、strictmypy380文件通过；修后default fullsafe3702pass49deselected407.06s（既存ZIPwarning1）。Core/model不改，14项产品/测试差异。

**Files:**
- Modify `cipher-lab/plugins/cipher_lab_plugins/direct_block/fit.py`：`FitSettings`加可选早停设置，复用core `EarlyStopping`；不改legacy默认。
- Create `.../cube_compare/reuse.py`：受限逐split原run/data读取与NPZ解析。
- Modify `.../cube_compare/{config,training,cli,report}.py`：新固定50reuse配方、明示mode/provider、实际epoch计量/结果/来源。
- Modify `cipher-lab/scripts/start-cube-sharedm.ps1`：最小可选config/source参数、10800期限/保留旧默认；修复原生exit作用域/捕获。
- Create `cipher-lab/configs/direct-block-cube-sharedm-simon7-50epoch.json`。
- Modify/Create `.../cube_compare/tests/test_cube_{fit,reuse,training,cli,report}.py`及必要guard，globalunique。

**Interfaces:**
- 旧`config.recipe()`不变；新增`reuse_recipe()`（名称可用明确常量）固定source run/SHA/NPZ metadata/50早停/caps，无science import。
- `SavedPanels(source_root:Path, meter:Meter, ...)`/等价typed factory保持原`build('train'/'dev')`和`label_test()`协议，可显式读原bytes给新RunStore写dataset，不调用cipher/秘密key。来源root从CLI显式传入，不hardcode本机路径。
- `FitSettings`可选`Settings|None`；每臂fit构造单key EarlyStopping，DEV完成后更新，正常停止并保存actualepochs/termination，返回best原契约不变。
- 训练装配共用同一provider/order及原双reload/test/score，science成功按新实际epoch公式，不伪装最大counts。
- CLI check stdlib零文件输出/claim/science；run新mode不导入生产密码生成器/执行预检。初始device/context/RSS仍验证，前向/更新无免费工作。
- PS真实exit捕获先dummy复现；不能改Python返回科学语义/掩盖错误。可使用明确作用域nativecode或等效可靠原生ProcessExitCode，但不恢复remote detachedStart-Process断连。

- [ ] RED：新固定recipe/rounds50/早停参数/数据sha；fake NPZ/源run身份的严格decode、成员/shape/dtype/overflow/路径/故障；坏输入不开始fit、不fallback生成。
- [ ] RED：warmup10+8最早18、累计小改善、delta边界、best严格最低与early-stopanchor分离、各臂分别18/更多、50上限；新实际epoch计数/旧默认计数兼容。
- [ ] RED：readtrain/dev无TEST，双reload2故障无TESTread，两个score完整统一保存；不读旧weights/testscores；源NPZ字节复用身份。
- [ ] RED：scriptdummy0/7预期契约，原LASTEXITCODE null不伪补；actor只静态/假子进程，现场dummy主控负责。报告显示实际轮/stop/best/source/探索性。
- [ ] GREEN：最小实现，保持fit/训练/解析/统计/CLI职责方向；不要第二runner/registry或加模型开关。
- [ ] 验证：新owner人工套、原owner93、相关core早停/保存/执行/布局回归，forcedRuff/strictmypy、源闭包纯stdlibcheck及fake CLI闭环。全safe由主控fresh一次，不science。
- [ ] 报告至本计划workspace task-1-report.md，含RED/GREEN/actualcommands/范围/新接口/allowlist和测试清单/变更diff，零Git。独立review＋必要fix复审，主控验收后才发布。

## Task 2: 主控退出码dummy、限定发布/精确同步、唯一GPU运行及保存值交付

**状态：RELEASING。** 先前现场空cmd0/7已验证global nativecapture，原local赋空shadow是旧UNKNOWN根因；没有调用科学路径。旧run记录不回填。

**Files:** 新spec/plan状态；仅审查allowlist产品代码发布，日志/数据/权重不入Git。

- [ ] 独立新workspace隔离af784fcb，冻结source/config/dataset SHA并保留main/index与旧产物。必要时launcher根因debug只读代理辅助，不授实现代理SSH。
- [ ] Windows同调用上下文无害`exit0/exit7`验证新nativecapture；不会调用Torch/cipher/训练，不修旧run的UNKNOWN。失败停止不science。
- [ ] 最终snapshot全safe+新owner/相关回归、Ruff/mypy、独立整体review；只本轮允许清单commit/push一次，远端Gitfetch/exactcheckout。
- [ ] 本轮全新`G:\lxy\shared-m-cipher-nd-runs\<new-id>`，原-02只读source root，GPU1当前空闲/UUID1card/free>=16GiB；全3h、不装依赖，不碰GPU0。
- [ ] 通过SSH前台调用已发布脚本，保留连接直至原生退出，source/current状态/实际epoch/stopreason/counts/nativeexit按原记录；任一failure/unknownstop，retry0。
- [ ] 取回实际data/models/result/logs，核SHA/身份/计量/nativeexit，不load模型/label/forward/evaluate；纯saved图表报告，TEST已见仅探索。旧产物原样，预算关闭不复用。
