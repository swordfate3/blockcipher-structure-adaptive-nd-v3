# 创新2积分支线 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 独立构建公开图→COPY指数路径CNF→独立验收的研究工具；缺现有proof checker时停于BLOCKED_CHECKER，不暗跑未认证正式矩阵。

**Architecture:** 此支线不进入生产训练backend，全部工具留`validation/innovation2-multicycle/integral/`。公开DAG、端口展开／CNF、独立路径验收、单次solver调用、外部checker、产物身份各有明确职责。CNF不存在数据bit或真实key，SAT只表示抽象路径，不表示合法密钥反例。

**Tech Stack:** Python3.12标准库；现有PySAT如可用；现有独立proof checker如可用。不得下载安装，不修改环境或PATH。

**Spec:** `docs/superpowers/specs/2026-11-02-innovation2-multicycle-design.md` §6、8—11。神经侧见同目录`2026-11-02-innovation2-neural-multicycle.md`，不得借其GPU／fit额度或混淆成果。

## Global Constraints

- 当前只写本地代码与无solver人工测试；真实加密／标签／GPU／SSH／安装／发布=0；solver/checker实际调用需要另行明确批准。
- 留main、保留dirty和旧toy／生产simon_cone；不扩旧toy n≤4/q≤2和生产cone1..6上限；不import历史科学run／backups／cipher。
- 新文件均位于`validation/innovation2-multicycle/integral/`；此目录被现有ignore覆盖，不修改ignore／强制stage。交付不能把本地工具冒称已Git发布。
- 纯人工oracle测试CPU和wall各≤30分钟；共享神经计划4小时开发验证wall总额，人工等待不计。不优化模型，无训练更新。
- 固定矩阵16项：r=8,9,10,11，bit0，k=16,32,48,63；pi[2j]=5j mod32，pi[2j+1]=32+(5j mod32)，S_k取前k项。
- 正式授权后最大toy solver64×2s、controls8×15s、formal16×120s；checker4正控＋16formal各90s；全周期90分钟，retry0。没有checker则不启动I1，不自行换成未认证探索。
- 公式≤250000vars/1000000clauses；proof≤512MiB；总产物≤4GiB；建议进程树2GiB，仅协作限制时如实记录。
- 同轮连续两项资源UNKNOWN关闭后续新科学启动；身份／编码／验收错误立即停。缺exit／无法计量是UNKNOWN，不释放预算再跑。
- `SAT_PATH_VERIFIED`、`UNSAT_UNCERTIFIED`、`UNSAT_PROOF_CHECKED`、`ZERO_SUFFICIENT_CHECKED`、`UNKNOWN_RESOURCE`、`ERROR`、`BLOCKED_CHECKER`分开，不用balanced bool。

## Review Focus

1. fanout漏了重复入边／最终输出／死线：显式端口计数与COPY/DROP测试（Task1/2）。
2. 最终指数被错误强制等于target：k_out≤target，target1允许0和1（Task2）。
3. encoder与verifier同门表自证：独立文件／独立实现者，不共享关系生成器（Task3）。
4. proof对错CNF、错误checker仍返回0、SAT模型缺变量：原件SHA/完整赋值/exit与checker输出同时核（Task3/4）。
5. 资源截断或朴素次数已证被包装成新积分：状态拒签零证书，候选预筛标签分列（Task5）。

## 文件责任与所有者

- `public_graph.py`：implement-deep A；共享DAG、节点稳定ID、完整公开轮接线／目标剪枝；不CNF、不solver。
- `path_cnf.py`：A；每次使用端口线性化，COPY／DROP／指数CNF，DIMACS映射；不真实求值／key。
- `verify_path.py`：implement-deep B；只消费冻结的图／Program／mapping数据结构，独立解释门的整数约束，不import encoder表或生成函数。
- `solver_call.py`：implement；显式一次既有PySAT solve、time/resource/exit原件；不选择候选／重试。
- `proof_check.py`：implement；已有checker适配，严格绑定原CNF/proof；不是自己实现一个“可信checker”。
- `records.py`：implement；规格／身份／状态、strict JSON（含重复键／NaN拒绝）、固定矩阵；不执行求解。
- `test_public_graph.py`, `test_path_cnf.py`：A；`test_verify_path.py`：B；`test_process_records.py`,`test_fixed_matrix.py`：implement。
- `READY.md`,`SOUNDNESS.md`,`engineering-ledger.json`,`verification.json`：各所有者只编辑分配的小节；文档整合由主模型最后统一，避免并行覆盖。

## Task1：公开图和冻结端口接口

**Owner:** implement-deep A。不启动solver。**Produces:** 下述stdlib dataclass与公开图，仅1..11范围研究工具。

**Interfaces**（同目录导入）：

- `Node(op: str, inputs: tuple[int,...], coordinate: tuple[int,int] | None=None)`；op允许INPUT/PARAM/XOR/AND/ID/ZERO/ONE，图中ID连续且拓扑序。
- `PublicGraph(nodes: tuple[Node,...], plaintext_ids: tuple[int,...], parameter_ids: tuple[int,...], outputs: tuple[int,...])`，plaintext64个MSB顺序；parameter为符号round/position坐标，不是密钥值。
- `build_simon_graph(*, rounds: int, target_bit: int=0) -> PublicGraph`；1..11／0..63严格built-in int。先公开递推再目标祖先裁剪，未用plaintext仍保留并DROP，不丢active假设。
- `PortStep(op: str, inputs: tuple[int,...], outputs: tuple[int,...])` 与`LinearProgram(steps: tuple[PortStep,...], input_lines: tuple[int,...], parameter_lines: tuple[int,...], output_lines: tuple[int,...], line_count: int, graph_sha256: str, line_sources: tuple[int,...], node_steps: tuple[int,...], input_coordinates: tuple[int,...])`由Task2定义，此处先冻结名字／语义。线号0-based；line_sources逐线标原图node ID（COPY保持来源）；node_steps逐node标执行该原门的step index，INPUT/PARAM为-1。graph_sha256绑定按字段完整canonical JSON的原图。input_coordinates与input_lines平行，记录32*half+index的原INPUT公开坐标；generic小图允许不连续坐标，编码器不得以node ID或顺序猜测坐标。provenance只是待验数据，B必须独立核，包括逐输入坐标回查，不因SHA匹配信任翻译。

- [ ] **Step1：RED** 写`test_public_graph.py`，独立MSB index递推验证1..6×64=384完整节点／operand有序结构，与当前生产simon_cone公开对象逐项canonical映射；不是只比support集合。不调用生产cipher。
- [ ] **Step2：执行RED** `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -o addopts= -q -p no:cacheprovider --confcutdir=validation/innovation2-multicycle/integral validation/innovation2-multicycle/integral/test_public_graph.py`，新模块不存在应FAIL。
- [ ] **Step3：实现公开图** 不做Boolean化简；固定未知round参数作为PARAM；7..11用测试独立全宽轮递推核每一目标接线（支持r1低半纯复制）。浮点或bool轮次、前向引用、非法arity／duplicate坐标拒绝。
- [ ] **Step4：GREEN** 同命令PASS；独立人工Boolean求值只用少量固定自造64bit输入与任意常量作公开递推接线oracle，不称真实加密或legal key；明确这是公开图值域人工检验、生产encrypt0。
- [ ] **Step5：交付review** 两新文件Ruff/mypy限定通过；给B冻结dataclass规格和节点编号规范，不共享门关系代码。说明与历史公开图接口差异，不修改生产cone。

## Task2：COPY线性化与指数CNF

**Owner:** A。**Consumes:** Task1冻结graph。**Produces:** 无solver的确定性CNF。

**Interfaces:**

- `linearize(graph: PublicGraph) -> LinearProgram`，每个原引用按使用端口展开二叉COPY；原wire最多消费一次，输出和重复入边都算使用，死wire显式DROP。
- `PathFormula(clauses: tuple[tuple[int,...],...], variable_count: int, line_variables: tuple[int,...], program: LinearProgram, active: tuple[int,...], target_mask: int)`；DIMACS变量1-based。冻结为无辅助变量：variable_count==program.line_count，line_variables恰为1..variable_count的排列，每条线唯一对应一个变量；不得静默引入Tseitin辅助变量。
- `encode_path(program: LinearProgram, *, active: tuple[int,...], target_mask: int=1) -> PathFormula`，active是plaintext坐标不是node ID；PARAM初指数0；背景初指数0；最终k≤target。
- `write_dimacs(formula: PathFormula, path: Path) -> str`返回fileSHA，不启动solver。

- [ ] **Step1：RED** `test_path_cnf.py`独立穷举每个局部门的CNF满足赋值与整数关系相同：COPY a=b+c、XOR c=a+b、AND c=a OR b、ID相等、DROP输入0，ZERO/ONE输出0。
- [ ] **Step2：执行RED** 指定新test（无solver），应缺符号FAIL；门真值测试直接枚举assignments，不调用PySAT。
- [ ] **Step3：实现** 用明确小CNF生成每种关系，no registry；检查空clauses/变量越界／map重复；公式上限构造时检查，超限返回资源异常供records捕获，不返回半公式当UNSAT。
- [ ] **Step4：GREEN人工整图** shared g²/单活跃与双活跃、x xor x假阳性、未使用active必DROP阻断、多输出／重复输出、参数共享、常量、死节点。n≤4/q≤2的固定toy与独立antichain向上闭包oracle比较，不import旧bitdp门表。最终指数0在target1下允许；target0不允许1。
- [ ] **Step5：健全性文档** `SOUNDNESS.md`给出切面非零矩向上闭包归纳与CNF“不约简单路径存在”对antichain查询等价；未知常量覆盖legal key的量词说明；对SAT不能推实际反例。review未通过不写正式READY。

## Task3：独立路径验证器

**Owner:** implement-deep B（新上下文独立人员），可读spec／dataclass和CNF映射格式，禁止import path_cnf生成表。**Files:** 仅verify_path.py/test_verify_path.py。

**Interfaces:**

- `verify_translation(graph: PublicGraph, program: LinearProgram) -> dict[str,int|str]`：独立核原图canonical SHA、plaintext/parameter有序坐标、每个原门的op及有序operands、所有重复使用端口、最终输出、COPY树及DROP。遍历Program自行推导每线来源并与line_sources比较；逐原门由node_steps定位并核输入来源，非叶节点恰执行一次，除COPY/DROP外不允许多余门；输出端口按原图outputs有序核对。每线恰产生一次且恰消费一次（最终output视为一次消费），COPY只分配同一原来源的用途；死线仅能DROP。不得调用linearize重建后比较，也不得仅核support集合。
- `verify_assignment(graph: PublicGraph, program: LinearProgram, *, active: tuple[int,...], target_mask: int, variable_count: int, line_variables: tuple[int,...], model: tuple[int,...]) -> dict[str,int|str]`；先verify_translation，再核路径，返回status=SAT_PATH_VERIFIED和两项验收数量；失败raise ValueError。强制variable_count==program.line_count，线映射为1..variable_count排列；模型必须完整覆盖此范围、每变量恰一正／负literal，无0／冲突／超界／未映射变量。
- Program内部路径合法与原图翻译正确分开记录；UNSAT证书升级也必须先通过verify_translation，不因没有SAT模型跳过原图翻译检查。检查初始、每门、wire只消费一次、完整输出、最终≤关系；不得只验证CNF而漏图含义。

- [ ] **Step1：RED** `test_verify_path.py`独立手写合法／非法赋值，含缺最后一个变量、额外未映射变量、辅助变量、冲突、同wire重复消费、坏copy、遗漏active、输出目标不满足、映射换CNF、非整数literal。另以固定原图构造内部合法但翻译错误的Program：交换plaintext坐标、换原门operand、丢重复用途、错COPY来源、错DROP、换目标输出；即使重算Program和provenance的自称SHA也必须被原图独立验收拒绝。
- [ ] **Step2：执行RED** 指定test（不solver）缺接口应FAIL。
- [ ] **Step3：实现独立关系验收** 用算术等式／独立逻辑判断，不调用encoder生成器；模型符号转bool后逐门核，不信输入status。
- [ ] **Step4：GREEN及交叉核对** A生成的有限人工公式用test内枚举SAT赋值，全部交B检查；单点损坏必须拒绝。x xor x可抽象SAT但不叫真实反例。
- [ ] **Step5：review与交付** Ruff/mypy／人工pytest限定；独立检查器不等于UNSAT proof checker，READY必须分清两者。

## Task4：单次进程、严格记录和现有checker适配

**Owner:** implement；不得改A/B文件。**Consumes:** CNF原件／SAT verifier／显式冻结规格。**Produces:** solver_call.py/proof_check.py/records.py，只fake进程工程，不实际solve。

**Interfaces:**

- `inspect_tools() -> dict[str,object]`：只读现有`.venv`依赖metadata、PATH及明确已有可执行路径／版本身份；任何--version只在确认无求解副作用后执行。不得全盘find或下载安装。未找到checker返回BLOCKED_CHECKER。版本/PATH敏感信息不打印凭据。
- `run_solver_once(cnf_path: Path, output: Path, *, seconds: int, solver_name: str="g4") -> SolverRecord`：只接受新output，单子进程，一次load/solve/delete；协作/硬限制能力记录；无自动重试。
- `check_proof_once(cnf_path: Path, proof_path: Path, *, checker_path: Path, output: Path, seconds: int=90) -> CheckRecord`：支持已存在drat-trim，若仅gratgen还需已有完整证书检查链，否则BLOCKED_CHECKER，不能将gratgen生成本身当独立认证。
- `load_record(path: Path) -> dict[str,object]`严格JSON拒重复keys/NaN/Infinity/溢出、schema/identity不匹配。

- [ ] **Step1：RED** `test_process_records.py` fake子进程完成／无exit／超时／非零／截断proof／wrong-CNF SHA／耗时与字节残留；未知占额不退款；checker exit0但无预期verified输出不通过。
- [ ] **Step2：执行RED** fake subprocess只写人工小文件，无真实solver；缺模块应FAIL。
- [ ] **Step3：实现最小一次调用与身份** stdout/stderr原件、CNF/proof/工具文件SHA、版本、exit与monotonic耗时；严格指定秒数与size上限；进程树限制无法可靠实现就标合作观测，不能假报硬限。禁止科学source SCP／网络连接。
- [ ] **Step4：GREEN** fake证据链状态分级；SAT必须独立verify才能升级；UNSAT保存proof不能直接叫ZERO；校验器不可用立即BLOCKED，正式调用计数不启动。缺proof／unchecked proof保持UNSAT_UNCERTIFIED，不用于正式阳性。
- [ ] **Step5：只读工具预检** 明确报告现有PySAT及solver可用性／checker路径或未找到，不能继承历史结果；不实际执行toy solver，无权限时Task5不调用。若drat工具版本旗标不明确，只核文件SHA/元数据，不冒称已运行版本验证。

## Task5：固定矩阵、预筛与状态收口

**Owner:** implement，records.py独占已由Task4交接。**Consumes:** 冻结图／完整记录。**Produces:** 16规格及8controls的JSON／READY说明，不新建通用runner。

- [ ] **Step1：RED** `test_fixed_matrix.py`精确检验r/k升序、pi无重复覆盖64位、S16⊂S32⊂S48⊂S63；八controls=两人工＋r6{0,4,10}＋r7{0,2}＋r8{0,1}/{0,2}/{0,31}/{15,16}。
- [ ] **Step2：实现固定规格** `records.py`只生成serializable specs，不触发solver；文件写新目录，不覆盖旧产物。preflight support检查所有active在cone，degree基线O(nodes)单独输出；旧投影支持只在16query／100k pairs每query／1m总数内实现独立有限小函数，不import历史工具。
- [ ] **Step3：GREEN预筛人工测试** absent dependency／degree已证／same-round已证子集继承分别标SKIPPED_TRIVIAL，不包装新发现；预算耗尽UNKNOWN不能声称项不存在；证据链各阶段单独状态，只有proof检查和图／编码审查均通过才候选ZERO_SUFFICIENT_CHECKED。
- [ ] **Step4：无solver演练停止规则** fake记录单资源UNKNOWN允许剩余固定同轮一次，同轮连续两项UNKNOWN停；ERROR立停；checker缺失I1禁止；88solver/20checker预算无类别挪借，失败残留计入4GiB。
- [ ] **Step5：主验收** 独立code-review核数学／共享端口／final不等式／checker身份；主模型限定Ruff/mypy/pytest，记录实际范围及0solver。无现有checker交付BLOCKED_CHECKER和完成的I0人工工程，不擅自安装，不启动16正式矩阵。

## 正式运行的独立授权门槛

只有用户明确批准toy64／controls8／formal16／checker20及90分钟周期，且现有工具门槛齐备，才可由experiment执行现有明确单次函数调用；不授予代理新研究／候选／安装权。该阶段顺序toy→controls→formal预筛后求解，全部原件和negative/UNKNOWN保留。

没有真实加密枚举、没有高维2^k follow-up，没有精确key miter、没有论文十一轮复制或神经预测成功宣称。没有checker则请求安装或未认证探索的新合同，二者均不在本计划默认批准中。

## Self-review／计划审阅门槛

Spec§6全部映射Task1—5；资源／状态映射Task4/5；依赖和独立验收已分开；门关系与最终查询固定；现有工具缺失不被“实现适配”掩盖。科学授权尚缺，未启动solver。

本计划与神经计划为两个可独立验收子项目，主线优先；审阅后采用用户已经指定的代理驱动。先完成无solverI0，I1保留BLOCKED状态，不与神经fit混报。
