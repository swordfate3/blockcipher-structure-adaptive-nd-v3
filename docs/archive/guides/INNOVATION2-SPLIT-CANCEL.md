# 创新2：SIMON64 split-and-cancel 有界符号积分入口

本分支是**选择明文积分认证**，不是原创新2随机明文单bit神经预测，不训练模型，不恢复密钥。
协议 `simon64-split-cancel-v1`，入口身份 `experiment=integral-symbolic`，fit/更新/evaluate/retry 均为0。
这是论文原理指导的自包含保守变体，不是作者代码或论文18/19轮结果复现；本文只描述已实现能力，不声明已正式执行。

## 已实现的统一 CLI

在安装入口的环境：

```bash
cipher-experiments check --config configs/innovation2-split-cancel.json
```

源码环境等价命令（不安装）：

```bash
PYTHONPATH=src /workspace/.venv/bin/python -m cipher_experiments.cli check --config configs/innovation2-split-cancel.json
```

`check` 只严格读配置，密码调用0，不导入数据层、密码、训练或新旧执行器。配置恰为
`name`、`experiment`、`protocol`，拒绝重复/未知/混合字段、非有限数和超过64KiB的文件。
`name` 仅标识，不改变科学参数；无 CLI solver/count/seed/deadline override。

**只有获得本周期正式运行授权并完成工程/独立审查门禁后**，才可在全新的输出目录调用已实现的执行入口：

```bash
cipher-experiments train --config configs/innovation2-split-cancel.json --output runs/<fresh-id>
```

源码环境也支持 `PYTHONPATH=src /workspace/.venv/bin/python -m cipher_experiments.cli train ...`，参数同上。
这里 `train` 是统一入口名，实际没有训练、checkpoint 或梯度。新协议不支持 `evaluate`。
目录先 exclusive claim；既有目录（包括失败目录）拒绝覆盖/续跑，不自动重试。
固定绝对截止为 **2026-11-05T20:00:07.107Z**；过期调用只写 UNKNOWN 失败记录，不导入生产数据或调用密码。
这些命令的存在不授予第二次正式运行、改期限或追加面板许可。

## 固定数学与执行顺序

状态 `(X高32,Y低32)`，位序 MSB0。输出观测固定：64单位bit，再所有 `i<j` 二次项，共2080维，
坐标 `i` 对应 `1<<(2079-i)`。精确 q=1 后缀在 Boolean ring 中允许**相同(state,key)身份**按GF2相消；
不同符号 keymask 不合并。前缀只传播 union/OR 支持，不作不健全的 GF2 相消。
ZERO 仅表示充分支持证据（全集3缺失），UNRESOLVED 不是反例。

固定阶段：

1. **A符号**9个请求：H/p7/X7[0] 原Q重放，H/p6/X6[j]（j=1,2,3,8,26,27,31），H/p7/X7[25]。
   原Q必须为 `{0,1,2}`；A4认证源是 X6[j]，不是 X7[j]。
2. **A面板**依次 A1=7/H/X0、A2=8/L/X0、A34=8/H/X0及7个h、A5=9/H/h0、A6=10/L/h0，
   每面板64组×四点。A3/A4仅一次 A34 采集，共五份面板1280 blocks。
   A2/A5/A6来自已认证A1和精确输入/输出恒等式，不要求保守oracle再次认证其整个扩展。
   A1/2/5/6出现非零即停；A4已ZERO出现非零即停；A3或A4未定线索的经验反例仅记录，不换候选。
3. **全部B符号**：(p7,H/L)、(p8,H/L)、(p9,H/L)，总轮数8/9/10；完整 unknown/约束右核K、
   单observable空间、结构基线W、确定商及验证证据逐cell落盘；任何失败空间为null，不把partial当完整。
4. **冻结selection**：所有6cell完成后，取商的前4个代表；商空则K前4作为实现正控；不读面板择优。
   selection及每个完整cell证据的SHA在任何B null或cipher前写盘。K=0记NOT_APPLICABLE，不补成本、不称实测0。
5. **全部适用B null**：各128组×四个独立uniform64输出，按真实公开多项式逐点评值再XOR；
   任一被检查组合128组全零则报警全停，不据null重新选择。
6. **B核对**：按固定cell序各check32、confirm224新组；二者都不参与选候选。
   任一认证组合出现非零即停，不重采、不换mask。确认与check属于同一累计60s实测cell。

每个cipher组同key四点；case/phase之间实际key全局不复用，背景清活跃后判重；仅A34明确共享观测。
随机域为ASCII JSON `[protocol,198101,role,case_id,phase,purpose]` 的SHA256大端整数；实际私有key不公开。

## 预算、失败和证据等级

- 科学cipher attempted≤7424（A1280+B6144）；null≤3072个64bit words。
  工程cipher≤512属于另一个显式工程入口，本runner不花工程额度。
- 每cell2080观测、每展开≤16项、出现/unknown≤33280；全局prefix首次cache miss≤220000。
- 总wall/CPU各1800s，A符号各120s，每B符号cell各180s，每实测cell各60s；检查在循环/采集/写盘边界。
  同cell不因显式额外验证或check→confirm而重置时钟。截止、预算、取消或异常均UNKNOWN；不是硬进程/I/O限时。
- attempt提交前reserve，坏输出/异常不退款。完整四输出才completed；完整row即使callback超时/写盘前失败也计completed，
  从公开partial按index补盘，不重新调用oracle。无法写盘时返回failed；缺exit永远UNKNOWN。
- 不打印原异常、key、轮key、实际状态、明文/点级密文或私有pair ledger；仅公开聚合。
  普通Python同进程特权内省不在保密保证中；禁止traceback locals dump。
- 实测零不是全key证明；全key认证来自健全符号证据。`d_extra=dimK-dimW`仅相对明确列出的基线，
  不等于创新性证明或所有已知关系的增量。UNKNOWN/NOT_RUN不补0。

## 保存 schema（供无加密独立复核）

JSON均严格有限数；packed组合mask固定**520位hex**，state固定**16位hex**，符号key固定**8位hex**，
均小写无`0x`，不是实际秘密。没有被截断的矩阵或经验核。

- `config.json`、`manifest.json`：固定身份/SHA、源码逐文件SHA（包括全部新模块、scalar/expanded密码与公开表示）、
  Python版本、预算/期限/阶段顺序、旧query142的原件路径和预期SHA/公开身份。只记录旧预期，不import/read旧runs。
- `suffix.json`：公共一份2080 observables及对应精确展开，项按(state,key)排序。
- `A-symbolic.json`：9请求的完整证据和A2/5/6继承链；`A-results.json`：所有A视图/面板进度。
- `cells/p{p}-{H|L}.json`：`{status,space}`；完整space含unknowns/constraints/K/W/baseline_single/quotient/selected、
  完整prefix evidence/support、dimensions、rank及全部validation标志。超时partial时space=null；若有已算诊断可存diagnostic_space，不能算完成。
- `selection.json`：6cell固定selected（最多4）及来源、cell SHA、冻结时钟与counts；任何B数据前写盘。
- `rows/{case}-{phase}.jsonl`：每行`index`（phase内从0）、`bit_parity`（四完整输出的64bit XOR）、
  `parities`（按已冻结mask顺序的所有组合parity）。A34保留8列；B各≤4列。
- 同名`.json`：完整公开identity、masks、requested_rows、nonzero_counts、status、draw计数、累计counts、
  JSONL basename/SHA256/line_count/columns，另列completed_rows以区分已完成与成功落盘。
  持续I/O故障可令completed_rows大于line_count，仍为partial/UNKNOWN，不输出私有采样值。
- `progress.jsonl`、`stages.json`、`results.json`、`exit.json`：阶段开始/完成/失败、时钟、原因、
  attempted/completed/null_words/attempt_unique/completed_unique、prefix misses、exit_code。
  results/exit中cell仅摘要/路径/SHA，避免跨cell累留大矩阵；完整空间在cell文件。

完整链路与配额是协议固定规则。函数keyword fake依赖只用于合成测试，不是用户可配置实验协议。
本文不包含未实现的独立产物verifier/工程命令，也不声称Windows或正式六cell已经实测。

## 附录：实际运行结果（2026-11-05）

本节记录该协议的实际执行结果；上文“只描述已实现能力、不声明已正式执行”的措辞是运行前的状态说明，已被本节取代。

- 唯一正式科学周期：`runs/split-cancel-20261105-1750/`，outcome `COMPLETE`，status `completed`，exit_code 0，invocation 1、retry 0，fit/updates/evaluate 0。
- 科学 cipher attempted/completed/unique 均 6400（上限 7424）；null 2560 words（上限 3072）；工程真实块 401（上限 512），科学+工程合计 6801（全周期上限 7936）。外层科学 wall 约 16.31 s，运行内部 wall 15.836296 s / CPU 18.552184 s，是运行成本而非开发成本。
- A 阶段：A1/A2/A5/A6 各 64 组全部零；A34 面板 X8[0] 视图 32/64 非零（预期反例，非失败正控）；A4 的 7 个 h8,j 全部 0 且各有新的前缀 `CERTIFIED_ZERO`；旧 Q={0,1,2} 重放一致；X7[25] 也取得新前缀 ZERO 证书。
- B 阶段 dimK/dimW：p7-H 9/9、p7-L 31/30、p8-H 2/2、p8-L 9/9、p9-H 0/0、p9-L 2/2。只有 p7-L `d_extra=1`；其余非空 cell 均在预列 W 内。p9-H 的 K=0 记 `NOT_APPLICABLE`，不调用密码、不称实测 0。
- p7-L 的唯一 extra 为 `Y8[3]·h8,2`（等价 `c2·c35 + c35·c36 + c35·c42`）。它超出预列 W，但 W 不含乘法闭包，且该关系可写成已有 h 乘 bit；要作为继承产品认证仍需对应前缀乘积认证，不能仅凭 `h=0` 或 xor 形式宣称全新。本附录不把它写成新颖性结论。
- B 实测：5 个适用 cell 共 13 个已选组合（4+1+2+4+2），check 32 组与 confirm 224 组全部零，未出现反例；5 个 cell 的 null 均未触发 128 组全零报警。
- 保存证据独立复核：`validation/innovation2-split-cancel/saved-verification/verification.json` 为 `VERIFIED_COMPLETE`（独立 stdlib，production_imports=0、cipher_calls=0、certified_cells=6）。限制保留：exact pair/unique 不能由 parity 再生；二次 parity 不能由 bit XOR 反推；事件顺序不是外部时钟 attestation；旧 query142 只核对保存身份与 SHA，未重开历史 run；未重新加密。
- 9/10 轮是同一七轮基底的标准结构延伸，不是新核心突破；有限面板全零不是全 key 证明。控制器最终 scoped 测试汇总在结果整理时尚未完成，因此本文不声称全部测试通过；全周期最终独立审查签收另行记录。
- 结果文档：`result/20261105-innovation2-split-cancel-cycle/{report.md,index.html,source.json}`。未启动浏览器；结果中的 artifact 链接是从仓库根可用的本地相对路径，只通过 HTTP 服务 `result/` 时不可达。


### 最终结果补记：UTC 复核修复与 p7-L 乘积证据（2026-11-05）

本补记取代上一结果附录中“最终测试尚未完成”的当时状态，不更改前述运行前协议或历史记录。
最终修复后控制器实测本周期 scoped 测试及旧 integral CLI/config 回归 **386 passed in 56.00s**，Ruff 通过、strict mypy **21 files** 通过；未运行全项目 pytest。最终 whole-change review 唯一 P2（中间 saved UTC deadline 漏验）已修，限定复审 `07d2dc9d-59a2-4b9` **Spec PASS / Quality PASS**。控制器再次只读复核 `controller-final-verification/verification.json` 为 **VERIFIED_COMPLETE**，未重跑科学实验。周期关闭，剩余预算不续用。此前340项及修复代理123项仍保留作历史证据。

最终 proof 为 `validation/innovation2-split-cancel/saved-verification-final/verification.json`，`VERIFIED_COMPLETE`，6 个 cell，production_imports=0、cipher_calls=0；旧 `saved-verification/verification.json` 原样保留。completed 的所有 saved stage start/end、progress、selection/freeze 与 final UTC 逐个严格 `<1793908807.107`；failed/UNKNOWN 越界仍可 partial，UTC 不要求单调。只读原树复核，无科学 CLI、重新加密、solver 或训练；新旧 proof 的原始产物 SHA 集合相同。

p7-L 唯一 selected 为 `c2*c35+c35*c36+c35*c42=Y8[3]*h8,2`。已保存 q1 展开中 `(1020000000000000,00000000)`（X7[3]X7[10]）和 `(1800000000000000,00000000)`（X7[3]X7[4]）两个 unknown 各出现两次精确相消；剩余 `(1000000020000000,00000000)` 即 X7[3]Y7[2] 与 `(1000000000000000,20000000)` 即 X7[3] 乘符号 key2。两个 state 的 p7/L 保存证据都是 support `[0,1]`、`CERTIFIED_ZERO`，由此分别认证残项零和（key2 组内固定）。不能从 XOR h=0 就推乘 bit 仍零；这里用了具体的前缀乘积/单位证据。此为 W 未列的乘积族，非 novelty。冻结 `dimK=31, dimW=30, d_extra=1` 不变，不事后改基线。

完整 RED/GREEN、命令日志与逐文件 diff 见 `.superpowers/sdd/2026-11-05-innovation2-split-cancel/final-fix-report.md`。结果 HTML 已核本地链接存在，未 browser render；只服务 result/ 时仍不能声称 runs/validation 链接可访问。
