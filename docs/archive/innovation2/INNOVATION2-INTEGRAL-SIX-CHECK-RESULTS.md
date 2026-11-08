# 六轮积分最小验证：真实16/16零和，随机对照8/16

## 结论

按[固定合同](INNOVATION2-INTEGRAL-SIX-CHECK-PLAN.md)，在SIMON64/128六轮、输出高字MSB bit0上，固定活跃明文位`(0,2,4,10,14,15,18)`，16个互不相交的128点cube：**真实侧16/16异或和为0；随机bit对照8/16为0**。与前一阶段[公开结构充分证书](INNOVATION2-INTEGRAL-METHOD-REVIEW.md)一致。

这是真实加密对积分关系的有限交叉检查，不是神经模型100%准确率、随机明文零查询预测、全44轮攻击、密钥恢复或新颖性证明。单次128查询即可形成一个积分统计；本轮用了16个统计共2048次查询。

## 实际执行与消耗

- run：`runs/integral-six-check-20261029-161843/`。
- 正式命令：

  ```bash
  PYTHONPATH=/workspace/src /workspace/.venv/bin/python \
      /workspace/validation/integral-six-check/run_once.py \
      --output /workspace/runs/integral-six-check-20261029-161843
  ```

  该命令是已完成的本次调用记录，**不是要求再次运行**。不新增生产CLI，旧`check/train/evaluate`不提供积分分析选项。
- 2026-10-29 16:18:43 UTC完成，exit0，`completed_real_all_zero`。
- CPU0.022122秒，墙钟0.026454秒（脚本计时范围；不含代码开发／审查／外层进程启动），小于60秒双时钟协作预算，不是硬进程时限。
- 真实调用attempted=completed=2048，2048不同明文，16组背景×128点；null2048bit；正式重试0。
- fit0、优化更新0、DEV／TEST选择0、GPU0、SSH0、安装0、发布0；无旧checkpoint读取或模型前向。
- 背景seed194501、null seed194502；key seed190502沿用标签侧派生约定，没有保存真实key／轮key／实际状态／完整ciphertext。
- “未知key”仅指没有向模型／公开证明器提供key值；公开确定性seed可被知情者重建，不声称对所有读者保密。

## 每组完整结果

每组128点。ones为该组目标输出bit中1的个数，偶数意味着XOR为0；不是必须64个1的严格均衡计数。

| 组（0起） | real ones | real XOR | null ones | null XOR |
|---:|---:|---:|---:|---:|
| 0 | 64 | 0 | 63 | 1 |
| 1 | 60 | 0 | 60 | 0 |
| 2 | 64 | 0 | 77 | 1 |
| 3 | 64 | 0 | 66 | 0 |
| 4 | 52 | 0 | 75 | 1 |
| 5 | 64 | 0 | 72 | 0 |
| 6 | 68 | 0 | 60 | 0 |
| 7 | 64 | 0 | 81 | 1 |
| 8 | 64 | 0 | 71 | 1 |
| 9 | 64 | 0 | 60 | 0 |
| 10 | 64 | 0 | 57 | 1 |
| 11 | 58 | 0 | 63 | 1 |
| 12 | 64 | 0 | 66 | 0 |
| 13 | 68 | 0 | 74 | 0 |
| 14 | 64 | 0 | 59 | 1 |
| 15 | 64 | 0 | 68 | 0 |
| 合计 | 1010/2048 | 16/16为0 | 1072/2048 | 8/16为0 |

候选、16组背景生成规则与全零事件在正式运行前固定，没有按输出筛选cube／key／null seed。
在理想随机函数的单bit模型下，不相交cube的奇偶独立公平，16组全零事件概率为`2^-16=0.0000152587890625`，即约0.001526%。这是预定事件的零假设概率，不是“结果为假的概率”，也不是将2048点当2048个独立积分统计。
固定Python seed对照只是一次可复现伪随机实例，概率陈述针对理想随机函数分布，不是说已经固定的文件还有随机性。
随机64位排列下，该事件概率与上述值的差最多为`q(q−1)/(2×2^64)=1.1363132657038477e-13`（q=2048）；未实际另跑随机排列对照。

## 代码、审查与工程验收

生产`src/`、`tests/`、`configs/`没有变更。一次性验证脚本位于ignored的`validation/integral-six-check/`，不把它写入训练层，不复制SIMON算法。
使用项目Python3.12.15与`PYTHONPATH=/workspace/src`，实际oracle加载路径：`/workspace/src/cipher_experiments/ciphers/simon.py`，与源码副本SHA相同。
Git HEAD为`e765eff105a9b1c610014ba7b4ce7fa6838b3ea9`，工作区有已知无关dirty；脚本未发布，不能把HEAD称为包含本实验代码的发布版本。

- 脚本SHA256：`1d3dab4751949108999a30374b350c44c0bbe53ffa9265974b31506416474fbe`。
- 密码源码SHA256：`994418a2f4d6736500d20bd79216548c333e953f5c60c1d611f42e9ae2052de5`。
- 冻结合同SHA256：`a08e3ef379cc09eea88ad09efe6a40337e3c52deef377b885b7055513ff1e286`。
- 正式前`frozen-before-execution.json`冻结以上身份，正式产物一致。

独立静态审查发现CPU-only预算与异常attempt漏计两项P2，均在正式前修复并复核：每次调用前CPU＋monotonic wall检查；attempt在调用前、completed在有效返回后累加。
DS仅实现／合成测试，主模型独立执行当前版**31项fake oracle测试通过**、Ruff通过、严格mypy两审计文件通过、`git diff --check`通过，未额外调用真实密码。
没有运行完整项目pytest／干净源码CLI训练闭环：生产代码不变且本轮固定2048次真实调用，不为工程数字额外运行包含真实加密的测试／训练；不宣称本次完成完整后端回归。
早前方法核查的另一ignored公开审计脚本存在7项Ruff风格问题，与本次正式脚本的Ruff通过分开，不混为全目录通过。

## 保存记录验收（不重算密码）

6文件共28271字节：`run-start.marker`、`config.json`、`start.json`、`progress.jsonl`、`results.json`、`exit.json`。
`results.json` SHA256：`272235f7f5185b3386fc79a54385fb4b34441431c95ccdb3ceb4e001901209fa`。

主模型独立验收：严格JSON、源码／脚本／合同SHA、oracle来源、各组128个保存bit的ones及奇偶、progress与results一致、calls计数、16组背景不同与2048明文无重叠、每组有序points SHA、退出与双时钟范围。
验收根据背景重建公开明文，不再调用encrypt、不再生标签、不跑evaluate。
首次只读验收把cube枚举位序写反，ordered-points SHA断言拦截；按脚本已声明的MSB-first assignment顺序纠正验收后全部通过。正式产物不变，无科学重跑。
验收关闭记录：`validation/integral-six-check/verification-closeout.json`；静态／保存记录审查不等于另一个实现重新加密复现。
独立code-review代理另核对协议、逐组保存计数字段／parity、记录身份与冻结快照、概率和结论边界，未发现具体矛盾；其没有执行程序，未独立重算raw-bit或字节SHA／总字节数。上文逐字符bit与字节验收为主模型实际执行，不用代理自述替代。

## 下一步含义与关闭

**已支持：** 积分结构分析在这个六轮固定cube上有明确、可直接实测的信号，不必继续用接近50%的逐点神经预测作为唯一进展指标。

**尚未支持：** 最小cube、七轮及更高轮、跨key实验泛化、新积分纪录、比文献更低成本、神经网络贡献或密钥恢复。符号充分结论的全key覆盖来自数学论证与电路正确性前提，不由这一把key的2048次实验证明。

若后续推进研究，应先选择“更多轮／更少查询的积分证书搜索”或“有明确额外用途的机器学习辅助”，与标准division-property等解析基线比较。不先让网络学一个恒零标签再宣称成功，也不立即做key恢复。

本轮已完成并CLOSED，2048次真实预算耗尽；无后台任务，不自动搜索cube、追加key／轮数、训练或发布。用户需另行选择并授权下一个有界目标。
