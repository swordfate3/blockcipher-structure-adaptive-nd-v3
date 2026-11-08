# 八轮精确反例判定结果：两个固定cube均有独立轮常量反例

## 实质结论

按用户“是推进”，在公平轮转搜索仍饱和后，按[精确miter合同](INNOVATION2-INTEGRAL-MITER-PLAN.md)完成两项固定公式求解。**两项均SAT_VERIFIED，不是UNKNOWN／超时，也不是UNSAT。**

| 八轮活跃高字明文位，目标X8[0] | cube份数 | CNF变量／子句 | 反例输出ones | cube XOR | CPU／wall秒 |
|---|---:|---:|---:|---:|---:|
| {0,2} | 4 | 2310／7934 | 3 | **1** | 0.033157／0.031967 |
| {0,1,2,3,4,5,6} | 128 | 67901／253950 | 69 | **1** | 0.781006／0.799352 |

第一项输出依MSB组合00,01,10,11顺序为 **1,0,1,1**，异或1。第二项全部128份输出／背景／132个出现的符号轮常量及完整SAT模型都保留在witness.json。

因此，这两项不只是当前过近似“可能出现全集项”：**在独立轮常量、同cube共享背景的公开Boolean模型中，确实不恒零。**七轮{0,2}充分关系不能在同样“任意轮常量”假设下原样延伸为八轮恒零；不反驳已验证的七轮关系。

### 不能扩大的结论

- 这里c[t,i]是符号独立轮常量，副本间共享，域包括全0；没有编码SIMON合法128位主密钥调度。SAT的轮常量未证明能由某把合法key产生。
- **不是“找到实际SIMON key反例”**，也不是选了一个实验key再失败；真实encrypt调用0，真实key读取0，witness不用于挑key／seed或模型教师。
- 只反驳这两个固定cube在该较强全称假设下恒零，不能排除其它八轮cube、其它输出bit或合法key约束下的更强结果。
- 无UNSAT结果，也没有独立proof checker；不宣称已完成零和证明认证链或标准division property。

## 方法和身份

为每个活跃赋值建立完整目标公开电路，全部副本共享outside明文与符号c；AND用3条Tseitin子句、XOR用4条、active固定为±truevar；末端输出parity断言1。完整写CNF／mapping后一次append_formula，现有Glucose4/conf_budget200000单次solve_limited。

模型拒绝缺变量／重复／冲突／0／非整数／越界，不补值，不coerce。先逐子句核全部CNF，再独立32-bit Boolean列表递推核全部copy target与XOR1。主模型另以packed32旋转公式，只读现有公开符号witness再次逐copy核验；未调用生产encrypt／key schedule、未重算真实标签。

- 当前源码 `validation/integral-miter/miter.py` SHA256：`056e7d542edc3ab082dd8fb55d8825a2f5803be5222c7522d5e82825e5459238`。
- 最终test SHA：`863ba4b1bf961ebfbb30715a25455be76cd9c73054b14157f94354438b44903a`；main独立**78项fake／人工小图／公开1..6全节点等价测试**、Ruff、两文件strict mypy、diffcheck通过。
- `freeze-v2-main.json`固定最终版本；旧v1为正式前修复历史，未覆盖。
- Python3.12.15／PySAT1.8.dev24，现有环境，无安装。公式源当前src公开cone一至六轮全图核对，七八轮静态递推；小轮工程等价检查不是八轮独立证明。
- 两项solver init/load/solve/delete均各1、delete失败0；retries0，均exit0。内部proof临时文件分别38028／279245字节，未超64MiB；SAT不导出UNSAT proof。

## 实际核验与原件

| 目录 | 文件／总字节 | results SHA256 |
|---|---:|---|
| `runs/integral-miter-20261029-190816-r8-k2/` | 9／320296 | 778c03c910ec0b1636e380261a0a150e52bffd26eec52e6231eceaa7ffdf5539 |
| `runs/integral-miter-20261029-190825-r8-k7/` | 9／7850242 | d7ae00fab4dc67904a8bf25734b9664c2f4e93e0e914918aa1f15ce68e1c8170 |

main实际重算所有artifact字节SHA，独立解析已存DIMACS确认全部变量赋值与7934／253950子句满足，packed32公开符号重算4／128份输出，核对计数／source绑定／完整加载／文件cap／原始answer和退出。核验不是再次调用solver、完整formula重建、真实cipher交叉验证或全空间枚举。

独立code-review另只读freeze／exit／results／ledger聚合字段，确认两SAT状态、次数与结论边界相符，未见新增不一致；其未独立运行byteSHA、全CNF或witness计算，不把主模型动态验收归为review实测。

`validation/integral-miter/verification-closeout.json`和ledger均CLOSED；正式solver2次用尽，无第三候选、无真实验证、无更高轮。

## 工程偏差与修复披露

代理初版误把“常量不是固定0”实现成“至少一个常量为真”，并把缺模型值补false，均在正式求解前撤销；后来proof超限／收尾产物错误状态经两轮静态review及fake fixture修复，78项是最终身份，旧59／71项不冒充最终验收。

代理还超出原先“工程不做r8formula”的范围，做了**2次旧版encoder-only内存观察**，无solver、无runs、无真实cipher。已在READY／freeze／ledger单列pre-execution attempts，不能隐瞒为0，也不是最终修正版公式计数或正式重试。正式CNF已删除错误OR，最终子句数相对旧观察各少1。

生产src/tests/config/CLI、旧科学原件和其它dirty均未修改；这是一项临时只读数学分析能力，不是新科研runner／训练分支。没有GPU／SSH／安装／发布或神经训练。

## 下一项有意义的研究决策

分散轮转修复覆盖却无证书，而精确miter给出了两个确切符号反例：下一步不应继续追这两项的“对任意轮常量恒零”证书。
需要比较**显式COPY约束的标准bit-DP换cube**与**加入合法主密钥调度的精确反例判定**：前者寻找不同关系，后者区分当前反例是不是独立常量模型过强带来的。两者均未在本轮执行，涉及新方法／公式语义先论证，不从本次两个SAT擅自扩展到全部cube／所有key或无限预算。
