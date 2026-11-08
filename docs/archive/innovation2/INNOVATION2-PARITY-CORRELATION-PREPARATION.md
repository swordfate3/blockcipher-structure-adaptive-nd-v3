# 固定64奇偶相关性核查：工程准备完成，正式统计未执行

按[计划](INNOVATION2-PARITY-CORRELATION-PLAN.md)，本轮只落实统计与合成验证，等待正式65536标签核查预算确认。
没有任何新的真实相关性、DEV/TEST分数或模型训练结果；不能把未执行填成50%或“无信号”。

## 已落实

- `src/cipher_experiments/parity_statistics.py`：纯标准库、给定numeric明文/binary标签/固定mask的统计，无密码/模型/标签生成依赖。
- 输入限2..65536正偶数、明文全局唯一unsigned64、标签builtin int0/1，1..64个非零unique两word32bit mask。
- 每半保存整数signed/feature/label sums及原始正确数；rho是sign乘积均值，不是Pearson相关。
- internal mask bit i→numeric bit31−i；A决定符号（0固定+），B锁符号正确数，不用B挑方向。
- family_tests至少2M；本批拟联合real/null使用**256**、alpha.05，半样本32768、eps=.02374030550829211。
- 每mask检测条件为两半误差带同号且均排除0；仅同号不能算发现。原始均值是描述量，不额外声称受rho带覆盖。
- 理想独立均匀抽样的Hoeffding union bound只是参考；统计函数不能审计实际伪随机唯一抽样的概率保证。

## 字典与无标签预检

直接读取上次公开编译保存的64mask，没有重新扩大搜索或接触cipher标签。
canonical编码是`json.dumps(mask_list,separators=(',',':'))`的UTF8字节（无末尾换行），SHA256：

```text
94e3991a0c2bb979911f935dd210c64431d6fb119633e8d91bc166dc7b6b3829
```

`validation/parity-correlation/dictionary-freeze.json`记录完整mask顺序、旧编译字节SHA与未来预算。
无标签data193901预检65536明文全局唯一，A/B各32768且不相交；与预列11面板零交集，不声称遍历全部历史。
未生成正式真实或null标签，也未生成新DEV/TEST；纯合成统计fixture不计真实密码查询。

## 实际验证

- **929tests、mypy84文件、Ruff、git diff --check通过**。
- 独立字符串MSB oracle、合成正/负/零相关、方向反转/A锁B、联合256估计参数、裁剪与类型/资源拒绝测试。
- 65536合成明文/64mask规模测试显式family_tests256、alpha.05、half32768，检查eps公式；不是正式标签核查。
- 干净源码CPU null `check/train/evaluate`4更新/16呈现，saved TEST与显式evaluate一致；仅旧统一CLI回归，不是统计任务的密码拟合。
- 本轮首轮mypy发现坏输入测试的ignore放在调用行而非参数行，已定点修复；未改统计公式或跳过类型检查。
- 独立静态审查`505e10bb-4887-4af`未见确认数学/实现缺陷，建议锁定256 family契约已采纳。
  审查不等独立跑测试/验证SHA或实际相关核查。
- 本轮未改config/model/training/evaluation，不加入新CLI入口/registry；上轮无标签编译器及未提交工作保留。
- 无真实密码标签构造、科学模型更新、SSH/GPU、安装、commit/push，保留papers/等无关工作。

## 待确认一次CPU正式预算

65,536次SIMON64/128六轮bit0真实加密构造、key_seed190502，仅数据层使用；同明文一份随机标签null193902。
TRAIN-only两半，无DEV/TEST或科学checkpoint访问；零模型更新/GPU，120秒协作参数非硬截止。
输出64个固定mask的完整两半聚合统计和SHA/退出，不输出实际key或逐样本秘密标签。
固定一次，不因弱结果增加数据/beam/seed或再选目标位。技术失败保留进度与未知状态，不自动再生标签。
若有稳定证据，也只支持另定新DEV/TEST的小型head提案，不自动拟合或降历史55%/3pp门槛。

证据：`validation/parity-correlation/{dictionary-freeze,panel-preflight,engineering}.json`及检查日志。
本轮当前源码/测试/文档未提交，main仍是上批结果提交；该事实不影响本地已实际执行工程检查的证据范围。
