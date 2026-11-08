# 三轮两位XOR：非零固定背景复验

用户在零背景高半固定、低半随机取得xor2 TEST99.6384%后授权“推进”。本轮只核一个预先指定非零背景，不扩轮数/表达式：高32位固定0xA5A5A5A5（2779096485），低32位随机；三轮C0 XOR C8及null，各一次。不是把零背景模型直接迁移，而是同一训练协议在新背景重新初始化拟合。单固定key/init，新输入受限空间，不是跨key或完整随机明文确认。

## 职责和限定发布

仅修改config声明plaintext_fixed_value、data输入拼装与training参数透传/元数据；默认0从to_dict省略、旧SHA和旧输入序列不变；full要求0，严格uint32拒绝bool。保留历史high32-zero/low32-zero枚举，值为历史空间标识，实际常数由plaintext_fixed_value与元数据决定，不将非零说成零背景。不同背景不改变uint32随机活跃序列，先去重再拼固定半，split从完整pool后取。模型/labels/cipher/checkpoint/evaluate/选模规则不变。

只发布本轮3源码+1测试+4配置+本计划共9文件，排除其他dirty、runs、validation。独立静态审查无可证实缺陷；主验收源码diff及限定测试、干净index副本检查，旧配置SHA有限固定fixture避免移动HEAD/Git运行时依赖。

## 固定预算

顺序xor2→null，max2fits/retry0；三轮MLP full/direct width128/24961参数；TRAIN65536 DEV8192 TEST65536、pool65536 batch256 epochs32 lr.001 clip1，每epoch DEV BCE选最小，strict load后TEST；每臂8192更新/2097152呈现/32DEV，共16384更新/4194304呈现/64DEV。

data195601/key190502/init17/null195602沿用；相同活跃随机数序列不等于新独立样本面板，背景改变了函数和输入。null仍独立随机stream且不计算表达式；其声明bit不是额外真实标签。真实臂139264次cipher调用、null139264随机bit，由代码和完成split推算，无逐调用计数器；两臂共享139264不同明文，TRAIN/DEV/TEST无重叠。

设备指定物理GPU1 UUID GPU-f80e8056-5b1a-a0c7-9b34-fb86b20206d5映射cuda:0，逐项live free>=24000MiB/util<=30%，no-wait/no fallback。沿用既有Python/Torch不安装、不升级；Windows fresh G:\lxy工作区且无reparse祖先，exact Git checkout后核46源码及两配置SHA，SSH仅调用同步好的CLI，无科学源码/任务包SCP。

每项max1800s协作训练预算、合计3600s，不含同步/验证/取回；非硬限或断线保证。任何身份/资源/非零或缺exit/超预算即停，不自动补run。正式后不追加epoch/背景/轮数/seed。

工程仅假张量/无训练测试和2份CPU/null CLI闭环共4updates16presentations、真实encrypt0。正式不因工程通过冒称Windows/CUDA验证。

## 如何解释

预定关注相对TRAIN常数的DEV/TEST BCE及ACC、null、与零背景已见99.6384%并列，完整报告两臂。固定子空间标签不保证平衡，必须比较TRAIN-only常数，不能把高多数类准确率当有效。若明显保留预测能力，仅排除“该一次信号只存在零背景”的一种特例，不证明任意背景。若消失，只否定此背景和预算，不证明全部非零背景不可学。不按新TEST改候选；CLI仍自动保存TEST，非sealed确认。新背景重训不等于已有模型跨背景迁移。

完整取回runs/received新目录，仅核bytes/SHA/源码config/输入和初始化身份/退出/更新/DEV选模，不load权重/再生标签/模型前向/evaluate。用完2fits或触停即CLOSED。结果文档本地归档，不自动带入未批准的额外发布。
