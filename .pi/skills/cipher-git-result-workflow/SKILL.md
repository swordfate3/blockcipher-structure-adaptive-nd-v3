---
name: cipher-git-result-workflow
description: 本项目代码提交发布、Windows Git同步、训练启动和SSH/SCP结果取回时使用。代码单向Git同步；授权后可用SSH执行远端已有CLI，不传任务包、不恢复远程编排。
---

# Git发布与结果取回

## 分工不可混淆

```text
本地代码/配置 → 本地commit/push → GitHub → Windows fetch/checkout指定commit
Windows本机train（可由SSH调用已同步CLI）→ runs/<run-id>/ → 本地SSH/SCP取回 → runs/received/<run-id>/
```

仓库为`https://github.com/swordfate3/blockcipher-structure-adaptive-nd-v3`。
本地`origin`可用既有SSH密钥push，不读取/复制凭据；不要以设置Git remote代替发布成功。
**远端Windows不commit/push，不用Git上传训练结果，不配置CI/webhook/新服务。**

## 发布

- 用户允许提交/发布后，先检查diff和索引，再执行本项目验证。
- 明确列出文件git add，排除backups/runs/outputs/results/downloads/validation/环境/权重/凭据。
  `git ls-files`检查无结果和备份，`git check-ignore`检查产物路径；禁止force push覆盖远端历史。
- 记录commit SHA，只有push及远端refs成功核对才称发布成功。
- 未得到Git发布授权就只本地修改并报告；旧某次commit请求不是永久推送权限。

## Windows运行

- 用户明确允许远端同步/验证后，读当前`remote-windows-gpu-conda-ssh`基础设施skill。
  使用现有alias/hostname/存储约束，拒绝未知root和reparse，严禁把安装/缓存写到其他项目。
- 新验证目录使用`G:\lxy\shared-m-cipher-nd-runs\<fresh-id>\repo`；已有工作区有改动即停，
  不reset/clean/stash、不在运行中pull。fetch检出明确commit，并核对源码/配置身份和Git状态。
- 训练由Windows本机统一CLI执行；用户授权后，**允许通过SSH执行已Git同步的CLI启动指令**。
  不要求用户必须手动打开Windows终端。SSH是命令通道，不是任务包/科学源码传输通道。
- 不通过SSH上传任务包、审批卡、科学worker或每次重打源码；配置和实验实现先经Git同步。
  不恢复WMI、Job模板、远程编排框架或常驻服务。机器环境适配只转交原CLI参数，不承载科学任务。
- 先明确设备、次数、更新与协作预算；代码同步授权不自动授权拟合/GPU。
  启动后以实际progress/exit确认状态，不把SSH连接成功称作训练成功；普通SSH连接不保证断线存活。
- 使用既有可用环境，正常保留驱动/运行时必要环境；不盲目env清空，不自动pip/conda/升级。
  若用机器特有环境适配器，把它放忽略的验证目录，不固化到科学代码。

## 取回

1. 确认run与代码commit，读取results/exit，不把缺exit视为成功。
2. 取回完整指定run到新`runs/received/<run-id>/`，不覆盖旧取回，不删除远端原件。
3. 校验远端保存manifest及文件SHA、实际数据/更新/退出和已明确预算。只读JSON/字节，
   不加载checkpoint、不重算指标/标签/模型。身份声明与实际重算严格区分。
4. 报告传输/执行/科学结论分别的状态；合成/四pair小测试不是正式预测成绩。

SSH/网络失败允许有界诊断，不无限fetch；失败不能冒称代码不可用。未安装全新依赖、未实测CUDA
必须注明，不能用之前小CPU通过宣称所有GPU/长训练就绪。
