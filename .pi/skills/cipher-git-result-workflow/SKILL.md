---
name: cipher-git-result-workflow
description: 本项目代码提交发布、Windows Git同步、训练机使用说明和SSH/SCP结果取回时使用。代码单向Git同步，Windows不commit/push，SSH不承担正式训练启动与控制。
---

# Git发布与结果取回

## 分工不可混淆

```text
本地代码/配置 → 本地commit/push → GitHub → Windows fetch/checkout指定commit
Windows本机train → runs/<run-id>/ → 本地SSH/SCP取回 → runs/received/<run-id>/
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
- 普通训练由Windows**本机终端**直接运行统一CLI。不要恢复SSH启动器、WMI、Job模板、旧审批卡，
  也不把每个模型变更当一套远程部署框架。
- 用户若明确要求SSH发起小型可用性检查，可在有界CPU smoke范围验证，但要标注例外；
  不能借此开始正式训练、占GPU或后台常驻。
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
