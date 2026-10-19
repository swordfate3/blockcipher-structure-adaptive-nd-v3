# Windows Git检出可用性验证（2026-10-19）

代码仓库：https://github.com/swordfate3/blockcipher-structure-adaptive-nd-v3

## 实际执行

Windows主机从GitHub直接clone并检出首个commit
`7797ee2a7410024a3898007094d2bf122fc03c0d`，不通过旧打包器复制源码。
在新目录中使用既有Python3.12.10、Torch2.14.0+cu130、NumPy2.5.3：

- SIMON64/128完整44轮已知答案测试通过。
- 实际CLI `check`通过。
- 实际CLI `train`完成8/4/4合成数据、1epoch、2个CPU更新、8样本呈现。
- checkpoint保存和SHA绑定重载、显式`evaluate`通过。
- Git工作区未被训练修改，manifest记录同一commit、dirty=false。
- 10个保存产物已通过SCP取回并核SHA；本地未加载tensor、前向或重算远端指标。
- 训练本体约3.609s，包含import/check/KAT/train/evaluate的验证约18.204s，SSH调用约20.192s。

这是工程检查，不是正式密码拟合或GPU验收；没有升级/下载依赖，没有使用GPU。
本次用户明确要求SSH远端验证，故由SSH发起小型检查；后续正式训练可在远端本机终端启动，不依赖本地SSH会话。

## Windows行尾处理

首次检出Git按Windows设置将13个Python文件变为CRLF，Git身份正确，但与Linux工作文件的原始SHA不同。
全部13项均能精确对应同一文件的LF→CRLF转换，没有把这种差异说成原始字节一致。
随后新增`.gitattributes`统一Python/JSON/TOML/Markdown的LF行尾，使未来检出得到一致源码字节。
原验证记录和hash保留，不覆盖。

## 环境边界

验证复用了机器上已存在的独立Python与site-packages；需显式加入该依赖目录以及保留Torch DLL目录句柄，
没有处理`.pth`、安装全新venv或执行`uv sync`。因此可以证明**当前代码在现有Windows环境中可运行**，
不能宣称全新机器依赖安装已验收。

如果有正常安装的Python3.12与PyTorch环境，可在仓库根目录直接使用README命令。
本机特有的既有Python/site-packages绝对路径不写入项目运行代码；验证适配器只存忽略的`validation/`。

本地原始证据：`validation/remote-git-check-20261019/`（不入Git）。
