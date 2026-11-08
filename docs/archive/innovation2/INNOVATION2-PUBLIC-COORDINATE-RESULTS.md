# 六轮公开可逆坐标MLP：A6000四项结果

## 结论

公开输入坐标phi(X,Y)=(Y XOR F(X),X)在本预算下没有带来六轮最终bit预测的泛化信号。
DEV原始输入49.5239%、公开坐标49.4751%；TEST50.1526%/50.2319%，随机标签50.3006%/50.0458%。
公开坐标相对direct TEST仅+0.07935pp（总正确数+52/65536），DEV−0.04883pp且BCE更差，
不能据此声称有效改善或可靠优势。两种模式DEV/TEST均未通过预定探索门槛。
这是特定key/init/数据/优化与预算的阴性结果，不证明全部可逆坐标无效、六轮不可学习或完整44轮安全。

## 授权、运行与配对

用户在[四项预算](INNOVATION2-PUBLIC-COORDINATE-PLAN.md)说明后“推进”，授权本批发布、远程GPU运行与取回。
代码/配置发布与远端运行commit：`f9d58dde218a0586e34a26d13b6ebb998bbccf52`。
本地commit/push后核GitHub main；Windows只fetch/checkout，未commit/push、未安装升级。
正式顺序direct-real、direct-null、coordinate-real、coordinate-null，全部一次执行/exit0，未重试或按分数增预算。
运行ID `public-coordinate-20261026-100050`，UTC 2026-10-26 10:02:00–10:06:54（含SSH/构造/启动，不同于训练elapsed总和）。

- SIMON64/128六轮、绝对密文bit0，固定未知key_seed190502，init17，data193701/null193702。
- TRAIN65536/DEV8192/TEST65536，139264不同明文；与事先列出的9面板零交集，不声称核遍所有历史面板。
- 四项同64→128 ReLU→128 ReLU→1、24961参数，初始state/trainable SHA全部相同，明文/shuffle相同，真实/null分别标签一致。
- 公开坐标可逆，不损失64bit信息，无真实key、真实状态或其他输出bit标签；U不是实际X1。
  参数与初始化相同不意味着有效函数类/优化几何相同，不能将本对照解释成同一函数类的加速。
- Adam.001/clip1/batch256/epoch32，32次DEV选择，各8192更新/2097152呈现；四项共32768更新/8388608呈现。
- 两真实臂标签构造278528加密调用、139264不同明文；null不做真实加密，公开坐标计算不是秘密查询。
- 每项1800秒协作上限不是硬截止。GPU物理1 UUID `GPU-f80e8056-5b1a-a0c7-9b34-fb86b20206d5`→逻辑cuda0；
  每项前重查free≥24000MiB/util≤30%，无GPU fallback。相同更新不等FLOPs；以下elapsed实测。

## DEV选中模型

最低DEV BCE、相同取最早；四项均选epoch1/256更新，但实际全运行至epoch32/8192更新。
严格加载选中checkpoint后才请求TEST标签。外层读取TRAIN/DEV写`dev-decision.json`后展示TEST；
CLI已自动保存TEST，不是访问强制封存。没有按TEST选配置或继续训练。

| 项 | TRAIN ACC | DEV ACC | DEV BCE | TEST correct/65536 | TEST ACC | TEST BCE | elapsed秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct-real | 50.6271% | 49.5239% | .69331604 | 32868 | 50.1526% | .69323047 | 55.407 |
| direct-null | 50.9399% | 49.6948% | .69348503 | 32965 | 50.3006% | .69333921 | 55.109 |
| coordinate-real | 51.3336% | 49.4751% | .69338150 | 32920 | 50.2319% | .69317306 | 64.750 |
| coordinate-null | 50.7874% | 49.4019% | .69336406 | 32798 | 50.0458% | .69330756 | 64.984 |

真实TRAIN BCE direct=.69277406、coordinate=.69268093；coordinate−direct DEV BCE=+.00006545149。
共同真实TRAIN-only常量：TRAIN50.0320%/.69314698，DEV50.0732%/.69314645，TEST49.7116%/.69315108。
两真实TEST BCE均劣于常量；坐标TEST ACC相对常量+0.5203pp、相对null+0.1862pp，远未达3pp。

预定每种模式DEV ACC≥55%、比TRAIN-only常量与paired null各≥3pp、BCE优于常量、null45–55%；TEST同条件单列。
两种模式在DEV/TEST均仅null范围条件通过，其他信号条件失败；没有放松55%/3pp门槛。
55%本就是探索筛选，不是此前两轮99%合同或高精度成功。

## 后期损失：有训练过程拟合，无持出改善

最后epoch32保存的是**在线TRAIN BCE**与当时DEV，不是last checkpoint的重新评估TRAIN准确率：

| 项 | 在线TRAIN BCE | DEV ACC | DEV BCE |
|---|---:|---:|---:|
| direct-real | .52413734 | 50.1343% | .87617681 |
| direct-null | .51677955 | 49.7437% | .88103089 |
| coordinate-real | .52712321 | 50.3174% | .86519443 |
| coordinate-null | .51410107 | 50.5859% | .87138528 |

两模式真实/null后期均降低训练过程损失，DEV BCE却恶化。不能说完全没拟合，也不能声称完整记忆、
推算未记录的最终TRAIN准确率或认为真实标签区别于随机标签的结构已被学会。

## 工程与结果验收

- 本地873tests/mypy79/Ruff/diff检查通过；两模式干净源码CPU null check/train/evaluate各4更新/16呈现通过。
- Windows现场两模式CUDA null check/train各4更新/16呈现通过；没有在远端重跑完整pytest或全新安装依赖。
- 远端独立fresh repo，41源码/4正式配置SHA匹配、exactcommit且clean；使用原有环境adapter转CLI参数，无科学源码/任务包SSH上传。
- 完整四run经SCP取回；direct各11文件、coordinate各12（含model_metadata），共46文件核远端SHA/大小，远端原件保留。
- 保存JSON核源码/配置/commit、面板与标签身份、四项初始化、进度/选择/计数/退出、checkpoint字节SHA；
  每项32进度记录/32DEV点、四项合计128进度记录，训练elapsed总240.250秒。没有本地加载正式checkpoint、再生科学标签或模型前向/evaluate。
- 运行前独立只读快照审查未见确认缺陷；审查不等独立训练复现。

最佳checkpoint SHA256：

```text
direct-real     bf2c5c6a91f13b1afa3da7b50a36c2c756199663714e5a126861e46b17481cf0
direct-null     4c0e764b387b44a61ae16a1fa35e9f754b1feb2c44632cecbd1d94815620c8ab
coordinate-real 3796a2bc6d72fcfdfe6fad3cf088c59894d7f8a25fe7aa6ec68f45812807c745
coordinate-null abba3b3d9605b3e2a1aa1ea1d82a58850e78cb314d90bcdbb52a2c7cc338a8a9
```

本地证据：`validation/public-coordinate/{frozen-plan,remote-inventory,remote-postrun,dev-decision,verification,test-report}.json`。
本地结果：`runs/received/public-coordinate-20261026-100050-{direct-real,direct-null,coordinate-real,coordinate-null}/`。
远端：`G:\lxy\shared-m-cipher-nd-runs\public-coordinate-20261026-100050\repo`，保持运行commit且clean，未在运行中pull。
本批关闭，无追加seed/key/epochs/坐标轮数/宽度；下一步机制或确认实验需新假设与单独授权。
