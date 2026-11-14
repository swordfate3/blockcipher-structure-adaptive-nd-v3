# 八轮SPECK Fourier论文复现 → 绝对位预测（GPU1 / 20GiB / 8h）

状态：APPROVED。独立设计审查31e8fe90指出GPU序号映射缺口，已补UUID/PCI绑定、tie差异与窄运行时兼容等约束；用户随后回复“是”并要求“继续”，批准本文详细协议，进入writing-plans。GPU1/约20GiB/共享8小时及本轮明确范围local commit/push与remote Git同步已确认；未开始编码/科学/安装/发布。旧SIMONFourier CLOSED_NO_CANDIDATE保持关闭。

## 1. 目标与路线

先复现Yuan/Wu/Zhang/Wu《Fourier Analysis of Neural Distinguishers》ToSC2026(1)441–467，DOI10.46586/tosc.v2026.i1.441-467，Eq4/Table3的8轮SPECK32/64教师及12项表达式。再在同算法同轮数、固定新未知64bit key、完整均匀32bit明文上预测绝对MSB0密文bit0（左16bit字bit15）。不改回SIMON、不做差分预测替代、不重训论文教师、不连续小样本科学调参。

已比较：A只抄12项公式（不足以复现提取）；B原教师权重+重新提取+同算法迁移（用户选择）；C重训教师10^9样本（不选，无新增训练许可）。不把论文12项密文对mask直接套到明文，不预设迁移恰有12项或能成功。

## 2. 证据与纠错

官方快照main/core/{speck.py,net.py,FourierAlg.ipynb,net8_small.h5,backup/,data_Fourier/}已只读核对，来源github.com/YuanYufeiISCAS/FourierNeuralDistinguishers；版本/逐文件SHA须实施前冻结，权重不入本项目Git，远端从官方固定revision取得并比本地SHA，不通过SSH上传科学源码/任务包。

官方算法：前7层完整枚举128个prefix（只是延迟剪枝，非限制prefix子集）；k8..64层各M=10,000,000 pairs，层内共享点给全部桶；桶权重mean(g(P)g(P')chi(PxorP'))，严格>.005；无beam32。coef另1,000,000点，代码严格abs(coef)>.05（与论文伪码>.1不同，按代码复现并披露）。固定tau=.1，官方此notebook无外层递减。不设tau网格/重复seed。

代码的real classification虽然变量名多次翻转，最终np.mean(Ysign!=res_flipped)与正确二值predict=(score<0)、标签Y∈{0,1}的accuracy等价；不得因变量名宣称方向自相矛盾，禁止TEST择优翻号。字序[c0L,c0R,c1L,c1R]每字MSB-first，与按同序打包64bit整体MSB-first等价。目标teacher hardlabel=p>.5，g=(-1)^label，重建score<0输出1，score=0固定0且保存tiecount；官方notebook的零分归1，因此保存官方tie1口径作为预定描述性复现对照，非零分两者方向等价，不依据结果选极性。两种极性仅作固定描述反例，主极性不调。

数据一条classification样本含两个实际加密块；官方speck.py确有两次encrypt，10m pairs=20m blocks，不把pair数当block数。原研究子代理相关相反叙述及SE计算不采信。10m二项比例p≈.2的SE约.000126；不是.0013。不得由近似SE给出严格检测下限/宣称八轮必无大系数；原论文记录为复现参考而非保证。

## 3. R：论文规模复现（一次）

R0：窗口内环境/CUDA/权重SHA/固定架构核验和单次SPECK官方22轮KAT及CPU/GPU公开向量一致性；参考和批量各最多256公开点(含KAT)，合计真实block调用上限512，不做真实微训练或吞吐探测。合成工程单测不生成真实密码标签。

R1：独立10,000,000 classification pairs，8轮、delta=(0x0040,0x0000)，Y~Ber(.5)，每样本独立随机key、两块共key，负类第二明文独立均匀；全部均真实加密。保存ciphertextpairs+Y作为只读公开评价面板（不保存key/plaintext），供teacher、发表12项及重新提取模型共用，不能借此调参。先评价teacher与Eq4；teacher accuracy须[.510,.518]才能继续GL；这是提案宽容复现门，不是统计置信区间。Eq4 accuracy参考.5087755，容差±.002；若不达则R标REPRODUCTION_MISMATCH，不做B。

R2：前7层无查询枚举；57层每层10m pairs=1,140,000,000 teacher rows，独立层seed，层内共享、全桶阈值，不宽度截断。超过400保留prefix时不按score选top400，停止ENGINEERING_LIMIT（400仅工程cap，不声称有限样本Parseval必保）。每层有boundedchunks的count/signedsum/kept masks/minmax与查询账。

R3：1m新均匀64bit输入，teacher hardlabels；全部候选估coef，abs>.05；10m新均匀64bit输入核teacher一致率（参考.8027425）；冻结提取模型并复用R1公开评价面板评分，不新生key/标签。预定复现门：提取10..14项、与发表12项交集≥11；匹配项coef误差≤.01；教师一致率参考±.006；classification accuracy参考±.002。所有规则是本轮容差，不能称作者原验收标准。输出不符合时如实记录，不重提取/换seed/翻极性，B不启动。

R+B搜索每层最坏检查800子桶，82层×10m×800=656,000,000,000 sample-bucket products的粗松工程上界，另记coef/评分；这是算术预算不是新增oracle调用，不保证8h跑完。teacher rows最大=1.140b+1m+10m+10m=1,161,000,000；R1只求teacher一次，保存hardlabels/概率与cipher面板给同面板报告用。R总real_blocks20m+R0≤512；无optimizer。

## 4. B：同八轮SPECK固定key绝对位预测（条件一次）

仅R正常通过全部门且资源剩余足够才B；一新private_random(8)不由publicseed确定、不落盘/keyhash/调度状态，不让模型收到实际key/轮key/其他密文bit/teacher。本阶段不使用SPECK差分教师作为标签。数据owner只返回明文与目标bit sign。

n32前7层枚举128，k8..32共25层，各10m pairs，输入每点边际全32bit均匀、paired共享suffix；层内多桶复用，500m完整八轮blockcalls。全桶>.005固定tau.1无beam32/无调阈值，工程cap400不得截断。空候选就停止无候选，不强造12项、不用余量重跑。

1m独立coefpanel估均值；按abs(coef)>.05保留后取至多12项（abs signed_sum降序、mask升序），不强行补项；截到12为本阶段明确模型选择，不等于搜索beam。DC理论0：固定key置换完整域输出每位精确平衡；模型intercept固定0，面板均值只描述。freeze模型JSON/registeredSHA/owner-read/strictcodec/完整config/source/task身份后才DEV。

DEV10m/TEST10m，public domains独立有放回，所有重复真实计费。新抽样不是保证从未出现：32bit域与大量搜索可能重合，交集未精确测则UNKNOWN，不能称unique/unseen/sealed；TEST抽样独立于冻结模型，不为去重改变paired estimator或全域分布。DEV只资源gate：accuracy≥.501且优于固定constant0，不再改权重/mask/阈值；失败无TEST。TEST只一次，保存正确总数、对constant0全N个D∈[-1,1]差值直方图、delta、Hoeffding单侧alpha.01 lower=delta-sqrt(2ln100/N)。TESTaccuracy≥.501且lower>0并正常身份/终态才positive exploratory；不通过只说明本模型未过预定门，不证明不可预测。

B最多500m+1m+10m+10m=521,000,000 real；全周期real≤541,000,512，policy542,000,000（差额非额外许可）；teacher≤1,161,000,000，optimizer0；另记parity/bucket/评分行数，绝不把teacher推理说0NNforward。无null新密码实验；R真实随机负类为论文对照，Bconstant0为预定基线，不追加独立key或随机mask拟合。

## 5. 资源、平台与时间

用户指定lxy-a6000 / DESKTOP-BBLPACJ / G:\lxy / physicalGPU1；不能fallbackGPU0，不停其他模型。启动前用nvidia-smi将index1解析并冻结为GPU UUID与PCI bus身份，主Torch/TFprovider均以该UUID绑定CUDA_VISIBLE_DEVICES，现场核对两者实际设备身份相同且对应用户指定index1，内部logical0；不能仅CUDA_VISIBLE_DEVICES=1就推定物理身份。映射/继承环境不符或不可核验立即停止；合成测试覆盖枚举顺序不一致及provider缺环境。首次正式环境初始化前统一origin，wall上限28,800s，包括环境/模型载入/R0/R/B/关闭核验；科学截止28,200s，最后600s预留在28,800内部用于停止保存回收取回关闭；取回未完成如实标NOT_RECEIVED，不延长计算窗口或冒称交付完成。CPU累计只记录、不能错误用单核8h限制多核CPU；wall为本合同上限。GPU时间不由CPU代替。8h开始前local开发/只读SSH不消费窗口；现场首次模型/CUDA初始化起不得重置。

用户显存20G按20GiB=21,474,836,480B记；TF allocator8GiB+Torch caching allocator8GiB（顺序阶段可释放TF后Torch额度仍不擅扩），其余留context/临时overhead；采样所有本实验PID GPUusedmemory合计，单非OOM计算batch降档50,000→25,000→12,500仅资源策略，总M不变/不换seed，不依据科学结果变批。OOM后不重放已尝试teacher/cipher调用，停止本cycle。WDDM若nvidia-smi PIDmemory不可用则不能冒称进程20G硬监控：必须取得等价可靠进程显存观测或停止GPU_UNVERIFIED，不能以allocator值代替。0.5sGPU/RSS检查，超20GiB即不发新工作并回收；非OS硬隔离可有采样超调，必须披露。开始前GPU1free≥24GiB/util≤30%方可launch，资源不合格不启动8h科学；不擅自等候无限。

host RSS本实验进程树合计32GiB、输出4GiB含data/model/report/log/IPC/temp和内部64MiBterminalreserve。生成物全在G:\lxy\shared-m-cipher-nd-runs\<freshid>\repo\cipher-lab\三原生目录及专属监督目录；TEMP/TMP/CUDA/TFcache/字节缓存位置同边界。不遗文件C:\Users/系统Temp。仅source/config/tests/docs用Git；权重从官方pinnedrevision远端Git取得，不本项目Git存权重；结果SSH/SCP取回新本地data/models/result接收子目录，原件不删不本地evaluate。

## 6. 已确认的最小架构调整

实测仅读conda-meta：predictor Python3.10.16 / Torch2.7cu118 / NumPy2.2.5；py39gpu_new Python3.9.13 / TF2.5 /Keras2.4 / NumPy1.21.6 /h5py3.7。只是元数据，不声称import/CUDA兼容通过。当前cipher-lab Python>=3.12，PEP695/UTC让旧入口不可直接导入；LinuxRSS不支持Windows。

建议保持一个统一CLI/RunStore/Meter主owner，主study在predictor3.10执行Torch计算；新SPECK路径做窄3.10兼容公共导入切割：CLI/config不eagerimport旧frontier/trainingconfig，或改唯一必要类型声明为3.10语法；UTC改timezone.utc等价/解析Z显式规范化，不能全项目降版或改变旧方法取值范围。pyproject保留主包3.12声明，远端仅本轮已审查source运行入口的Python3.10明确兼容子集，非宣传全包支持3.10。优先等价改写实际公共导入闭包少量3.12专属语法，不先大拆所有类型导入；兼容文件使用py310/py39适当Ruff目标与mypy版本配置，禁止自动将timezone.utc改回UTC，必须现场对应解释器synthetic导入/IPC闭环。现Policy/Meter对旧方法CPU门控不变；本新study采用专属walldeadline与进程树CPU只读累计的窄资源适配，实施计划明确落点，不悄然把所有旧CPU限制去掉。

原教师在已有TF3.9单独本地provider进程，精确重建官方架构后load_weights（不是load_model任意反序列化），通过版本化有界binaryIPC接收公开64bit教师输入并回hardlabel/probability；不接B秘密，不持有主RunStore，不SSH/不Git/不调度。该provider是单机模型适配，不是旧远程科学worker/任务包系统；源码同Git同步，不能stdin传程序。仅串行单请求在途、bounded stdin/stdout帧，无常驻服务/队列/重连/恢复/文件任务包；协议magic/version/count/bytes/SHA/sequence/EOF严格，超时/进程退出/超长/格式错立即停，未完成reserve不退款；provider推理rows主ledger计一次，不双计。主进程A期Torch与TF各8GiB预算合计，B关闭并reapTF后再开cipherGPU，不另争占其他GPU。现场TF不兼容就ENVIRONMENT_BLOCKED，无自动Torch权重转换或安装fallback。

新职责：ciphers/speck.py纯密码，ciphers/speck_tensor.py仅等价GPU密码运算；speck_study_config.py纯完整recipe；speck_study/{data,parity,search,teacher,teacher_provider,method,report}专用小模块；platform/{windows_resources,...}仅RSS/GPU进程观测；CLI只分派/错误/claim；artifacts保存opaque/持久读不承载科学；保持现旧schemas与方法边界。

## 7. 工程/发布/停止

先TDD公共合成toy/fakeprivate/teacher/cipher/GPU/RSS/IPC所有工程真实科学0，不做真实小训练/一串小科学样本，不运行未保护旧全pytest。独立spec/quality审查、主控fresh tests/Ruff/mypy与干净源码CLI闭环，通过后才Git本地明确路径提交推送。当前原dirtymain基线fd3ed2d451ef5a4c56380cc1dedde2473741640b，必须检查import闭包发布，不卷无关旧科学/报告/备份，不reset/clean/stash；新增入口若依赖既有未提交代码，单列最小依赖清单再审，不用HEAD差异冒充本轮diff。

Windows只fetch/checkoutexactcommit、无远端commitpush；SSH只启动已Git同步CLI与读取状态，不能上传源码。单周期R→条件B、每阶段最多一次，首mismatch/exception/resource/timeout/UNKNOWN停，retry0。父监督不写childowner；采样/起点/缺exit/缺显存证据failclosed；所有进程回收，不留8h外teacher。停止后saved-only报告方法流程图与实际层/准确率图、全选择机会、source/config/权重SHA、计数/exit/限制。没访问TEST不填50%，没有做浏览器像素验证如实写，取回不再前向重算。

## 8. 本文尚待核验

WindowsWDDM能否可靠读本进程GPUmemory；现TF栈/CUDA与H5是否加载；TFprovider3.9依赖来源冻结及main3.10窄切割实际改动；远端Git网络/磁盘可用。无法满足就停止并说明，不把用户8h/GPU许可解释成安装/换runtime/多key/阈值搜索/全域FWHT授权。

## 9. Task5平台证据补充（用户已确认）

用户确认采用DXGI“进程实测自报＋父监督核验”：每个实验PID通过Windows `IDXGIAdapter3::QueryVideoMemoryInfo(LOCAL).CurrentUsage` 取得实际使用值，父按Job成员、PID+创建时间、采样tick/新鲜度、适配器LUID核验后合计；缺测/过期/错身份/超20GiB停，不把缺失当0，也不用allocator/不可靠PDH计数替代。此证据类别记为 `dxgi-self-per-process+parent-luid`，不等同NVML、不声称完全独立于被观测进程。CPU-only的0必须实际查询返回；初始化期只能DXGI描述级身份，禁止据此释放科学工作；正式首次CUDA初始化必须在共享8h内，取得UUID+PCI+LUID后严格绑定 physical GPU1，未收口停。初始化期观测覆盖所有枚举适配器可保守合计，不能以两块同型号描述猜GPU1，不能在GPU0创建CUDA上下文。采样线程独立于教师预测调用；卡住/失联同样停止。

保存沿既有RunStore的file-fsync＋replace＋注册SHA/owner严格读回合同；Windows父receipt可追加长度/SHA读回证据，显式标记 `namespace_powerloss=UNVERIFIED`，不声称掉电强持久，不用父receipt替代child身份/模型持久读。POSIX目录同步保留，若显式提供同步实现却失败仍不能完成。此补充不改变R/B科学选择、预算、停止条件、Git或现场权限。
