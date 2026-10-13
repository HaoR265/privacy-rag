# 第一版密码核心实施记录

本记录对应 CRY-001—CRY-015、SEC-006—SEC-007、RUN-002/RUN-004/RUN-007、CODE-P01—P08。实施依据为当前完整机读规范；没有引入历史设计或自行制定正式测试/验收标准。以下工程细化由实施代理依 DEC-010 自主作出，不表示用户逐项确认过。

## 已形成代码与接口

`native/include/prag/secure_ops.hpp` 提供单方局部59位环份额与XOR位份额的低层算子：本地加减/公开常数，双向独立 SplitIKNP 交叉乘积，现场生成一次性Beaver材料，固定宽度A2B/B2A，布尔AND/OR/NOT，有符号或无符号比较、相等、Select、精确floor右移及一次A2B后的固定公开形状秘密one-hot树查表。张量形状、尺度、模型配置由上层typed tensor描述验证；低层Word本身只表示本方余数，不能用于秘密条件判断。

每次MulShared只打开被一次性A/B遮蔽的差值，公开乘积仅S加入。B2A对局部布尔输入使用一位选择OT，保持 b_C+b_S-2 b_C b_S 的完整功能；不发送另一半原始份额。矩阵运算由上层显式索引组合这些乘法；没有调用SCI高层会改变位宽或隐含截断的接口。

`native/src/bfv.cpp` 包含锁定SEAL上下文实例化、固定8192输入块的系数打包、S私有权重完整RNS提升与NTT乘法、整个16384系数的独立全环掩码、独立Enc(0)重随机化、仅c1传给C、C本地计算c1*sk、各方局部精确CRT、SCI同提交HalfGateGen/HalfGateEva与独立IKNP输入OT，以及精确大整数长除法。只有目标系数的低59位商交付C。S保留本地线性结果加掩码。全零权重同样运行所有消息和GC；没有直接向C交付完整返回密文。

所有BFV块都在实际第一数据层，q从该层上下文取得。v1.0.6对至多19个块先同态相加，再每输出组统一掩码/重随机化/逐目标解码；结果保持原始尺度f_X+f_W，上层决定截断。`private_linear`的S参数为按[输出,输入]排列的私有矩阵；C同样传公开输出行数，权重参数为空。

## 构建与开发隔离

运行 `tools/build_crypto.sh` 构建生产库 `libprag_crypto.a` 和无秘密输入的上下文导出程序；加 `--development` 另构建 `libprag_crypto_development.a` 与 `prag_dev_protocol_probe`。两库独立编译，生产库不接受开发模式，不存在运行配置开关可开启合成重构入口。开发驱动的重构代码只编译到独立可执行文件。

SEAL固定4.1.2提交119dc32；SCI固定f24bf3，不引入其历史HE前端、不另装EMP。OpenSSL/GMP与编译器实际版本见 `crypto-dependencies.json`。SEAL构建将透明中间对象异常、Gaussian采样、外部压缩依赖与HEXL明确关闭，CBD采样与Blake2xb明确固定。当前构建脚本使用j1控制内存与后台资源占用。当前native版本明确限制x86_64/小端/SSE/AVX2/AESNI；其密码后端没有GPU执行路径，不将RTX2060存在视为后端GPU支持。

当前源接口版本为 `prag-native-v1.0.4`，根代理已重新生成正式ART-002 bound/manifest；独立构建、实际mTLS核对与登记证据见下文。生产BFV原语已在解析依据复核后开放，必须满足源码提交/源码未改、SEAL/CBD/PRNG编译配置与实际上下文/精确整数不等式守卫；普通生产库仍拒绝开发模式和未认证连接。完整系统继续由上层数值与集成及GPU能力门控制，局部依据不等同于完整模型或系统合格。`artifacts/bfv_native_manifest.json`绑定实际生产静态库、代码、参数及证明文档哈希，无“跳过检查”环境变量。

## 原生计算帧与传输

`prag::Channel`独占已连接fd。原生计算帧是96字节公共头加由公开SCI调用决定的定长payload；这是独立于Python管理到worker控制帧的一层。96字节头布局：magic[8]、session[16]、configuration SHA256[32]、方向内sequence u64、payload_length u64、stage u64、operation u64、format_version u64。头整数为大端；原生payload固定小端x86_64字布局，P-256点固定65字节SEC1未压缩编码，无native size_t变长前缀。没有按秘密值压缩前导零，也没有让秘密候选数改变帧数。v1.0.2已将GC半门表固定为64KiB块；其他payload未统一填充到64KiB；接收端按照相同公开调用计划严格核对实际长度、会话、配置、阶段和操作。

上层以 `channel().set_stage(public_stage)` 切换公开模块；ledger在每次随机材料登记时更新operation，双方按同一调用树消费。TLS通过 `enable_tls(role, ca, certificate, key, peer_dns)` 启用，只接受TLS1.3、Ed25519签名、X25519组，双向验证CA与对端名称。`peer_fingerprint_sha256()`供正式受理入口把认证叶证书绑定到快照身份；仅“CA验证通过”不能替代ACL/用户绑定。连接创建/监听与管理身份映射由会话worker层负责。`Heartbeat`通过两个额外、方向独立的mTLS Channel发送/接收公共8字节计数；每个SSL对象仅由单个I/O线程使用，固定30秒发送、180秒未收到心跳则公开中止，无会话总时限。构造器强制心跳与计算的session、configuration及peer证书指纹一致。独立watchdog调用计算Channel的shutdown解除阻塞；回调`SecureOps::abort_session()`烧毁状态。`close()`先停止、shutdown心跳并join线程；正常公共关闭事务完成后调用`SecureOps::close_session()`。对象顺序必须保证Heartbeat先析构、SecureOps次之、三个Channel最后析构。

Linux进程入口应忽略SIGPIPE，使TLS断连按 `E_DISCONNECTED` 处理；开发入口已执行这一策略。其余错误不附带秘密输入、解密相位或秘密失败位置。原生库无通用Reveal接口。

## 一次性状态与清理

ledger以O_EXCL新建，先写公开session/config绑定；每次使用材料前追加reservation并fsync。正常关闭或异常中止追加`terminal=closed/aborted all_reserved=burned`，关闭后拒绝新reservation；互斥量保护watchdog中止与计算线程登记。即使最终marker写入失败，先前fsync的reservation和旧路径O_EXCL拒绝仍禁止复用。预处理两方向和GC状态各自独立，每个BFV结果掩码/Enc(0)/GC调用也登记。登记行不保存秘密材料，所有reservation永久视为已消费或已烧毁；崩溃后的旧ledger不可再次打开，不存在断点恢复。长期资料份额不属于这些一次性文件。

会话对象析构释放BFV/OT/GC状态、socket/TLS与局部buffer，并对显式可访问的BFV私钥和若干敏感临时数组进行OPENSSL_cleanse。没有因此宣称操作系统分页、分配器、GMP临时副本或完整物理内存已可靠擦除。开发运行的临时CA/身份私钥已删除，仅留证书摘要和不可复用ledger；本阶段未启动驻留服务器。

## 实际编译与运行核对

SEAL单独构建、SCI低层组件与SEAL组合构建成功。导出程序读取tc128合格上下文，密钥层420位，第一数据层360位；GC相位加法361位，分子保守420位。上下文全部实际素数/parms_id/Q在 `artifacts/bfv_context.json`，它不是算法安全结论。

独立公开合成数据入口实际运行了两个进程，核对MulShared、A2B→B2A、负数floor、比较/相等/Select，并运行非零与全零两行BFV私有线性受控返回。早期v1.0.1成功执行使用原生TLS1.3 mTLS（底层本机AF_UNIX socket），应用层计算帧计量为169316739字节，退出0；见 `crypto-debug-run.json`。这不是互联网部署、完整RAG、模型质量或正式性能结论。后续直接链接生产`libprag_crypto.a`的独立公开合成探针以`isolated_development=false`再次执行成功（相同计算帧字节数）。此次包含c1公开系数范围核对、ledger绑定与profile守卫。另一个三条真实mTLS连接探针已跑通初始双向心跳、主动断连公开burn、关闭后reservation拒绝与线程join；没有实际等待180秒静默分支，也不把初始心跳当成该分支运行证据。

## 解析依据与剩余边界

`crypto-bfv-correctness.md`从锁定CBD/三元采样、一次模切换、固定NTT明文乘法和精确解码推导条件性确定界。`artifacts/bfv_correctness_bound.json`以真实源码和构建配置哈希、实际q以及精确整数不等式记录依据。对当前固定profile，有界噪声引起的每次解码失败概率为0，任意有限会话的该事件并集仍为0；这不把软件错误、硬件故障、协议安全优势或模型数值回绕包含在“0”内。

当前模型图、公开执行描述、worker 与管理 scheduler 已形成接线代码，并已有独立局部数值和管理入口运行记录；完整模型R2与完整隐私RAG尚未实际运行通过，受保护全过程、最终交付和部署整体验证仍未完成。已批准的成套数值域已实施，R0浮点/R1整数参考进度以对应模型记录为准；完整R2仍因真实隐私GPU后端和资源/集成能力未满足而保持关闭。没有软件或硬件恒定时间证明，也没有完成对组合协议的独立安全审计。上游当前SEAL主仓提示后续版本含重要安全修复，锁定4.1.2未擅自升级；这里不把兼容编译与半诚实开发运行写成面向恶意输入的安全部署认证。

## v1.0.4 受控解码的等价60轮电路

专用入口证明 `N=v*2^59+floor(q/2)<q*2^60`，因此以秘密 `N>>60<q` 作初始余数，固定处理第59至0位，共60轮；保持361/420/361位宽、q/t、掩码、Enc(0)、c1及输出角色不变。实际独立公开合成两进程8个相位边界和2个完整BFV返回均匹配，生产/开发库构建及符号隔离核对通过。根代理重编生产库后另以真实mTLS、isolated_development=false核对2个完整BFV返回，应用帧10,086,423字节，两边ledger关闭烧毁，临时PKI已删除。正式manifest已纳入该记录；完整R2仍未启用。详细证明、实际计量、各入口运行边界和登记命令见 `crypto_decode60_implementation.md/json`。每次返回AND门由457505变为67625；该v1.0.4阶段完整query所计BFV通信子集为4,000,939,070,651,616字节。这是静态公开形状计数，未测旧新版同条件速度，不代表完整R2已可运行。

续做入口：将每次冻结更新的ART-002及native manifest同步到上层worker配置，继续核对身份/阶段生命周期及管理到worker接线，并按真实模型工件/数值前提逐项处理受影响路径。无需重新采样或重用已有ledger，不恢复旧密码状态。

## 开发入口与调用约定

`prag_dev_protocol_probe prefix bfv pki_dir`链接开发库，仅供公开合成输入；`prag_production_primitive_probe prefix bfv pki_dir`以相同独立诊断入口直接链接生产库，必须使用mTLS；两者均只在显式`--development`构建时存在，重构代码不进入生产静态库。`prag_dev_heartbeat_probe prefix pki_dir`使用三条真实mTLS连接，主动断开一条心跳检查burn与线程退出。pki目录文件为ca.pem、U.pem/U.key、S-compute.pem/S-compute.key（开发证书DNS localhost）。这些调试规模不构成正式测试或验收标准。

生命周期集成：创建并认证compute、hb_send、hb_recv三个Channel，创建SecureOps，再创建`Heartbeat(hb_send,hb_recv,compute,[&op]{op.abort_session();})`。生产主进程先忽略SIGPIPE；异常路径调用abort_session并关闭Heartbeat；正常公共会话结束先close_session再close心跳。Heartbeat析构会join线程；不得从Heartbeat自身回调线程调用其close/destructor，也不得在其存活期间析构SecureOps/Channel。

v1.0.1运行核对之后仅补充构造失败释放ledger fd及无堆分配terminal marker；已重新编译，未把此前probe的binary hash改写为新库。`crypto-debug-run.json`分别保存实际跑过的库哈希与该清理修复后的库哈希。

## v1.0.2 批量实现续做

实际完成同packed input的加密/传输复用和固定64KiB半门表分块，逐输出独立mask/Enc(0)与精确解码不变。实现、安全边界与资源安排见 crypto-batch-optimization.md，实际生产库+mTLS的3列/8193列双行调试结果见 crypto-batch-debug-run.json。原3列样例帧字节降至80025635；这仍不是完整系统运行，该v1.0.2阶段完整query的已计BFV子集下界约18.50 PB。bfv-long开发参数使用8193列，核对跨输入块和尾padding，绝不将其作为正式验收规模。


## v1.0.3 公开表秘密查找

`lookup` 现在只对59位秘密index执行一次A2B，以高位全零谓词和公开宽度二叉树产生秘密one-hot叶。每层AND按公开批次合并OT，叶B2A也按公开1024项分块；结果仍是59位环份额，域外index返回0，空表仍抛E_SHAPE。非2幂表只累加真实叶，padding不贡献结果。没有秘密重构、秘密数组地址或数据相关帧数。

锁定SCI的send_post与recv_post内部单帧批宽不同，`cross`统一把每次SCI调用限制为公开1024个OT，使严格帧长度匹配；两方向OT状态和每项新鲜掩码仍独立。详细功能、等价依据及逐帧计数见 crypto-lookup-optimization.md。

真实生产库+mTLS独立双进程已核对空表、公开长度1/3/256/1280共21个正常/负数/高位/边界index，以及18组MulShared跨1024个OT边界；见 crypto-lookup-debug-run.json。计量90437882字节，所有输出匹配公开参考，两边ledger均关闭烧毁。临时测试证书私钥已清理。仅256项查表的稳态应用帧量由静态2493513216降为4371374字节（都不含一次性base OT、TLS及本地fsync）；这不能解释为完整模型速度。该v1.0.3阶段完整query的BFV子集静态下界约18.50 PB，v1.0.3未减少BFV返回个数。


## v1.0.5/v1.0.6 已实现且原语已核对的当前阶段

v1.0.5完成公开步长s=2d−1的多目标系数打包，容量min(16384//s,8192//d)，每最终返回仍全多项式独立mask+fresh Enc(0)、只发c1、每目标独立GC。v1.0.6进一步加入S-only streamed权重回调，最多19输入块先同层合并；K>1固定d8192、尾块补零、目标8191。旧matrix API包在同一实现上，返回顺序及尺度不变。解析界同步为B_K=8194*(K*2^71+1)，B19严格满足实际q下的充分解码不等式。任何协议阶段异常自动abort并断开对端，禁止复用已预留材料。

源已冻结，独立和正式库SHA均为8f55a6d62509218014690fb58b79d77aea69436771ace18c9f1956cad93a3e72。真实生产库+mTLS三个公开合成案例已匹配（短维度3目标、K2非齐尾/抵消/零行、K19上限），共22输入、5最终密文、7目标GC、58,097,814应用字节，两ledger关闭烧毁，临时PKI移除。另开发错误回调案例双方aborted/burned，K20在网络前拒绝。详见crypto_merged106_implementation.md/json及crypto_merged106_mtls_run.json。生产库符号检查不存在开发相位注入入口。

本阶段待登记工件已写records/bfv-v1.0.6-pending，72个manifest文件和来源哈希已逐项核对；当前artifacts在交接时仍为v1.0.4，根端统一切换为v1.0.6。固定图成本已经独立复核：非空query的明确BFV帧子集3,401,298,000,293,088字节，含每目标freshGC初始化240字节，排除GC输入OT及其他协议通信。它仍约3.401PB，不是耗时测量，也不足以直接启用完整R2。已写入的整Token streamed接线未实际运行，完整模型R2/GPU私有后端仍未运行和启用。
