# 完整层与单 Token Decode 实施记录

本阶段按已授权范围推进 M1 完整层、M2 单 Token Decode、M3 两文档私有检索。M4 完整 Prefill 与 255 次 Decode 整链不在本阶段运行范围。生产门禁保持关闭。

v1.0.11 的历史运行记录继续绑定其原始来源身份。历史完整归档不随本次源码发布；旧运行结果不改写为当前源码的执行证据。

当前混合后端候选：CPU 执行既有 SEAL/SCI BFV、OT、GC、Beaver；RTX2060 执行本方份额的模 2^59 加法、公开整数乘法、公开布局和 KV 份额存储。GPU 接口不包含联网、秘密重构、解密或秘密下标访问。CPU/GPU 传输只能包含本方份额、S 自有权重与公开参数。完整 Transformer 入口要求实际 LocalShares 后端；独立旧原语诊断仍可保持 CPU 局部运算。

模环依据：uint64 加法与乘法先模 2^64 再取低 59 位，与直接模 2^59 等价。GPU 不对份额使用浮点运算。KV 固定容量为 2×28×8×1280×128 个 uint64，每参与方 587202560 字节，析构/中止时清零并释放；不把该操作称为设备取证级擦除证明。

RoPE 候选：公开的 1280×128 表，每行含 64 个 cos 和 64 个 sin。GC 对每个秘密位置执行固定 11 位 one-hot 解码，一次解码复用于 128 列；位置无符号值大于 1279 时输出零且置共享失败位。公开表按小端 uint64 序列计算 SHA256 并与对端绑定。原有四次共享乘法、加减、floor20 和专用输出域不变，Q/K 头仅复用公开系数的秘密份额。

SwiGLU 保持输入检查、绝对值分支、三段 exp LUT、六轮倒数 NR、SiLU floor24、joint floor12 与范围顺序。最后 Token 的 argmax 改为固定平衡树，严格大于才选右侧，输入 ID 递增，因此相等时保持最小 ID。未改变允许的 ID 集。

完整层及L2/检索组件已有下述有限实际结果；单Token Decode已完成，固定预算两文档链路仍在运行。编译成功、GPU局部运算成功、全层成功分别登记，不能相互代替。已验证的等价实现直接更新原规范对应篇章。

实际 M1 Embedding：2 行、完整第 0 层、hidden=1024、16Q/8KV，最终 2048 个整数和失败位匹配独立 CUDA R1，numeric_failure=false。C 计时 103.759 秒、应用层双向累计 10735518208 字节（只取一个端点避免重复），GPU 本地 10 次 kernel。双端 ledger 均 closed/all_reserved=burned。记录：`hybrid112_layer_embedding_2_0_1ff4142dbc84438db1e7382d6715cf32.json`。该结果不包含257/160固定预算完整Embedding，不能据此声称M3完成。

实际 M1 Generation：2 行、完整第 0 层、hidden=2048、16Q/8KV，最终4096整数与失败位逐值匹配CUDA R1，numeric_failure=false。C计时213.51秒、应用层18377618914字节，双方ledger关闭并烧毁。记录`hybrid112_layer_generation_2_0_2296cff92f6345f28be2dfde120bec00.json`。两种M1对应源码及证据已另存`hybrid112_M1_source_snapshot.tar.gz`；完成层采用候选构建，之后正式原生版本记为1.0.12。

L2与检索连接已作独立真实双进程核对：三条1024维L2输入含零向量，随后固定全扫描两记录、Top2/I7与两槽证据提取。4103个最终整数（含eligibility、证据和独立诊断中的keep份额重构）及失败位匹配CUDA参考；numeric_failure=false，32.0751秒、1250243754字节。记录`hybrid112_postprocess_fcc69841b66f4fa08782cef499ac1c8a.json`及源码快照。此记录不包含完整28层Embedding，不能代替M3实际结果。

新增RoPE查表操作码14的边界核对：0、1、1278、1279正常；1280、环负一、高40位和符号位的无符号越界位置均返回零与秘密失败位。1024个整数和8位标记匹配CUDA独立生成的公开oracle，0.755701秒/5846137字节。`hybrid112_lookup_97ece7006ee243c0912d17ed95acb7a3.json`绑定实际源码和库；这是算子调试，不是正式验收。

实际M2：`hybrid112_decode_e245f7c77f914382806a5a8e7a07689e.json`。真实28层Decode、词表头、秘密argmax和GPU KV更新完成，57345个整数及失败位全部匹配CUDA R1，numeric_failure=false。C端6249.32秒、应用层双向累计500658003539字节（取单端计量，不把两端重复相加）。两方正常退出，账本closed/all_reserved=burned；源码归档身份、最终trace的28层、448 QK、448 PV、113坐标范围和56 SiLU计数已核对，见`publication_M2_verification.json`。初始1024槽零KV为已登记公开开发输入，不是已完成的R2 Prefill。结果只属于其原始构建，不代表后续源码改动均已重跑。

当前活动M3：`hybrid112_M3_0ef42a29ee7b4f499a4a70681b65c868`；实际257/257/160槽、每次28层、固定全表Token查询，D侧AES-GCM/OS密钥库、真实HPKE授权交付、完整U同步、PG不可变快照以及Top2/I7/证据输出。`run_m3.py`协调真实受理/claim/computing/releasing/final ACK持久化，原生产门不修改；查询不进入Generation。独立CUDA参考三次均无数值失败，峰值分配1349014528字节。当前仅已启动首份资料准备，尚无完整M3结果。源码和实际二进制已在运行前归档，禁止重建正在运行的`build/compute112m3`。临时PG、PKI、AES/HPKE密钥在该运行结束后由控制器清理；活动阶段保留它们是后续计算所必需，不是遗留后台服务。

`python -m tools.build_hybrid112_evidence`逐项核对完成记录的原始源码归档与材料账本，生成`artifacts/hybrid112_evidence.json`。活动registration单独列为pending，不采用日志中间值宣称完整通过。没有密码中间状态恢复或材料重用路径。


当前生产形状构建位于`build/compute112release`，全部开发编译选项为OFF，原生版本1.0.12。实际BFV上下文逐字段等于锁定工件；条件解码界登记的源码身份逐项一致。v1.0.11的272项归档验证独立保留，其中40项BFV、OT/GC底层及依赖源码仍相同，变化限于批量接口新增公开表秘密查表操作码。条件界文件保留原版本与来源，新的`hybrid112_materialization.json`说明其适用前提；没有把历史运行重新命名为1.0.12运行。生产库不含开发解码符号，worker实际返回BACKEND_NOT_READY，见`hybrid112_production_build.json`。

GPU KV中止清理补全：`Transformer::forward`覆盖输入Token查表到final RMS的整个异常区间，先清空调用方持有的KV，再中止密码会话。实际两个故障注入（权重加载器、阶段回调，在模型算术前触发）验证双方各560MiB缓存立即释放、不能复用、validity清零，账本aborted/all_reserved=burned；见`hybrid112_cache_cleanup_54c56e0a08434c7d8b0675c856f4b054.json`。此为故障清理依据，不是模型输出依据；已完成的M2和活动M3均绑定各自启动时归档，未冒用此补丁的核对结果。相应源码与新生产构建归档在`hybrid112_release_cleanup_source_snapshot.tar.gz`。

当前登记命令依次为`tools.build_hybrid112_manifest`、`tools.materialize_configuration`、`tools.materialize_spec_parameters`、`tools.sync_spec_artifacts`、`tools.record_spec_revision`，均使用项目venv和PYTHONPATH=.。完成运行后先重新登记其真实终态和源归档，再刷新工件；配置字节变化不会改变已经运行的M2/M3自有configuration。禁止覆盖活动二进制、常量或权重。`tools.check_configuration_bindings`独立核对实际配置文件与620项量化载荷，文档校验不代替该步骤。

可复用构建入口`tools.build_hybrid`只接受新目录并绑定构建前后350项源码/依赖；当前一次实际构建、BFV Context及C/S两方prepare/query四个关闭门禁均通过，见`hybrid112_fresh_build_entry.json`与其独立源码归档。构建本身没有运行模型。`build_crypto.sh --dependencies-only`仅准备锁定依赖，不重建旧原生目录；M3开发入口可通过`--build-directory`指定新构建，复制二进制后再次匹配已登记SHA。现有活动M3仍按启动归档执行，不能把后续入口修改记为该次运行已验证。公开进度工具`tools.hybrid112_status`逐行读取trace，不持有秘密份额或启动常驻轮询；`hybrid112_public_schedule.json`是公开算子次数推导，已核对两模型M1四份端点trace，不是完整通信量、耗时预测或正式验收规范。

后续开发入口统一采用`registered_build.py`：完整层、Decode和M3均要求显式新构建目录，核对构建源码/依赖与全部输出，禁止把改变后的源码登记成旧二进制的来源。运行前独立归档，临时副本再次核对；层与Decode控制器异常也保存未完成记录。`hybrid112_registered_development_build.json`及其源码归档实际绑定375项来源和输出，三个入口的可执行文件复制检查已通过；编译和登记核对没有执行模型。M2结果和活动M3均绑定各自原先归档，后续修改不冒充其执行结果。

终态证据检查严格区分closed、aborted和没有确认终态标记：成功记录必须有双方closed/all_reserved=burned账本；未成功记录保留实际终态，不补写标记。三个合成解析分支已核对并删除临时文件，见`hybrid112_evidence_terminal_debug.json`；该检查不执行密码协议。公开算子次数核对入口会在已登记M2/M3真正完成后核对最终trace，不把部分trace当通过。
