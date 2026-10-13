# 实施进度与续做入口

任务：依独立机读规范实现第一版隐私 RAG。根目录既有资料不覆盖；工程位于 `implementation_v1`。本记录不是验收规范。

## 当前实施位置

混合后端已实际完成两模型各2行的完整第0层，16Q/8KV、真实权重与全部残差/FFN均在内；最终整数与失败位匹配RTX2060 R1。另已完成独立L2、两记录全扫描/Top2/I7/证据装配及RoPE查表边界核对。28层单Token Decode已完成，57345个最终整数与失败位匹配CUDA R1，numeric_failure=false；固定257/257/160槽的两文档私有Embedding/交付/检索仍在运行，未记作成功。CPU只执行授权的SEAL/SCI密码协议，GPU承担本方份额运算和KV；没有CPU明文模型回退。详情与唯一续做入口见`hybrid112_implementation.md`，完成证据和pending登记见`artifacts/hybrid112_evidence.json`。生产整链仍关闭，本阶段不运行完整1024 Prefill与255 Decode。

## 已完成的可复用阶段

规范：完整阅读 13 个包内文件并运行一致性检查，62 条技术规则未出现文档内部不一致。原始读取哈希保存在 `spec_read_manifest.json`。工件登记更新后的文档检查另行保存，不能代替协议运行。

管理与持久化：实际 PostgreSQL schema、身份关联、授权、版本、准备份额登记、逐用户不同随机重分享、HPKE、原子发布、完整授权同步、不可变会话快照和全局单会话门；D 状态 AES-256-GCM，密钥取操作系统密钥库。真实 mTLS 管理客户端已核对，资料准备源为合成调试值。没有声称完成私有资料 Embedding。

密码：锁定 SEAL/SCI 组合可编译并真实执行，未使用历史 HE 或额外 EMP。固定参数、第一数据层 Q、RNS/GC 位宽、确定性有界噪声证据已形成。生产密码库经 TLS 实际完成 OT、Beaver、位转换、floor、比较、选择和 BFV 非零/零权重受控返回。独立心跳与断线烧毁已核对；180 秒静默分支没有实等运行。

模型：两个真实 checkpoint 和 tokenizer 已下载并 hash；各 310 规范张量已 floor f12 量化并登记范围/行 L1。模板、全部 exp/RoPE 常量及原生二进制已形成。常量整数经过高精度双计算一致核对，未称为形式区间证明。原生加载器对全部常量和两模型 1860 个固定位置进行了源权重交叉核对。

计算图：真实 SecureOps 接入 NR、秘密桶形移位、LUT、RMS/L2、Softmax/SiLU、全扫描检索/Top2/I7/提取、秘密 Token 查询、两个 28 层模型、1024 Prefill 和 255 Decode、客户端最终释放。C++ 接口形状/尺度/缓存顺序缺口已修复；模型完整图尚未实际 R2 执行。新增 typed BoundRecord 真正接入会话边界，内部标量仍需持续补全表示契约。

## 当前数值配置与未完成部分

活动配置 `v1-f12-coordinated-20261013` 已形成实际参数工件，接入独立 R1 和真实 SecureOps 模型源码。普通域 ±2048，joint/down输入 ±4194304，down输出/残差 f12 坐标界分别为67108863和94906265；Q/K独立界、权重域、尺度、floor顺序及NR轮数不变。RMS平方和上限由原指数范围导出，先检查坐标、再检查每次加法预算；任一行失败时将整行和平方和清零，再选择正数dummy进入NR。L2采用对应整行预算与有效位处理。

394个实际矩阵的文件哈希和全部行L1已进入编译绑定，最大down任意部分和界37112142389837824严格小于2^56。条件无回绕依据在 `artifacts/constants/range_evidence_v1_coordinated.json`；其前提包括每项秘密输入检查和真实权重身份，不能据此断言所有Token输入均通过。

完整160×28 R1问题此前已实际通过：73.774秒，numeric_failure=false，eligible=1，1024项非零，f20向量范数约0.9999997787。两次 CPU 完整 R1 尝试均已中断，不能记作完整成功：第一次在两段资料、问题及 28 层 Prefill 完成后，为等价内存/调度优化而中断，未完成全部 255 次 Decode；第二次完成两段资料与问题 Embedding，最后完整记录为 Prefill 18/28 层、0 次 Decode，按用户禁止 CPU 模型推理的要求停止并确认进程退出。CPU 模型推理保持禁止。CUDA 已恢复，实际 R1 CUDA 后端已接通；小规模算子和完整 Embedding 的结果见下，完整CUDA R1 RAG现已实际完成，见下。此前调试失败与中断证据保留，不作为当前规范定义，也不改写为成功。

CUDA的19项公开算子与独立整数参考逐值一致。完整问题160×28在RTX2060执行8.781秒，两段资料各257×28执行8.728秒与8.672秒，均eligible=1且numeric_failure=false。随后完整固定预算R1 RAG实际完成：两段资料、问题、Top2、1024×28 Prefill、255次Decode、256输出槽/30有效Token，950.012466秒，numeric_failure=false、first_failure=null、status=0、insufficient_evidence=false。运行开始与结束的profile/常量/权重清单/tokenizer/源码身份完全一致；模型未回退CPU。峰值设备分配5,102,529,536字节、保留5,337,251,840字节。进程870992已退出，nvidia-smi无剩余计算进程。

首轮CUDA完整尝试的log2_floor分配57倍临时张量导致OOM，已改为固定六步等价整数位移，19项算子重新核对后完成上述第二次运行；数值域、模型与预算未变。完成后只修正未来runner未完成分支的scope措辞，不重写本次执行源哈希。公开回答、执行及清理证据分别保存在models_cuda_full_public_answer.json、models_coordinated_full_r1_cuda_rag_run.json、models_cuda_full_completion_review.json。本次完整R1不是完整R2，也不证明所有输入可正常回答。CPU旧问题记录没有完整向量，不能声称完整CPU/CUDA逐值对照。

管理scheduler、三socket lane、AUTHENTICATED/RELEASING落库ACK、worker退出与准备份额持久化已接代码，独立PG/TLS核对通过；尚无完整worker模型执行证据。本人取消、先取消后受理、原子claim、正常信号与异常子进程回收已经实现，实际PG/pipe/mTLS核对通过；claimed会话在资源停止前不释放全局槽。SIGKILL/断电后的恢复必须由操作员确认旧worker与三连接已停止，不以owner lock消失推断。生产门继续关闭。

已归档批量内核阶段的原生协议为 `prag-native-v1.0.11`；当前构建为 `prag-native-v1.0.12`，见首节。批量共享数值内核、完整 RMS 和完整单头 Attention 已接入正式密码库及 `compute/model.hpp` / `compute/numerics.hpp`，模型执行门仍关闭。

- 批量 GC 实现范围守卫、精确 floor、比较/选择、A2B/B2A、秘密指数与移位、平方和预算树及公开 LUT，输出新鲜份额，失败位保密；固定最多32768个逻辑坐标。
- 完整 RMS 保留原数值域、floor 顺序、六轮 NR 和私有 gamma。主干及逐头归一化按公开最多16行分块。
- 共享 QK/PV 使用用户授权的 BFV 矩阵 Beaver 预处理。两次受保护 BFV 计算的系数是本方新鲜随机矩阵，C 始终持钥；真实 Q/K/V 仍是两方秘密份额。向量及逐元素乘法继续 OT。
- 普通私有权重矩阵、联合 QKV/gate-up、私有 gamma 和固定155648项 Token 查表保持既有语义及参数。它们此前的 v1.0.9 运行有独立源码归档，不冒充本次构建运行。

生产构建、worker构建和诊断构建的密码库哈希一致；worker另存自身二进制哈希。公开合成输入的独立整数参考全部在 RTX2060；真实两方密码内核使用当前 SEAL/SCI CPU 后端，未运行 CPU 完整模型。输出重构仅在独立诊断程序内。

| 当前实际运行对象 | 耗时 | 应用层字节 | 核对结果 |
| --- | ---: | ---: | --- |
| RMS：1×128 | 0.401秒 | 45,445,604 | 匹配 CUDA R1 |
| RMS：16×1024 | 14.139秒 | 3,378,020,915 | 匹配 CUDA R1 |
| RMS：16×2048 | 29.402秒 | 6,729,356,883 | 匹配 CUDA R1 |
| Attention：1个查询×1280个键×128维 | 7.216秒 | 651,757,962 | 完整 QK→Softmax→PV 匹配 |
| Attention：16个查询×1280个键×128维 | 56.950秒 | 6,222,295,830 | 完整正常输入匹配 |
| 同形状含无效行、缺失键及Q/K边界 | 56.708秒 | 6,222,295,830 | 输出及秘密失败位匹配 |

共9组完整RMS、4组完整注意力头、29组基础/矩阵核对及13类双方公开拒绝路径。这里的“核对”是实现调试，未设正式测试/验收标准。边界样例预期产生秘密失败位，不作为正常模型成功。RMS使用S端真实gamma映射与哈希/L1检查；Attention使用公开合成Q/K/V与真实数值常量，不声称加载或完成模型层。

16×1280正常头的阶段计量为：共享QK约669.54MB，QK缩放507.77MB，行最大值/掩码492.42MB，exp/LUT及相邻乘法3.5105GB，倒数六NR20.72MB，概率乘法/floor358.05MB，概率检查373.22MB，共享PV181.45MB，PV输出处理108.26MB。原始记录含双端峰值RSS、帧数、方向切换和OT/Beaver/GC材料数量；父阶段包含子阶段，不能重复相加，方向切换也不是网络RTT。时间是本机诊断观察，不外推模型性能。

既有v1.0.9约10.69TB仅是矩阵/Token/gamma的静态子集。当前完整RMS已含gamma，不能再把旧gamma成本加一次；剩余SwiGLU/RoPE/L2、检索、argmax和会话等未形成当前完整成本，完整查询通信与耗时保持未知。见 `records/batch111_cost_scope.json`。

完整层、完整Decode和完整R2仍未运行，真实私有RTX2060后端尚未实现。原生与Python入口仍在处理输入前返回 `BACKEND_NOT_READY`。完整模型禁止CPU回退；内核完成、文档一致性和系统完成分别记录。实现依据与逐次运行见 `records/batch111_implementation.md`、`records/batch111_stage.json`；最终工件/规范核对见 `records/crypto_batch111_stage_completion.json`。

## 实际核对记录

| 范围 | 证据 | 结论边界 |
| --- | --- | --- |
| 规范 | `spec_read_manifest.json` | 文档内部一致 |
| 管理/密钥库/份额 | `management_debug.json` | 合成数据开发核对通过 |
| TLS/API/客户端 | `management_mtls_debug.json` | 真实 TLS/临时 PG；未运行私有模型 |
| 公开取消/子进程回收 | `cancellation_debug.json`、`cancellation_mtls_debug.json` | 真实PG、公开pipe子进程、mTLS，无伪造私有worker或绕门 |
| 配置绑定 | `configuration_bindings_debug.json` | 当前配置登记653文件/C侧33文件；实际复核状态以记录所绑定SHA为准；隔离tokenizer篡改检查 |
| 原生密码/心跳 | `crypto-debug-run.json` | 两进程真实原语；非完整 RAG |
| 全矩阵权重注册 | `compute_linear_registration_debug.json` | 394文件真实哈希/全部行L1与错误注册拒绝，非R2执行；以完成记录为准 |
| 平方和预算 | `compute_guard_implementation.json` | 真实两方10组边界与独立int128 oracle吻合；约2.058GB应用字节，仅guard kernel，非完整RMS/模型 |
| exp(0)原实现 | `models_numeric_operator_exp_zero_run.json` | 实际两方协议与R1相同；279.272秒、约7.52GB应用字节，非完整模型 |
| 优化后7项数值算子 | `models_numeric_operator_all_v103_run.json` | 实际两方SecureOps与R1吻合；总104.63秒，exp(0)约1.908秒/52,499,762字节；非完整模型 |
| Q/K及SwiGLU边界 | `compute_numeric_qk_swiglu_debug.json` | 真实秘密协议覆盖正负边界、严格2^56、NR和Top2指定公开样本 |
| 数值与 Top2 | `compute_numeric_debug.json` | 指定公开样例的秘密协议与 R1 值吻合；未跑完整 LUT/模型 |
| 会话输入/释放/加载器 | `models_native_session_run.json` | 真实 OT 与固定分享/释放；无完整模型 |
| 浮点 Embedding | `models_r0_embedding_run.json` | 完整 160×28，1024 维有限单位向量 |
| 整数 Embedding | `models_r1_embedding_run.json` | 完整执行并触发原数值域失败 |
| 此前CPU整数问题Embedding | `models_coordinated_question_run.json` | 完整160×28无数值失败，eligible=1；不能代替完整RAG或R2 |
| 生成Prefill观察 | `models_r0_generation_prefill_observation.json` | 真实1024×28、144有效位置、0 Decode，用于定位范围冲突 |
| 完整 R1 第一次尝试 | `models_coordinated_full_r1_rag_attempt1_interrupted.json` | 已中断；28 层 Prefill 完成，但全部 255 Decode 未完成，不能记作完整 R1 成功 |
| 完整 R1 第二次尝试 | `models_coordinated_full_r1_rag_attempt2_interrupted.json` | 用户要求停止 CPU 推理；最后完成 Prefill 18/28 层、0 Decode，进程退出已确认 |
| CUDA设备恢复 | `cuda_uvm_recovery_run.json`、`cuda_environment.json` | 普通UVM重载后cuInit=0/gpu_ready=true，实际小型FP64/int64运算通过；保留外屏与原KWin/OBS，无完整R2 |
| CUDA整数算子 | `models_cuda_operator_development_run.json` | 19项公开小核对与独立整数参考逐值一致；不是完整模型逐值验证 |
| CUDA问题Embedding | `models_coordinated_cuda_question_run.json` | 完整160×28，eligible=1、无数值失败，8.781秒；CPU旧向量未保留，不声称完整逐值对照 |
| CUDA资料Embedding | `models_coordinated_cuda_document1_run.json`、`models_coordinated_cuda_document2_run.json` | 两段均完成257×28，eligible=1、无数值失败；不代替完整生成或R2 |
| 完整GPU整数参考 | `models_coordinated_full_r1_cuda_rag_run.json` | RTX2060实际完成1 Prefill/255 Decode/256槽，950.012秒、无数值失败，非R2/非正式验收 |
| 60轮BFV受控解码 | `crypto_decode60_implementation.json`、`crypto_decode60_mtls_run.json` | 8个边界及独立真实mTLS双返回通过；全模型R2未跑 |
| 多目标与19块合并 | `crypto_merged106_implementation.json`、`crypto_merged106_mtls_run.json` | 实际生产库mTLS三组边界通过；独立回调故障双方中止烧毁，非完整Token/模型R2 |
| 合并成本与源码审查 | `crypto_merged106_cost_review.json`、`crypto_merged106_independent_review.json`、`compute_token_merged_review.json` | 独立公开计数及静态复核；不能替代完整模型执行 |
| 矩阵协议 v1.0.7 | `matrix107_shapes_*.json`、`matrix107_small_verified_*.json` | 3个小矩阵与4个真实尺寸全部匹配GPU R1；词表维度151936输出实际完成，非完整模型 |
| 真实Q/gamma权重映射 | `matrix107_checkpoint_runs.json` | 锁定生成模型第0层两项映射通过，S-only真实加载器；非完整RMS/层 |
| 完整RMS尝试 | `matrix107_rms_attempt_incomplete.json` | 900秒资源限时未完成，无完整输出比较；双方停止、临时PKI清理，不能判断数值结果 |
| 矩阵转录与失败清理 | `matrix107_transcript_cost_check.json`、`matrix107_failure_runs.json` | 七项转录计数吻合，3类公开故障路径实跑，非完整会话验收 |
| R2设备能力守卫 | `private_model_device_policy.json` | Python与原生worker独立阻止CPU整模型；正式worker编译/门拒绝已核对 |
| 模型资源入口 | `models_resource_policy_development_check.json` | 小规模入口拒绝检查，不加载 checkpoint；不是实际 CUDA 能力或模型运行验证 |

完整 R0 的此前运行已实际结束：两文档257、问题160、Top2、1024 Prefill和255 Decode，历史 CPU 四线程运行耗时857.58秒。见 models_r0_full_rag_run.json。它是历史独立浮点参考证据，不是继续使用 CPU 推理的许可，也不是R1成功、R2成功或性能基准。

## 工程选择与资源

Python 管理/客户端 + C++17 计算；公开二进制常量与 int32 NPY 只读 mmap；私有投影按最多16输入行分批，每次全部输出tile复用输入密文；公开R1仍采用独立CUDA计算布局。所有加法常数由 S 注入一次。不同层协议帧分别冻结，不把 Python manager-worker 帧和 native SCI 帧混称同一 ABI。

**当前资源政策：R0/R1明文模型仅允许RTX2060/CUDA，禁止CPU回退；R2按用户已授权的混合后端执行，CPU承担锁定SEAL/SCI的BFV、OT、GC和Beaver，RTX2060承担本方份额运算、公开布局与KV存储。** RTX2060已通过实际CUDA预检；此前cuInit=999及uvm EIO故障经普通卸载/重载nvidia_uvm恢复，cuInit现在为0，gpu_ready=true。没有GPU reset、PCI解绑、核心NVIDIA模块卸载或桌面/OBS重启，用户选择保留的NVIDIA DP外屏不变。恢复细节见cuda_uvm_recovery_run.json。

R1 CUDA整数后端已实现且完整固定生成已实际完成；它不代替完整R2。每次运行前执行 `.venv/bin/python -m tools.check_cuda_environment`，该入口只做设备及小型运算检查，不加载模型；GPU预检失败必须停止，不回退CPU。正常E→G路径释放Embedding权重后再加载生成模型，CUDA词表输出按4096行分块；资源足够与否仍须按实际运行峰值报告。

此前 R0/R1 使用 CPU 的执行结果仅作为历史证据保留。管理、下载/工件整理、编译和独立密码原语调试可继续；SEAL/SCI 已核实为 CPU 后端，这不构成密码 GPU 支持，也不构成启动完整 CPU 模型的许可。模型、密码方案、安全边界和固定预算均保持既定配置，没有通过换模型或缩预算规避资源限制。

## 下一位实施者入口

1. 完整CUDA R1此前已完成，本轮未重跑完整模型。当前完整RMS与单头Attention已实际核对；优先优化16×1280头内实测3.5105 GB的Softmax exp/LUT及相邻乘法，批量化剩余SiLU/RoPE/L2及残差检查，并实现真实R2 GPU后端。旧约10.69 TB是此前矩阵/Token/gamma静态子集，不能再与含gamma的完整RMS重复求和；当前完整查询成本仍未知。保持生产门关闭，继续有依据的等价优化与跨模块集成，不启动整模型CPU计算或缩减固定预算。只有出现新变化/失败才追加必要的R1开发核对，不重复已通过的完整运行。
2. 阅读 README、该记录、numeric_profile和range_evidence当前工件、compute_independent_review；核对 artifacts/configuration.sha256 与真实文件。
3. 直接修改机读规范原文维护当前定义。用户已授权合理范围内有依据的调整，无需逐项确认；不得擅改角色、隐私、模型修订、密码规格或固定预算。新参数同步工件、编译头、配置和实际核对。
4. scheduler代码已接通；继续检查实际worker事件与完整私有计算集成。使用冻结S/U各自快照，结束/中止烧毁一次性材料，不重放查询或释放。
5. 生产启用条件必须有实物和对应证据，再改 source capability；不允许用 mock、参考结果或用户可编辑 approved 字段开启。

临时 PostgreSQL、TLS probe、PKI 和一次性密钥按各模块记录正常清理。公开账本与摘要保留为不可复用证据；不清理与本任务无关的后台应用。


已归档v1.0.11最终配置、工件哈希与规范核对见 `crypto_batch111_stage_completion.json`。该文件仅在完成全部同步及核对后生成；不能将局部原语成功当作完整R2。
