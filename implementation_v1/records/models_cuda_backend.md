# 独立 R1 CUDA 后端

该后端是完整整数模型的独立明文参考，仅用于已授权的公开/合成输入，不能由生产路径调用，也不提供 C/S 密码协议。用户禁止 CPU 大模型推理后，规范入口改为要求实际可用且唯一匹配的 RTX 2060；CUDA 初始化、设备选择或分配失败即报错，没有 CPU 模型回退。旧 CPU 模型源码只保留为开发依据，构造入口无条件拒绝。

| 模块 | 输入与形状 | 输出与尺度 | 数据持有与计算位置 |
|---|---|---|---|
| `CUDAIntegerQwen.embedding` | 160 或 257 个 Token ID 和同长有效位，真实锁定 Embedding 权重 | 1024 个 f20 整数、eligible | 单进程独立参考持有完整数据；28 层和 L2 均在 CUDA，最后向量才传回主机 |
| `CUDAIntegerQwen.forward` | 物理槽、有效位、逻辑位置，f12 权重 | 全物理槽最终隐藏状态，f12 | 全部矩阵、RMS、RoPE、注意力、MLP、残差均在 CUDA |
| 生成缓存 | 28 层，每层 K/V 各 `8×1280×128`，有效位长 1280 | K 使用既定专用 f12 域，V 使用主干 f12 域 | CUDA int64 持久缓存，物理位置固定写入 |
| `CUDAIntegerQwen.generate` | 1024 槽正式模板与有效位，锁定生成权重 | 256 个 Token/有效位、数值失败、长度标记 | 一次完整 prefill、255 次 decode；EOS 后仍跑固定步骤，失败抑制最终输出 |
| `CUDAIntegerNumerics` | 完整带符号整数与当前 profile 的界 | f12/f20/f24 原定尺度 | 全部激活算术在 CUDA；点积完成后才执行原定 floor |

CPU 仅负责文件读取、来源 hash、公开元数据和循环调度、Tokenizer、最终结果传输，以及独立 RAG 中已得到的 1024 维向量的小型检索参考。它不承担 Transformer 矩阵乘法、激活、归一化或缓存运算。完整参考仍没有隐私计算属性；实际两方协议的核对由独立 R2 实现负责。

整数矩阵乘法满足绝对乘积和及任意部分和严格小于 2^52 时，使用 CUDA float64 GEMM：输入、乘积、每个整数加法或 FMA 的结果均可精确表示，求和次序不改变整数结果。较大合法内积将输入写成 `x = 2^16 floor(x/2^16) + low`，其中 low 在 0..65535；高、低两项分别由实际已绑定权重行 L1 证明小于 2^52，再在 CUDA int64 中重组，并预先检查重组界小于 2^56。没有 CPU GEMM 回退。浮点结果回转整数的相等性检查只是诊断，不能替代上述绝对项和与来源绑定证明。

平方和 guard 使用保持相同最终行接口的 GPU 饱和树。对非负平方项，每节点计算 `min(left+right,L+1)`，因此根节点超 L 当且仅当真实总和超 L，也当且仅当原逐特征剩余预算检查至少失败一次。所有失败行的完整输入和平方和仍归零，RMS 选择 A=F，L2 撤销 eligible 并选择零向量/A=F。节点在加法前已有 `left,right≤L+1`，当前全部 L 满足 `2(L+1)<2^56`。有效行每一步精确求和，floor 位置未改。坐标域失败独立 OR 进 row_bad。密码代理已对该等价性和高低位 GEMM 条件作独立只读复核。

真实 profile 中全部二维权重最大绝对整数为 Embedding 5120、生成 6560。因此校验文件 hash 与矩阵登记后，可在 GPU 使用 int16 无损保存这些 f12 整数；gamma 仍保留 int32，激活和整数算术均为 int64。这不是改变量化。静态缓存重量为 Embedding 1,191,684,096 字节、生成 3,441,397,760 字节；生成 int64 KV 缓存 587,202,560 字节。两模型顺序使用，显式 close Embedding 后再分配生成模型。大词表头先按 4096 个输出行切块再转换，单 FP64 权重块 67,108,864 字节；Token lookup 只把当前选中的行展开，绝不生成整张 FP64 词表。普通矩阵最多一个层内临时转换，实际 allocator/context 开销另外记录。

第一次完整 CUDA 尝试在第 7 层因辅助 log2 的 57 倍广播临时量 OOM，前 6 层没有数值失败。该失败已单独保存，未当作完成。现使用 32/16/8/4/2/1 六级整数移位，不扩张张量维度；正数得到精确 floor(log2)，非正数得到 -1。19 项真实 GPU 小算子核对已通过，包括 2^k±1、负数和零边界、高低位 GEMM、失败行替代等。三个完整 GPU Embedding 也已正常完成。修复后完整 GPU RAG 已实际完成，记录 `models_coordinated_full_r1_cuda_rag_run.json`：950.012466秒，全部2doc/query与1024×28 prefill、255 decode、256输出槽（30有效），numeric_failure=false/status0；启动与结束身份不变。峰值分配5,102,529,536字节、保留5,337,251,840字节，进程退出0且GPU计算进程清空。该公开案例说明独立整数参考全链已执行，没有评定通用回答质量或验证完整R2。

新运行在开始时捕获 profile、常量、两份量化 manifest、模板/Tokenizer 与参考源码身份，结束时重新比较。模型张量在首次读取时逐文件实际 SHA 校验。原三份单独 GPU Embedding 记录没有启动身份快照，已在 `models_cuda_embedding_postrun_identity_review.json` 明示限制；没有把事后 hash 冒称启动绑定。pipeline 和 Embedding 工具在 finally 中清理缓存并断开回调闭包，支持失败后释放本进程资源。这里记录实现及开发核对边界，没有制定正式验收标准。

完成后仅修正runner对未来准备阶段提前返回的描述：只有actual fixed_generation_executed=true才写完整固定预算。已完成记录保留执行时runner SHA `88cb9e5f7dd0d9840270d5af75c867034bc11dd7cf9e757d48bcf818571f5374`，不冒称执行过修正后脚本；对应纯表达式分支及语法核对见 `models_cuda_full_completion_review.json`。
