# 当前模型数值域与实际运行状态

当前版本为 `v1-f12-coordinated-20261013`。用户已授权在现有安全边界内进行合理数值/参数调整；本次使用已审查的成套域，保持模型/revision、量化、f12/f20/f24、ring59、严格2^56中间预算、全部floor位置、NR六轮、RMS与L2指数范围、固定Token预算及R0/R1/R2隔离。

生效profile仍沿接口兼容路径 `artifacts/constants/numeric_profile_v1_qk.json`，最终SHA-256为 `ff53725bf7f455075d5276141051227c7292f348fe10a872b2bc5996c352f1ee`。其中普通激活域为2048，权重/gamma仍为256；原Q/K每头、RoPE/cache及attention专用整数界不变。SwiGLU joint及down输入为2^34个f12单位，down输出和残差采用Embedding 67108863、生成94906265的f12坐标界。

所有394个二维矩阵重新核对SHA与完整行L1，profile绑定每项输入cap及完整内积/任意部分和界，包含393个实际linear、56个down、生成tied LM head以及只供Embedding查表的token矩阵。R1按来源hash首次检查每个权重文件，线性完整求和后才floor；输出行分块仅限制内存，不分割内积或提前截断。

RMS与L2在平方前检查坐标、加法前检查剩余预算。RMS上限为d*(2^42−16)−1，L2为2^52−1。任一行失败后向量和平方和归0，RMS选择A=F、k=0；L2撤销eligible并选择零向量/A=F。原始无效记录、最后位置无效或零向量自身不新增numeric_failure。R1与原生R2已约定同一逐项次序和替代语义；公开合法稀疏行、坐标越界、整行能量越界及L2阈值开发核对在 `models_coordinated_guard_development_run.json`。

为独立R1加入CPU矩阵加速：只在每个整数乘积和任意求和次序的绝对部分和均严格小于2^52、输入整数可精确转成binary64时，使用float64矩阵乘法表示整数。标准double GEMM的每个乘积、加法/FMA都精确，因此不改变R1整数结果；否则保留int64。QK使用它的已定部分和界，PV使用非负概率总质量小于4F的界，普通/下投影使用实际输入最大值乘已绑定权重L1。已实际确认NumPy2.2.5链接OpenBLAS0.3.28，调用双方及返回dtype均为float64，没有混合或降精度累加。负数、近2^52界、int64回退、513输出行完整矩阵、8193输出行分块与PV小例都与原int64结果逐位一致，见 `models_r1_exact_gemm_development_run.json`。此加速只位于独立plaintext R1，不进入R2。普通矩阵采用完整输出行kernel，大词表头按4096输出行分块；没有切开内积或新增floor。用户随后禁止CPU大模型推理，因此这条模型路径现已由资源策略关闭，以上为已完成的代码与小算子核对事实。

当前三个完整R1 Embedding均已正常完成28层：问题160槽耗73.774秒、第一份文档257槽121.699秒（原int64），第二份文档257槽25.282秒（精确矩阵加速）。三者numeric_failure=false、eligible=1，f20输出范数分别0.9999997787331023、0.9999992761342691、0.9999995678189383。相同问题使用加速再次完整核对耗16.145秒，输出范数与原int64记录完全一致；原运行未保存整向量，因此不声称这一整模型对照逐位验证了全部元素。

历史CPU完整R1 RAG没有完成。第一轮完成2份资料、问题及28层1024槽prefill且未出现numeric_failure，之后为等价内存调度调整而正常SIGINT退出，未保存完整decode计数，不能声称固定255次decode完成。第二轮重新从资料Embedding开始，资料分别累计25.101秒与50.542秒完成，问题累计64.388秒完成；最后完整日志为prefill第18/28层、累计193.985秒、numeric_failure=false。用户此时明确禁止CPU大模型推理，根代理立即SIGINT中断，进程退出130且已确认不存在。第二轮没有完成prefill，也没有执行decode。两次中断记录分别在 `models_coordinated_full_r1_rag_attempt1_interrupted.json` 与 `models_coordinated_full_r1_rag_attempt2_interrupted.json`，这些CPU记录不能宣称完整R1 RAG通过；当前GPU完成事实单独见下。

当前本机执行限制：用户要求使用RTX2060并禁止CPU模型推理。根代理已恢复UVM，真实CUDA环境预检查通过。`references/resource_policy.py` 要求恰好一块RTX2060且实际分配成功，任何失败都明确报错，无CPU fallback或覆盖开关。R0保持CUDA；R1新增完整CUDA整数模型后端，历史CPU模型构造无条件拒绝。当前模型输入、矩阵、整数非线性、归一化、RoPE、注意力、KV缓存与生成循环都在GPU执行；主机只负责文件/hash、公开元数据/调度和最终输出的传输。独立检索对1024维结果的小型参考计算仍与生产R2隔离。

GPU实现说明及内存/数学依据见 `models_cuda_backend_static_evidence.json`。真实19项GPU小算子核对已通过（0.28736秒），包括FP64直接/高低位分解GEMM、负数floor、NR、exp尾部、SiLU、Softmax无效query、RMS/L2失败行替代、RoPE等，全部与独立小型整数期望逐位一致。完整问题160×28在GPU耗8.781秒、两资料257×28分别8.728/8.672秒，全部eligible=1、numeric_failure=false；最大已分配显存约1.69GB。GPU完整向量已保存，原CPU记录未存全向量，因此范数一致不能宣称整个模型逐元素CPU/GPU对照。完整CUDA R1 RAG已按原公开案例完成，耗时950.012466秒，包含全部2doc/query、1024×28 prefill、255 decode及256输出槽，其中30有效Token。numeric_failure=false、first_failure=null、status=0、insufficient_evidence=false、length_limit=false；起止工件/源码身份一致。峰值分配5,102,529,536字节、保留5,337,251,840字节。PID870992退出0，随后nvidia-smi无计算进程。最终结果在 `models_coordinated_full_r1_cuda_rag_run.json`，完成与资源复核在 `models_cuda_full_completion_review.json`。


参数profile/范围工件保持冻结；资源限制与运行结果只进入代码、records和不参与全局配置互相hash的模型运行manifest。模型工件集合digest仍为 `9f39e9d0ff17720e7ce9b46c4e6087109a9211f5f06f90e0ccfc8c0be6967425`。这里没有制定正式验收标准，独立R0完成、完整GPU R1完成以及真实小算子R2核对均分别报告，完整R2模型尚未执行验证。

首轮完整CUDA尝试在第7层因log2的57倍广播临时量显存不足，见 `models_full_r1_cuda_attempt1_oom.json`；没有把该次当作完成。六级整数移位替换后先作19项小算子核对，再完成上述固定全链。完整执行用runner SHA `88cb9e5f7dd0d9840270d5af75c867034bc11dd7cf9e757d48bcf818571f5374`；结束后仅修正runner未来准备阶段提前返回的scope措辞，当前SHA `9df6ad25419462f05943f4b0c6178e52b74fd5670c5217c0640c68e883c66cb0`，模型数值源码未变，没有为该记录修正重跑模型。
