# 模型与输入实施记录

最新状态：当前版本 `v1-f12-coordinated-20261013` 已应用成套域，模型/尺度/floors/预算不变。遵照用户禁止CPU大模型的资源要求，旧CPU完整R1两次运行均已停止并如实记录；根端修复UVM后，已实现并启用只允许RTX2060的完整R1 CUDA后端，无CPU fallback。19项GPU小算子逐位核对通过；GPU问题160×28和两条资料257×28均正常完成，耗时8.781/8.728/8.672秒，numeric_failure=false。完整CUDA R1 RAG已按原公开案例完成：950.012秒，2doc/query、1024×28预填充、255次解码、256输出槽（30有效），numeric_failure=false/status0，启动结束身份不变。最终记录 `models_coordinated_full_r1_cuda_rag_run.json`；进程已退出且GPU无本任务计算残留。首轮CUDA辅助log2显存不足单独保留，已作等价常量形状调度修复。当前说明见 `models_coordinated_revision.md`、`models_cuda_backend_static_evidence.json`。profile/范围工件冻结，完整R2模型仍未核验通过。真实SecureOps的7项exp/SiLU/Softmax小探针与R1一致，记录 `models_numeric_operator_all_v103_run.json`。以下首版记录只保留历史依据，不表示当前有效域或CPU执行授权。

适用 MOD-001—MOD-011、NUM-001—NUM-010、CODE-P02/P06/P07/P08，DEC-010 范围内的自主工程选择。未更改模型、修订、量化、预算或隐私边界；这些选择不是用户逐项确认的新设计。

已完成：两个锁定 revision 的真实 checkpoint/tokenizer 文件下载及 SHA-256 清单；每模型 310 个规范张量流式 `floor(exact_checkpoint_value*4096)` 量化、独立 int32 NPY、每文件 hash、min/max 与最大行 L1 范数。原 BF16 直接按位扩为精确 float32，再以 float64 完成无误差的 2^12 缩放和 floor。两模型量化权重均通过 abs≤256 的本地登记。生成 checkpoint 额外存的 lm_head 与 token 表逐字节完全一致，仅存一份规范量化文件，明确 tied 别名。

模板从锁定官方 Jinja 实际渲染，保留 system/user/assistant 与空思考前缀。96 个公开 Token；问题在 offset79/cap128，证据一 offset211/cap256，证据二 offset471/cap256；末公开段 offset727/len9，剩余736—1023为无效填充。dummy 为普通 Token 0，有效位为 0。全部 added token 控制字符串（含 special=false 的 think/tool 控制串）在输入持有方拒绝。正文与公开段分别分词；Embedding 另用官方 postprocessor 并核对恰附加151643。

本地输入处理完成 UTF-8 strict、首 BOM 删除、CRLF/CR→LF、NFC；按字符边界回退并用两 tokenizer 复核正文256/Embedding257预算，目标重叠64，不跨 authorization_binding 混块。问题正文128和完整格式化Embedding160分别核对。这里的本地明文仅归 D/U，服务管理接口不调用它来收取问题或资料。

公开 exp 三张 LUT、1280×64 sin/cos RoPE 与 attention_scale 已生成并 hash；mpmath80与120位所得全部floor整数一致。这是高精度双计算一致证据，没有声称形式区间证明。条件无回绕界在 `artifacts/constants/range_evidence.json`，不等于模型激活保证或 BFV 噪声证明。

R0、R1完整模型、物理缓存和固定生成代码在独立 `references/`，生产不导入。已实际运行：输入/拒绝/零L2/带孔最后有效位置开发核对；R0完整160物理槽×28层Embedding输出1024维有限单位向量；R1同一真实模型完整160×28层执行完毕，119.50秒并返回 numeric_failure=true/e=0/零向量。R1不是通过结果。

首个失败已定位并与R0核对：layer0 K RMSNorm后、RoPE前，有效位置R1范围[-473.59448,455.00586]，R0[-472.98856,454.99985]；超过NUM-003运行域256。见 `models_numeric_domain_conflict.json`。只暂停受影响配置启用，不扩域或把安全替代值当正常模型答案。改变该锁定运行域需要用户讨论。

GPU检查：nvidia-smi可见RTX2060且约4.7GB空闲，但系统与项目venv的torch2.6.0+cu124均实际CUDA初始化/分配失败 `CUDA unknown error`；未修改驱动。已使用CPU流式生成工件与参考，不能把设备枚举当可用CUDA后端。参考模型按层读取，避免一次装入全部float32生成权重。

继续入口：先读取上述冲突证据与父级总进度；保持生产数值配置关闭；得到受保护设计变化决定后同步所有参考/范围/工件版本，再重新核对实际模型。当前没有制定测试范围、规模或正式验收规则。

补充原生工件与会话边界核对：`model_tools/build_native_artifact_probe.py` 从两份原始 safetensors 的每个规范张量取首、中、末三个标量，直接按精确 BF16 和 floor f12 生成 1860 个期望值。`compute/dev_session_probe.cpp` 仅在开发构建中存在，S 本地读取两个 `CheckpointWeights` 后全部吻合；C 使用不存在的本地路径仍可仅持有公开维度，未读权重。`Constants` 原生加载后的 164609 个 u64 值按规范顺序重新摘要，与独立 JSON 期望一致。

同一开发 probe 实际运行双进程 `SecureOps` 并执行一个真实 OT 乘法；128/160/257 输入份额仅对合成公开数据作恢复检查。答案释放的 256 Token/有效位及 status0..3 均只在 C 恢复，S 返回空向量；每次释放与固定 ACK 的帧总长都是 2512 字节。此开发进程未运行生产 TLS 或完整模型，没有把公开恢复 probe 当作生产路径。运行退出码 0、实测约0.96秒，详见 `models_native_session_run.json`。

完整公开合成 R0 RAG 已执行结束：两条资料各257槽Embedding、问题160槽Embedding、Top-2、1024槽规范prompt、一次prefill加255次decode，输出数组固定256槽，其中30个有效Token，length_limit=false，insufficient_evidence=false。CPU4线程float32按层读取真实权重，实测857.581秒（约14分18秒），进程已正常退出。复现用公开问题/资料及边界记录在 `models_r0_full_rag_run.json`。这证明独立浮点参考连接实际跑完；未评定答案语义质量，未运行R2整模型，也未解决R1数值域冲突。没有把它扩充为正式验收范围或性能承诺。
