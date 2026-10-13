# R1 / R2 算子次序只读审查

范围：当前 `compute/numerics.hpp`、`compute/model.hpp`、`compute/pipeline.hpp` 与 `references/r1_numeric.py`、`references/r1_model.py`、`references/pipeline.py`。这次只读源代码和已物化工件，没有修改这些计算文件，没有制定正式测试，也没有执行完整R2模型。文件hash见同名JSON，结论仅对应这些已读版本。

合法固定形状路径中，以下连接的尺度、截断及数学次序一致：

- RMS：先检查输入256，完整平方和后除维数（C++对128/1024/2048使用log2右移，等价R1整数除法），再加epsilon整数16；先提取k并对异常同时替换a=1、k=0，再形成mantissa。缩放向量B=floor(X·2^(12-k))，检查96f24；先floor24(BY)，再floor12，最后乘gamma并floor12。Q/K只在最终gamma输出使用各自profile整数界。
- inverse sqrt：初值F/2，六轮，每轮先检查Y∈[0,2F]；依次floor24(Y²)、floor24(MU)、floor24(Y(3F−V))，再数学floor除2，最后再检查状态。负数均为floor，未发现把w/2移到不同位置。reciprocal同样是六轮、两处floor24，然后按k还原。
- exp/SiLU：负数exp输入累计失败并返回0；大于32f12按尾部规则返回0；三段8位LUT与两次floor24相同。SiLU的正负支路、reciprocal(1+exp)、最后floor24及256域相同。
- Softmax：先用有效位和因果掩码取合法最大值；无效query统一选择dummy key0和零score。合法query但没有合法key时，双方均累计失败并走单key安全分母。R1将整行work置0，C++只将唯一被选中的x0置0，其他位置m=false，故传给exp的有效差值和最后概率相同。概率保持f24，完整PV后才floor24，再按query有效位置0。
- Q/K每头归一化、GQA及RoPE：16个Q head按h//2对应8个K/V head；128维拆成前64与后64，先计算ac−bs及bc+as，再统一floor20。双方采用相同逻辑位置及物理前缀，Q/K的profile界相同。
- L2：先形成完整平方和；record/末有效位置/非零条件不成立时先选正dummy分母和零输入向量，再缩放、floor24和floor4；检查2f20并在累计失败时输出e0/零向量。
- 主干：线性点积完整求和后一次floor12；残差在加法后检查256；SwiGLU完整乘积后floor12，再检查当前256；down_proj后残差检查顺序一致。此次MLP冲突不是双方在这些顺序上的差异。

发现的具体差异及影响：

1. **参考pipeline在证据不足时没有抑制答案。** `references/pipeline.py` 最后只设置 `insufficient_evidence = selected[0] is None`，仍保留generator生成的tokens/validity。`compute/pipeline.hpp` 的 `generate` 则以 `permitted=any_evidence && !failure` 统一将答案有效位置0、Token置dummy，并给status1。建议修复独立参考的最终结果处理，仍执行完整生成日程后才抑制输出。此差异不影响已运行的R0记录，因为那次确实有两条有效证据；但当前R1不足以作为无证据路径的结果oracle。
2. **全无效prompt入口：** R2 `Pipeline::generate` 将 `has_prompt=false` 累计为失败；R1 `IntegerQwen.generate` 忽略 `last_valid` 的exists值。规范模板固定有公开有效段，因此正常装配不可达，但直接参考入口对畸形prompt的失败结果不一致。建议参考也保留失败语义。
3. **异常输入/生命周期入口有意强弱不同：** R1对Token越界、RoPE位置越界、预先发现整数预算不成立直接抛异常；R2对秘密Token越界选择dummy并累计失败，RoPE依赖可信的逻辑位置构造。R2还强制cache顺序/所属会话及执行失败后不可重用，R1的低层forward未实施全部会话限制。正常fixed generate调度位置均在范围内，但不要把这些参考异常入口当生产等价协议执行。
4. **若干检查在R1冗余显式执行：** R1再次检查RoPE输入，以及QK原始值、floor12值、attention scale原始乘积；R2依靠前一步已检查的Q/K输出域及绑定的解析界直接运算。对当前真实绑定的profile，数学上这些冗余检查不能额外失败；若将R1故障注入用于比较失败位，双方检查位置并非完全相同。原生通用project也依赖其调用方先给出合法域输入，而R1 linear再次检查输入。

原生 `scale_shift` 当前已排除恰好−2^56，与R1左移的严格abs<2^56前置条件相符；±(2^56−1)的符号扩展判定保留。需要区分：原生秘密异常会累计失败并选安全值，而R1开发期的整数预算条件不成立会抛出异常，后者不是模拟协议继续执行。

权重域条件方面，R1在算子中还检查实际权重幅度；原生算子依赖固定工件的启动校验。已读取 `private_rag/configuration.py`，其S侧启动会逐个权重文件核对字节数与SHA-256，故此审查在该绑定前提下比较真实算子，不把未经启动校验的任意MappedWeights直接当正常生产输入。

本审查没有将SwiGLU候选应用到任何计算路径。当前Q/K修订已批准，MLP局部修订仍待批准。

后续修复记录：按工程协调要求，已修复第1项，在 `references/pipeline.py` 增加R0/R1共用的最终释放函数。仍先完整调用 `generator.generate`，再根据证据存在与累计失败统一处理256个Token/有效位，保持numeric failure的状态码优先级，并清除被抑制答案的length_limit标志。公开小型函数核对覆盖证据不足、数值失败优先级、允许释放的长度上限及不修改原始结果，均通过，见 `models_reference_release_policy_run.json`。没有重跑已完成的完整R0。初始审查文件hash保留，修复后pipeline hash另见同名JSON的followup_resolution。第2–4项继续作为边界差异报告；未因此修改受保护的数值域。

第二项后续修复：`IntegerQwen.generate` 现保留 `last_valid` 的 exists 并将其否定累计进 numeric_failure，未添加提前返回，256输出/255decode日程保持原样。公开合成hidden核对实际选择与累积语句，并检查当前生成函数只有循环后最终return，见 `models_r1_invalid_prompt_run.json`；此次没有执行完整生成模型。
