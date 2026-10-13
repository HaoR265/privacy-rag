# 已授权的 SwiGLU 局部修订

用户已明确批准经过审查的局部修订。当前配置版本为 `v1-f12-qk-swiglu-local-20261013`，沿用 `artifacts/constants/numeric_profile_v1_qk.json` 路径以便现有接口绑定。上一个仅Q/K修订版本保留在 `numeric_profile_v1_qk_authorized_history.json`；原始冲突记录、Q/K修订后冲突记录和未应用时的候选记录均保留，不能把历史失败解释为新版本失败或把候选的旧状态解释为当前状态。

只将已经经过floor12的 `SiLU(gate) × up` 和紧随其后的 `down_proj` 输入域调整到实数绝对值65536，对应f12整数268435456。SiLU、up输出仍检查256；down_proj输出及残差仍检查256。所有尺度、乘法、完整求和后截断的次序、模型、权重和预算不变。R1通用linear保持默认256输入，只有down_proj调用显式传入局部输入界。

依据来自上游有效域而非观测最大值：两个绝对值至多2^20的f12整数乘积至多2^40，floor12之后绝对值至多2^28。down_proj的完整内积及任意部分累加满足 `abs(sum X_j W_j) <= 2^28 sum abs(W_j)`。物化时重新读取全部56个真实down_proj矩阵，核对文件SHA及完整行L1，与量化manifest及历史候选逐一匹配。Embedding最大行L1为2136368，完整或部分内积界573476918263808；生成模型最大行L1为2160211，对应579877224841216，均严格小于2^56。每层证据见 `artifacts/constants/swiglu_domain_evidence_v1.json` 与profile内 `models[].down_proj_bounds`。

这只解决该局部算子的域和中间算术界，不能保证后续模型激活都在其已定域内。新版本仍需分别运行完整R1 Embedding及生成诊断，R2完整执行需独立报告。运行记录使用新前缀 `models_swiglu_revision_`，不会覆盖历史失败证据。

编译及小型控制流核对已通过；完整问题160槽和两份文档各257槽已顺序执行28层，分别77.845、127.930、127.113秒，均在layer2 down_proj输出首次触发仍保持256的主干域检查。输出均为numeric_failure=true、e=0、零向量。问题首点R1为5924.11279，独立R0同点5929.43164。随后完整R0 Embedding与生成prefill观察表明后续残差及普通归一化输出也超256，因此形成 `models_coordinated_domain_candidate.md/json` 的成套待讨论建议；尚未应用、没有扩大其他域、没有继续完整R1 RAG。新冲突记录见 `models_swiglu_revision_down_proj_conflict.md/json`。
