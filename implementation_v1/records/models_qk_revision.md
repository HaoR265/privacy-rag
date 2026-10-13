# 局部 Q/K 数值域修订实施记录

本轮用户已明确同意已审查的局部修订，并授权继续核对。数值配置版本为 `v1-f12-qk-local-20261013`，工件为 `artifacts/constants/numeric_profile_v1_qk.json`。这是对本次明确范围的授权；其他运行域变化仍需另行讨论。

修改只覆盖每头 Q/K RMSNorm 的 gamma 输出、Q/K RoPE 输出及缓存、QK 原始累加及 attention scale 乘积界。RMS 输入、Q/K/V 投影输入输出、V、残差和 MLP 均保留 abs≤256（f12 整数1048576）。未改变权重、修订、尺度、floor顺序、NR轮数、环位宽、Token预算。

| 模型 | Q norm f12整数界 | K norm f12整数界 | Q RoPE/cache f12整数界 | K RoPE/cache f12整数界 |
| --- | ---: | ---: | ---: | ---: |
| Embedding | 1094807 | 8943813 | 1548293 | 12648476 |
| Generation | 874687 | 6302376 | 1236996 | 8912916 |

profile包含两模型量化manifest与checkpoint manifest的SHA-256、112个实际gamma张量的文件SHA-256、来源文件/张量名及各自实际最大值。物化脚本重新读取实际gamma并核对文件hash与已审查最大值，未从候选记录盲目复制来源证据。常量JSON hash及实际 max(|cos|+|sin|) 也绑定在profile中。R1加载时核对模型manifest和常量hash，并从唯一profile读取本轮精确整数界。

NR tighter bound的推导仍保留在历史候选和新profile中：m∈[1,4)、y∈[0,2]时，两次floor导致的δ<5/2^24，下一步Y≤2^24+5。以 d=128 的 B 界379625064、H12界92682及真实gamma界传播，原始QK和attention乘积均严格小于2^56。`range_evidence_v1_qk.json`及当前`range_evidence.json`记录这些条件无回绕界；原范围工件保存在`range_evidence_v1_original.json`。

历史第一版失败证据与只读候选没有覆盖或改写。当前运行核对使用 `model_tools/check_qk_embedding_revision.py`，只处理已列公开合成文本，不定义正式验收规范，也不启用生产worker。每次完整28层运行会保存新文件，并保留首个失败算子及实际整数（如有）。

本轮同问题160槽完整R1已结束78.0945秒，首个新失败在layer2 MLP的SwiGLU乘积，输出e=0/零向量。R0同输入同点核对确认真实值同样约3175，明显超过仍保留的256。详见 `models_qk_revision_mlp_domain_conflict.md` 与对应JSON。未扩大MLP域，未声称Q/K局部修订使整个模型运行域成立。

两份公开资料的257槽完整28层也已跑完，均首次失败于layer2 SwiGLU joint，并返回e=0/零向量；没有将它们作为成功资料记录发布。

| 公开输入 | 物理槽 | 秒 | 首败最大绝对值f12整数 | 对应实数 | failure / e |
| --- | ---: | ---: | ---: | ---: | --- |
| question | 160 | 78.095 | 12993396 | 3172.215820312 | true / 0 |
| document1 | 257 | 130.826 | 13355742 | 3260.679199219 | true / 0 |
| document2 | 257 | 136.010 | 13651678 | 3332.929199219 | true / 0 |

进一步局部候选仅作证据保存于 `models_swiglu_domain_candidate.md` / `.json`，没有应用。该候选来自上游既定域及两模型全部56个down_proj的实际完整行L1界，不由观测值猜阈值。
