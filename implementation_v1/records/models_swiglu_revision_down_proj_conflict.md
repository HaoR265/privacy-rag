# 已批准SwiGLU后的down与后续主干域冲突

当前已批准配置 `v1-f12-qk-swiglu-local-20261013` 下，完整R1问题160槽、两份文档各257槽均执行28层，但首个失败都为layer2 down_proj输出超过主干256。最大f12整数分别24265166、25998451、26834075，每次该点有3个失败元素；输出均numeric_failure=true、e=0、零向量。对应运行用时77.845、127.930、127.113秒。记录分别为 `models_swiglu_revision_question_run.json`、`models_swiglu_revision_document1_run.json`、`models_swiglu_revision_document2_run.json`。

问题的R1首点5924.11279296875与独立R0相同点5929.431640625接近。完整R0 Embedding逐层观察发现少量残差特征维度持续超域、后层普通RMS/gate输出也超过256；生成1024槽prefill也独立显示普通输出730.77与残差约14859.67。因此不能仅把down输出提升某个值就声称数值域已解决。

依据NUM-003的运行域与NUM-004的严格中间整数预算，未获批准前继续保持现有域和失败策略。成套候选与全部真实权重L1/平方和/PV推导见 `models_coordinated_domain_candidate.md/json`；它尚未应用。完整R0观察不替代R1，更不替代R2；当前没有完整R1 RAG成功结果。
