# 独立参考入口

这里的 R0、R1 会读取完整权重与完整输入，只能处理公开、合成或各持有方另行授权的离线材料。生产管理 API 和 R2 核心不得导入这里的代码，也没有配置项可以启用明文回退。

- `r0_model.py` 使用固定 Transformers 4.51.3 的 Qwen3 decoder/RMS/RoPE 类，按层读取真实 checkpoint；必须通过真实 RTX2060/CUDA 预检；CUDA 不可用时拒绝执行，禁止 CPU 完整模型推理与回退。保留固定物理位置、秘密有效位的明文参考值、完整 GQA/QK norm、末 RMS、Token 绑定和固定 greedy 预算。
- `r1_numeric.py` 用受检查的 int64 运算实现已定 f12/f20/f24 数值语义。每个乘法、累加、左移先检查界；负数右移用数学 floor。`r1_model.py` 将这些算子连成完整 28 层模型及固定 256 输出路径。
- `r1_retrieval.py` 给出 f40 点积、有效优先/较小编号并列处理、Top-2 和 I7 保守区间的整数参考。
- `pipeline.py` 接通公开资料切块、两个模型、Top-2、规范 1024 槽装配和 1 次 prefill + 255 次 decode。R1 准备失败不会作为可发布资料继续运行。

输入 JSON 为 `{"question":"公开问题", "documents":["公开资料正文"]}`。运行时必须明确指定参考级别并确认明文入口：

```bash
.venv/bin/python -m references.pipeline --profile R0 --public-input-json /absolute/public-input.json --output /absolute/reference-result.json --acknowledge-plaintext-reference
```

这不是隐私计算示例。任何参考成功都不能代替 R2 成功。当前资源策略要求完整模型使用 RTX2060，资源和时间成本应据真实运行记录报告。

当前数值配置为 `v1-f12-coordinated-20261013`：普通激活范围为 ±2048，Q/K、SwiGLU、残差及 down 输出使用已登记的独立范围和守卫。R1 读取 `artifacts/constants/numeric_profile_v1_qk.json` 的精确整数界。原始配置的超界诊断保留为历史记录，不能视为当前配置的运行结论。

原生工件／会话开发核对独立于 R0/R1，不运行完整模型：先运行 `.venv/bin/python -m model_tools.build_native_artifact_probe`，再运行开发构建生成的 `prag_dev_session_probe FRESH_LEDGER_PREFIX artifacts records/models_native_artifact_expected.txt`。每次必须使用新的账本前缀；仅比较公开合成输入与原始公开 checkpoint 的固定位置数值。这个可执行文件只在 `PRAG_BUILD_MODEL_DEV=ON` 时构建，生产 worker 不链接它。
