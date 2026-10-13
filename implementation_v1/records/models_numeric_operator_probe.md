# 公开合成非线性算子开发核对

`compute/dev_numeric_operator_probe.cpp` 是隔离开发入口，只在 `PRAG_BUILD_MODEL_DEV=ON` 时编译，并链接 `prag_crypto_development`。它启动两个独立进程，调用真实 `SecureOps` 和物化 `constants_v1.bin`。输入是源文件内固定的公开合成整数，输入份额非零，期望值由独立R1及真实常量生成。最终重构严格限于这些公开合成结果；生产worker不链接此入口。

范围为 exp_negative 的0、恰好32f12、32f12+1；SiLU的±1f12；两列Softmax的有效query与无效query。无效query仍输出dummy key0概率，query mask在后续PV才应用。恰好32f12与尾部的量化输出都为0，因此这两例核对的是实现输出及failure位，单凭输出不能区分内部支路。当前预期值及来源hash见 `models_numeric_operator_expected.json`。

生成期望值及构建：

```sh
.venv/bin/python -m model_tools.materialize_operator_probe
cmake -S compute -B build/models-probe -DPRAG_BUILD_MODEL_DEV=ON -DPRAG_BUILD_ISOLATED_DEVELOPMENT=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/models-probe --target prag_dev_numeric_operator_probe -j2
```

运行时提供一个尚不存在的ledger前缀。每次运行均建立全新两方ledger，旧ledger不得复用。例如以下路径只允许第一次使用：

```sh
build/models-probe/prag_dev_numeric_operator_probe build/models-probe/fresh-numeric-case-001 artifacts/constants/constants_v1.bin records/models_numeric_operator_expected.txt exp_zero
```

最后参数可选择 `exp_zero`、`exp_32`、`exp_tail`、`silu_positive`、`silu_negative`、`softmax_valid`、`softmax_invalid` 或 `all`。选择项是公开开发调度；每个具体算子内部调用完整安全实现。分例运行便于先测量资源负担，不将全模型交给未知耗时的探针。

首次构建对应 native v1.0.2。当前标量查表实现每次指数调用扫描3×256个公开表项，每项equal执行295个Boolean AND，每个AND含两个交叉OT，仅9次exp调用的LUT部分就产生4,078,080次单比特交叉OT；每次保留ledger并fsync。这个静态计数不含其余比较、截断、reciprocal等，不能据此声称完整模型可运行或给出可靠性能。实际分例计时另存JSON；未完成的例不计为通过。

本入口不建立正式测试或验收规范，不使用私有真实输入，不运行完整模型，不证明生产TLS或整体隐私协议的正确性。

已完成首例 `exp_zero`：真实双进程SecureOps输出与R1相符，进程均正常退出；具体通信量和耗时见 `models_numeric_operator_exp_zero_run.json`。其余6个case尚未运行，因标量LUT实测成本较高，按当前工程安排暂缓，以先完成已批准数值域的R1诊断。此首例不代表SiLU、Softmax或边界尾部已经核对。

后续使用等价查表优化的native v1.0.3重编后，全部7个case均真实执行并精确对齐R1：总104.631秒，exp0为1.908秒/52,499,762字节；两SiLU各约23秒，两Softmax各约26–27秒。数值及failure位一致，双进程均正常退出。见 `models_numeric_operator_all_v103_run.json`。这替代了“其余case尚未运行”的当前状态，旧标量首例记录仍保留作历史性能事实。仍未执行完整R2模型；剩余比较、NR、截断的标量OT成本仍显著。
