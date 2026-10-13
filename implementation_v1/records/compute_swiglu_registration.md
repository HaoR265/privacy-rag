# 已批准 SwiGLU 局部域的原生绑定

用户于2026-10-13明确同意DEC-012，仅把完整门控乘积floor12后的joint/down输入设为65536；其他主干和down输出保持256。活动配置为v1-f12-qk-swiglu-local-20261013。

`tools/materialize_numeric_header.py` 从实际数值配置生成Q/K界、joint整数界和两模型各28层down矩阵的SHA-256与最大行L1；脚本逐层确认tensor身份、数量、预算乘积严格小于2^56。`compute/model.hpp` 仅在该joint使用扩大域，down路径仍先累加完整原始点积，统一floor12后检查256。

`compute/artifacts.hpp` 的S侧down加载器在第一次使用前读取实际映射字节并核对编译头SHA-256，逐行重新累计绝对值、核对最大行L1和权重单值256域，之后同会话缓存只读映射。文件属于不可变模型工件；运行中不允许替换或原地写模型。该检查不读取或打开秘密激活，也不把单方随机份额当作完整数值。

实际执行 `compute/dev_down_registration_probe.cpp`，重新核对全部56个文件的真实哈希/全部行L1；对两个模型首层故意传入错误哈希或错误L1均被拒绝。记录在 `compute_down_registration_debug.json`。另对模型、worker及数值探针执行C++17语法编译通过。这些是加载路径与源码核对，不是完整R2模型执行。

SwiGLU修订后的真实R1查询又在down投影输出发现5924.11超出256；这与上述无回绕界不矛盾。行L1证明保证计算原始点积安全，不保证其输出留在256；后续输出范围仍需独立处理。
