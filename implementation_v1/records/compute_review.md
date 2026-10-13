# C++ 模型与检索实现的独立静态复核

范围：`compute/numerics.hpp`、`retrieval.hpp`、`model.hpp`、`pipeline.hpp`，并读取 `secure_tensor.hpp` 与原语接口确认调用归属。本记录是实施期代码审查，不制定正式测试规模或验收条件。审查过程中根代理同时继续修改加载器和模型；下列行号指本轮读取快照，修改后的消除情况应再核对。

## 应修复的接口正确性问题

1. **KV 前缀可在公开错序调用下读取未初始化行。** 读取时 `model.hpp:26-31` 的 Cache 没有已写前缀；`model.hpp:99` 只检查 decode 起点位于 1024..1278。第一次调用直接 `L=1,write_start=1025` 或正常 prefill 后跳到 1026 会通过这个检查，随后 `model.hpp:118` 复制尚未填充的空 Row，`model.hpp:126` / `134` 使用 `[c]` 越界。应记录已完成前缀和状态，在进入任何协议/权重操作前拒绝非顺序调用；cache 路径还应要求 generation ModelShape。规则 CODE-P01、RUN-002、MOD-010。已发给根代理。

2. **O/down 的公开输出形状没有在索引前拒绝。** `project` 检查 cols 等于输入宽度、f12，并以矩阵 rows 返回；但 `model.hpp:138-144` 按固定 hidden 下标直接读 attention/down，未核对其 rows/向量长度。异常公开权重描述会导致越界而不是规定公开 BAD_SHAPE。建议层级一次验证所有 Q/K/V/O/gate/up/down 形状，并在输出使用前保留边界核对。规则 CODE-P01、MOD-003。已发给根代理。

3. **Token 表和 gamma 的公开尺度没有一致核对。** `model.hpp:57` 检查表形状但忽略 table.fractional_bits，`numerics.hpp:122` 只检查 gamma 形状。两条路径向 private_linear 固定声明权重 f12，因此错误尺度描述被静默接受；应显式要求 f12。实际加载器若已保证，仍应在此边界保留拒绝。规则 CODE-P01、NUM-002。已发给根代理。

4. **显式份额/版本类型尚未真正进入实际计算路径。** `secure_tensor.hpp:15-26` 的 Tensor 有 shape、f、configuration；但模型、检索和 pipeline 主要传 Row/vector<uint64_t> 与 bool，Record 也是裸 vector。当前 API 无法通过实际类型拒绝 f12 embedding 被当成 f20、不同配置记录混接、Public/ArithmeticShare/BooleanShare 误配。管理快照和加载器可以固定调用来源，但这不是 CODE-P01 的完整强类型边界。应继续接入带版本/尺度/保密表示的模块边界或明确登记未实现，不能把一个未使用 Tensor 定义算作完成。

## 当前公式与秘密控制的复核结果

未在本轮静态读取中发现以下主路径的直接算术差异：RMS 的平方和后 floor 除维度、epsilon=16f24、固定六轮逆根、私有 gamma；L2 先选末有效 hidden、秘密零分支正 dummy、末端 f20；SiLU 的秘密符号、指数尾项与倒数；masked Softmax 的无效 query dummy 和 f24 概率；QK 的先 floor12 后乘 f20 常量再 floor20；PV 在 f36 完整累加后 floor24；FFN 门控和两残差。

I7 的 B 使用 `1024*(2*2^21*4095+4095^2)`，低精度整数先 floor12、近似分数左移24，阈值为第二大有效下界，严格 upper<tau 才排除，不足两条全保留。高精度所有 N 条仍计算。Top2 比较采用有效优先、分数降序、槽位升序；第二轮以秘密选中索引排除。证据提取扫描全 N 的 t/m，未见普通明文命中寻址。

生成 prefill 1024 后固定 255 次 decode；反馈位置 Leff+step，公开缓存写1024+step，最后1278；EOS 仅影响秘密 alive/token/validity。最终数值失败优先于证据不足与长度上限，正常 Token 在失败时先秘密屏蔽。查阅范围没有直接按秘密 EOS、有效长度、候选数提前退出。

这些是静态代码关系核对，不是完整 R2/R1 一致性、BFV 正确性、权重质量、全部侧信道或完整系统实运行的证明。实际权重与配置验证、生产输入份额通道、真实计算调度和最终释放的端到端连接需依其各自工件与运行记录判断。

## 本轮已修复并核对的项目

上述 1–3 已在 `model.hpp` 和 `numerics.hpp` 修复：Cache 绑定创建它的 SecureOps，只接受 generation 模型、首次 1024 prefill 和随后 1024..1278 的连续单步 decode；进入计算就标记不可重用，完整成功后才推进前缀。缓存公开容量、已写行形状以及层矩阵全部形状/所属方/f12 在使用前验证。O/down、last_hidden 另有边界检查；Token 表和全部 gamma 均要求 f12。保留原先 32 行 tile 与 server_reader。

运行了包含 `session.hpp`、`artifacts.hpp` 的 `model_compile.cpp` 的 `g++ -std=c++17 -fsyntax-only -maes -msse4.1 -mavx2`，提供 native/include、compute、SCI/src 包含路径，返回 0；只有现有 SCI 对 OpenSSL 已弃用函数的警告。这是编译语法核对，没有执行模型密码计算。第 4 项仍需由实际模块类型/加载器/worker 边界继续接线；仅补公开形状检查不能把它算作已完整解决。

随后重读根代理新增 `session.hpp`：实际 SessionRunner 查询入口已接受 `BoundRecord`，其 tokens/validity 是 SharedVector<0,256>、embedding 是 SharedVector<20,1024>、eligible 是 SharedVector<0,1>；into() 核对所属方、59-bit残数和与 Channel 相同的配置哈希，才转换内核 Record。这已在最外层记录入口补上第 4 项的关键尺度/版本/所属方检查。内核仍使用固定签名 Row/Bit，未来若公开更多模块调用入口，应继续保持这层强类型边界；新 native worker 如何验证并构造这些对象不在本次读取范围中。
