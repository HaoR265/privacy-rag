# 已授权 M2048 接线的独立源码复核

2026-10-13。本轮完整阅读 root 实现的 `compute/model.hpp`、`artifacts.hpp`、`secure_tensor.hpp`，核对 `numeric_profile.hpp` 的字段与全部 394 条矩阵登记，并追踪它们与本代理实现的 `numerics.hpp` 运算前 guard 的连接。本次没有修改被审查的 root 文件。旧 256/QK/joint65536 版本审查保留于 `compute_independent_review_v1_qk_swiglu.md/json`；同名当前 JSON 保存本版源码和工件哈希。

结论：在已注册工件、诚实参与方及现有 SessionRunner 入口前提下，没有发现这次数值域更新引入的确定性运算前超 `2^56`、尺度错误、遗漏输入 guard、失败行秘密状态公开或部分和预算错误。结论依据是源码与公共整数界；没有将缺少整模型 R2 实跑列为代码缺陷，也没有把源码复核称为完整系统成功。

## 权重登记与线性计算

`numeric_profile.hpp` 的两个 197 项线性矩阵表，与 profile JSON 和对应量化 manifest 的名称、SHA-256、形状、最大逐行 L1、输入 cap 一一相符。所有 56 个 down 的登记也与既有 down 表相符；每个模型还含 113 个归一化向量，其 manifest 数值范围没有超过冻结权重上界。普通矩阵 L1 来自 models 已完成的全部文件扫描；本轮独立重核其元数据对应和整数算术，没有重复扫描全部权重文件。此前独立扫描的 56 个 down 文件以相同哈希衔接。

`CheckpointWeights::get` 在 C 和 S 均从同一编译 profile 填入 `maximum_row_l1/input_abs_f12`。S 首次 mmap 每个矩阵时，`require_linear_bound` 检查整个实际文件的哈希、形状、全部权重幅度及逐行 L1；逐项先检查 `magnitude<=row_l1-sum` 再累加，因此登记计算本身不会先溢出。最大行 L1 必须恰好等于登记值。后续矩阵映射复用同一所有者对象。

`Transformer::project` 要求调用的 input cap 与该矩阵登记相等，并以除法核对 `cap*L1<2^56`。公开的 32 输出行分块只限制内存，每一行仍进行完整 f12×f12 点积；普通路径完整点积后 floor12，生成输出头保留 f24。任何点积部分和的绝对值均不超过该行绝对权重和乘输入 cap，与求和次序无关。

| 模型 | 普通矩阵最大原始绝对界 | down 最大原始绝对界 | residual cap C | 两个 C 相加的界 |
|---|---:|---:|---:|---:|
| Embedding | 6817597882368 | 36702522768883712 | 67108863 | 134217726 |
| Generation | 11300839096320 | 37112142389837824 | 94906265 | 189812530 |

全部严格小于 `2^56=72057594037927936`。逐条核对结果在 `compute_coordinated_bound_review.json`。

norm 向量与 exp/RoPE/attention 常量不属于新 394 项矩阵 L1 表；它们继续依赖管理侧 `check_configuration` 对完整文件清单的哈希检查。该检查在 worker 启动前执行，且编译 profile 还由 worker 独立核对。不能把矩阵表解释为替代所有工件的身份核对。审查也不把任意绕过 SessionRunner 的自定义 Weights/常量调用纳入已证明输入前提。

## 输入域及残差接线

逐个检查 `project` 的现有调用来源：普通 RMS 输出、Q/K/V/O、gate/up、PV concat 均先检查 `main_abs_f12=2^23`；生成 `greedy` 只接收最终 RMS 的最后有效行，one-hot 选择不扩大坐标幅度。token lookup 保持 f0 one-hot×f12 权重，其块内及跨块累加不进行错误的 floor12，最后检查 main cap。即便采用忽略 one-hot 的保守界，19×8192×权重 cap 也远小于预算。

SiLU 输入和结果分别检查 main cap；sigmoid 只采用安全状态给出的 `<=2F` 粗界即可证明原始乘积 `<=2^48`。SwiGLU 两个已检查的 f12 因子原始乘积 `<=2^46`，完成后 floor12 得到 joint cap `2^34`。down 以这个 cap 和对应真实 L1 计算完整点积，再 floor12 并检查对应模型 C，之后才与已检查 C 的残差相加。普通 attention 残差的另一项更小，统一采用 `2C` 的界仍成立。代码没有先对大 down 值相加再检查，也没有把 residual C 输入直接送入按 main cap 登记的普通投影。

RMS 对残差采用 C 与平方和上界两个独立 guard：坐标先选择安全值再平方；逐项剩余预算先检查再加；任意行级错误最终整行与和归零，RMS 选择 `A=F/k=0`。有效 RMS 的 `A=floor(S/d)+16<=2^42-1`，k 范围仍为 `[-10,8]`。Q/K 的 128 维 RMS 默认输入 cap 为 main，输出保持既有专用 cap。较大的残差坐标没有自动放宽归一化后的普通输出或 Q/K 上界。

L2 同样在平方前、加法前 guard，平方和上界 `2^52-1`；记录无效、最后位置无效、零向量或本行数值失败均选择零向量和 `A=F`，仅数值失败增加 numeric failure。已经存在的全局错误不会被误当成本行平方和错误；最终 eligibility 仍与全局失败的否定相与。

## Q/K、PV 与失败分支

Q/K 专用输出和 RoPE cap 保持不变，RoPE f20 系数在配对加减后 floor20；128 项 QK f24 完整累加，再 floor12、乘 92681、floor20。含负数 floor 的绝对值取整后，Embedding/Generation 的 attention scale 乘积界分别为 56719459554461229 和 31932200060414491，严格小于 `2^56`。

RMS 的缩放向量在乘 NR 状态前检查 `<=96F`，状态在使用前检查 `[0,2F]`；通用乘积界 `192F^2=54043195528445952` 仍成立。NR 修正乘积采用不依赖收敛的 `26F^2` 粗界。失败替代值遵守同样界和固定六轮，不存在用原始超大平方和进入 NR 的路径。

Softmax 继续在正常、无效 query、无 allowed key 的替代行上保持非负概率总和 `<4F`；倒数每轮和最终状态的安全替代也满足这条推导。V 已在 main cap 内，因而 PV 每个任意部分和绝对值 `<4F*2^23=2^49`；不能使用扩大的 main 域后旧的 `1280*F*M` 粗界来论证。最终 concat 在进入 O 投影前再次检查 main cap，失败不触发公开缩短路径。

本次新 guard 的 10 组真实两方开发核对另见 `compute_guard_implementation.md/json`：精确合法阈值、上界加一、正负越界、平方前大输入替代、整行归零与既有全局失败分离均匹配。该 guard 由本审查代理实现；models 代理另行只读对照了 R1/R2 的坐标检查、剩余预算、整行替代、RMS/L2 失败和零分支、k/floor/state 次序，确认当前源码语义一致。这是跨实现的源码核对，仍不等于完整模型的实际 R1/R2 对齐。运行使用 profile SHA `59db14e6c761224f5b7b0f0aa2dcb509b7731b1f3f51b3447a07a7f907f1bcff`；后续仅元数据更新的 profile 身份另列在当前 JSON，不改写已运行二进制的身份。

## 未改变的范围与证据边界

`retrieval.hpp`、`pipeline.hpp`、`session.hpp` 与上一轮审查的源码哈希完全相同。此前固定 N 检索、Top-2、固定 256 槽生成、EOS 后继续运行、缓存位置、仅 C 最终释放及 prepare 份额语义结论继续适用于这些未变源码；它们依赖的归一化接口和结果尺度保持一致。

本轮没有执行完整 RMS/L2、Transformer 或 RAG；完整模型 R1/R2 对齐仍需要另列实际证据。范围条件不保证所有可能 Token 序列始终留在允许域；越界依规范秘密置失败并完成固定路径。全系统 production gate 仍不由这份审查开启。没有给出正式验收规范、性能承诺、恶意安全或恒定时间认证。

## 最终元数据身份同步

最终 profile SHA 为 `ff53725bf7f455075d5276141051227c7292f348fe10a872b2bc5996c352f1ee`，当前生成头 SHA 为 `4645b0df127689565c7e1a61f4871bd5e07ec6cd4f384eb78707c334c73fa3fe`。将新头文件中唯一的 profile hash 字面量替换回本次运行使用的 `59db14e6c761224f5b7b0f0aa2dcb509b7731b1f3f51b3447a07a7f907f1bcff` 后，完整头文件 SHA 恰好等于运行记录中的旧头 SHA；numerics 源码和 constants 二进制哈希也未变化。因此这是逐字节验证的编译参数等价，未据此虚构新二进制的运行记录，也无需因元数据清理重复 guard 运算。
