# BFV 受控返回：60 轮精确除法实现（native 1.0.4）

本次只改变秘密整数除法的公开电路调度。SEAL 4.1.2、实际首数据层六个素数的 q、t=2^59、361/420/361 位布局、S 生成/C 求值、只向 C 输出低 59 位、每个结果独立全多项式掩码和新鲜 Enc(0) 均保持不变。现有配置与 `artifacts/bfv_native_manifest.json` 暂不改绑，由根代理统一集成；完整 R2 保持未启用。

依据为现行 `参数与接口.yaml` 的 `cryptographic_backends.BFV.controlled_decode`（1730—1767 行）、`linear_return`（1720—1729 行）、`packing`（1702—1717 行），以及 `04_协议与数据流程.md` 的 BFV 线性流程和 `技术规则.yaml` 的 CRY-003/004/005。公式仍为

\[
v=(u+c_0)\bmod q,\quad N=v2^{59}+\lfloor q/2\rfloor,
\quad m=\lfloor N/q\rfloor\bmod 2^{59}.
\]

## 前条件和省略高位的证明

`decode_locked_bfv59` 仅接受锁定参数函数生成的首数据层 q，验证 q 为 360 位，且各自所有者的本地整数满足 `0 <= local < q`。生产输入来自本地 RNS 到非负整数转换。两方秘密相位和小于 `2q < 2^361`，361 位加法不溢出；一次秘密条件减 q 后得到 `0 <= v < q`。

令 `t=2^59`。因为 q、v 为整数，

\[
0\le N\le(q-1)t+\lfloor q/2\rfloor
<q(t+\tfrac12)<q2^{60}.
\]

因此 `floor(N/q) < 2^60`，420 位商表示的第 60—419 位（共 360 位）恒为公开零。并且

\[
R_0=\lfloor N/2^{60}\rfloor<q.
\]

代码直接复制秘密 numerator 的第 60—419 位标签到 361 位 remainder 的第 0—359 位，保留其第 360 位公开零；不读取标签的秘密值，也不重构相位。随后固定处理 numerator 的第 59—0 位。每轮以前轮 `0 <= R < q` 为不变量，移位加入一位后 `0 <= 2R+b < 2q < 2^361`，一次秘密比较/条件减 q 即得到新的标准余数。60 轮结束后得到与完整 420 轮恢复除法相同的商和余数。

必须保留第 59 位商的这次秘密比较。实际 q 满足 `floor(q/2) >= t`，故 `v=q-1` 时完整商为 t，而向 C 交付的低 59 位应为零。不能直接以 `N>>59` 作为未经归约的初始余数只循环 59 次；对该初始余数增加一次秘密试减，本质上仍保留了这里的第 60 轮。本实现循环恰为 60 次。

专用 helper 是 `Decoder` 的 private 成员，唯一生产调用点紧接本地电路构造的标准化 v 和固定公式 N，不暴露任意 numerator/divisor 的快速除法 API。所有迭代数、宽度和帧数都仅依赖公开锁定参数。

## 代码、门数与实际核对入口

修改 `native/src/bfv.cpp` 和 `native/CMakeLists.txt`，新增只在 `PRAG_ISOLATED_DEVELOPMENT` 编译下存在的开发诊断入口及 `prag_dev_bfv_decode60_probe`。开发头在缺少此宏时直接编译报错；正式 `prag_crypto` 不应包含该符号。生产仍通过 `BfvEngine::block` 调用专用 decoder，完整结果密文、c0、u、v 和其他系数均未增加释放。

AND 门数由实际调用计数得到：phase add 为 `2*361`，phase 减 q 为 `2*361`，phase mux 为 `361`，numerator 加常量为 `2*420`，每轮为 `3*361`，总计 `5*361+2*420+60*3*361=67,625`。`GcBatchChannel` 会校验实收/实发门字节数与公开计数相等。

公开 Python 参考只用整数，核对 8 个边界：零、首个舍入跳变两侧、最后舍入跳变两侧、v=q−1、相位和恰为 q、相位和 2q−2。它们覆盖双方输入所有权以及完整商的第 59 位，结果与 Python `divmod` 逐例一致。详见 `bfv_decode60_public_reference.json`；这不属于 GC 或 R2 运行。

实际原语诊断使用独立 AF_UNIX 两进程：8 个公开合成相位案例经过真实 OT/GC，再执行两个完整 BFV 受控返回（非零权重和零权重）。全部结果匹配；C 的应用帧计量共 30,333,184 字节，其中直接 GC 批次 20,246,761 字节，完整 BFV 批次 10,086,311 字节，末尾开发结果重构帧 112 字节。两边 ledger 均 `terminal=closed all_reserved=burned`，stderr 为空。诊断重构只存在于该独立可执行程序中，输入是公开合成值；它不进入生产库，也不代表 mTLS、模型或全系统核对。最终实际结果和二进制哈希见同名 JSON 记录。

正式库、开发库、上下文程序及新增最小 TLS probe 均在 `native/build-decode60` 中低优先级、单编译任务实际构建。`nm -C` 已确认正式库没有 `development_bfv59_decode`，开发库存在该符号。构建日志包含 SCI 调用 OpenSSL 3 弃用接口的警告；添加 TLS target 后一次未重新配置即构建的调用报告 unknown target，随后显式 CMake 配置并编译成功，未隐藏该日志。没有因此运行或修改 GPU。

另已编译 `prag_production_bfv_decode60_probe fresh-prefix pki-dir`：链接正式 `prag_crypto`，强制 mTLS、`isolated_development=false`，只执行两个 3 列 BFV 返回；不包含合成相位注入入口。所需证书为 ca.pem、U.pem/U.key、S-compute.pem/S-compute.key，DNS 为 localhost。根代理随后重新构建正式 `native/build` 并实际运行该最小 mTLS 入口：2 个返回均匹配，应用帧共 10,086,423 字节，两边 ledger closed/burned，临时 PKI 已删除。见 `crypto_decode60_mtls_run.json`。正式库 SHA 与独立目录构建的库相同：`fb68b2204bb131f0a20dfe088057669ed8eb03ff78ad9a58401e849936e8389a`。此处是额外真实 TLS 记录，未把前述无 TLS 的开发记录改称 mTLS；记录中的一次调试耗时不构成性能对比。

生成脚本现输出 v1.0.4：`tools/build_crypto_cost.py` 保留旧成本 JSON 并写入 `crypto_static_cost_v1_0_4.json`。`tools/build_crypto_evidence.py --build-dir native/build-decode60 --output-dir records/bfv-v1.0.4-pending` 形成待登记 bound/manifest，默认无参数仍供根代理正式重编译后写入 artifacts。该脚本自动核对并绑定根代理实际 mTLS 记录、对应源码、probe 二进制和两份 ledger 哈希；其 `--extra-evidence path` 可重复绑定其他后续依据。本任务自身没有改写当前 artifacts；旧 manifest、context、correctness_bound 哈希在交接前已单独核对不变，最终登记由根代理进行。

## 静态通信范围

每次返回的实际 GC 表为 `67,625*32=2,164,000` 字节，按现行 64 KiB 公开帧填充为 34 帧；包括 96 字节应用头后，GC 帧共 2,231,488 字节。计入 c1 帧 786,528、S 输入标签帧 5,872、C 输出置换帧 155，所计每输出子集为 3,024,043 字节；共享输入密文帧仍为 1,572,960 字节。

沿当前固定图的 1,162,678,272 次 BFV 返回与 308,304,077 次输入密文，此子集为 4,000,939,070,651,616 字节（约 4.001 PB，十进制）。它排除 GC 输入 OT、初始化、其余共享/布尔计算、TLS/TCP/心跳/管理层，是当前实现公开调度的通信子集，不是所有协议的理论下界，也不是实测时间或吞吐。本次没有同条件旧/新版本性能对比，不声称实际速度提高了某个倍数。

本次不会使完整 R2 可实用运行；剩余输出数量与逐输出 c1 仍极大。不得因局部电路核对成功而开启全系统。
