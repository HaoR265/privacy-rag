# 当前协议：利用有界 BFV 相位的整块受控解码

状态：整块协议已接入原生v1.0.8的private_matrix与private_linear_streamed；具体实际运行状态以同版记录为准。不是正式验收规范，也不是完整R2结论。

## 资料核对与路线选择

本地私人论文库已查询 BFV/circuit privacy/noise flooding、Cheetah、Gazelle，无匹配馆藏；没有声称读过不存在的附件。查阅了 [GAZELLE 原论文](https://www.usenix.org/system/files/conference/usenixsecurity18/sec18-juvekar.pdf)中 HE 与 GC 混合及表示转换的设计背景；本节的有界相位推导是本工程推导，不能归因于该论文。

锁定 SEAL 4.1.2 的 context.cpp 通过明文模数上的 CreateNTTTables 决定 using_batching，batchencoder.cpp 拒绝 using_batching=false。t=2^59 下 2 不可逆，x^16384+1 在 mod 2 上退化为 (x+1)^16384；不能直接假定存在 n 个独立的常规 SIMD 槽。本候选保留已成立的系数布局，既不引入 BatchEncoder/automorphism，也不改变 BFV 参数。全密文返回及噪声淹没与现行返回约束不符，此候选均不使用。

## 精确解码定理

令 t=2^59，q 为实际六素数数据层乘积，C 本地量 u=(c1*s) mod q，S 本地量 v=c0 mod q。令

A=floor(t*u/q), a=(t*u mod q)/q；B=floor(t*v/q), b=(t*v mod q)/q。

两方分别用精确 GMP 整数 divmod 计算自己的高部和余数，不交换这些量。欲得明文为 m=round(t*((u+v) mod q)/q) mod t，因此 m=(A+B+round(a+b)) mod t。

原生固定操作和锁定采样给出圆周相位误差 |e|<E，E=8194*(512*2^71+1)。实际严格满足 **4*t*E<q**。所以 a+b 与某个整数 k∈{0,1,2} 的距离小于1/4。

仅取各自两位区间索引 p=floor(4a)、r=floor(4b)，皆在0..3，则

k=floor((p+r+2)/4)。

证明：p+r≤4(a+b)<p+r+2。若 k=0，a+b<1/4，故 p+r=0；若 k=1，3/4<a+b<5/4，故 p+r∈{2,3,4}；若 k=2，a+b>7/4，故 p+r=6。上述公式在这三类均精确取 k。这里只使用已证明的 BFV 相位域，不把任意独立随机 u/v 当成有效密文。全环 m、负结果的环编码以及模 q 回绕均包含在证明内；没有浮点近似，也没有引入概率舍入或新的数值误差。

两位进位电路可用4个AND：c=p0&r0；h=p1 xor r1；s1=h xor c；s2=(p1&r1) xor (h&c)；k0=s1 xor s2；k1=s1&s2。

## 整块份额返回与隐私

每个公开 tile（至多512坐标）只新建一次 GC 对象。C 批量输入2T位区间索引；S 批量输入2T位区间索引及59T位新均匀掩码 rho。电路批量计算每个坐标的 k-rho mod 2^59，只向 C 输出该被掩码值。S/C 各自输出

C: A + (k-rho) mod t；S: B+rho mod t。

S 再加原有 W*X_S+R；两方份额之和正好等于完整线性结果。与 v1.0.7 相比，本地份额分配改变，总值、数值尺度、floor位置不变。C 不得到 k、B、rho、c0 或联合相位；S 不得到 p、A 或 C 份额。rho 独立全环均匀，使 C 的新增输出在给定其本地 A/u/c1 时仍均匀。结合既有 BFV、半诚实 GC/OT 隐私，可按相同参与方和允许泄漏进行组合；没有增加电路隐私/噪声淹没假设。全多项式 R、fresh Enc(0)、c1-only、秘密失败固定调度均保留。

每 tile 一个 GC、一次 evaluator input OT batch，GC内部122T个AND（4T进位+118T秘密掩码减法），除法轮数为0。局部精确整数除法不产生协议消息。输出字节数和总门数随公开输出量增长是不可避免的；这里去掉的是逐元素的完整BFV联合解码、独立GC对象和独立OT扩展批次。该候选不是把67625门的旧解码电路简单拼成大电路。

## 原型与证据边界

experiments/tile108 单独编译候选BFV对象，链接已形成生产库中的未变 Channel、SecureOps、SCI GC；不会替换当前生产库。真实双进程 TLS1.3 mTLS、isolated_development=false；公开输入/权重与独立 RTX2060 整数参考比较，重构仅存在独立诊断可执行程序。

公开整数核对包括16个进位真值组合、21436个小环精确相位/全部本地拆分、49220个实际q/t边界（包括各区间边缘与最大正负噪声界）。记录 tile108_algebra_candidate.json 是有限运行核对；通用正确性依据是上述定理和实际4tE<q，不以样例代替证明。

## 生产接口、依赖和公开调度

输入B×K为两方59位加法份额，W为S私有O×K；输出B×O为同环份额，小数位数f_X+f_W，不隐式floor。private_linear和短private_linear_streamed统一转B=1矩阵路径；长输入K>8192至155648使用最多19个固定8192块，每输出c1仍为独立密文，但按连续至多512目标积累本地相位后共用一次GC。秘密Token的155648项全扫描不变，不能跳块。

矩阵描述符80字节大端u64为[PRAGMAT1,2,B,K,O,f_X,f_W,16,floor(512/B),512]；长输入为[PRAGLIN2,2,1,K,O,f_X,f_W,8192,512,19]。先C发送、S核对回送、C核对，再输入密文。字段绑定已有session/config/stage/op/seq和TLS1.3；无新HPKE或AAD。v2与旧v1不兼容并在描述符阶段拒绝；配置哈希也更新。公开shape先检查；材料开始后异常自动abort并断开对方，不可复用。

原型与生产共用同一解码源内容，但二进制分别形成。真实运行中GC对象数、OT扩展批数按tile增加，门数和输出数据按公开坐标数增加；没有声称输出量可以不随维度增长。未引入旋转、automorphism、密文乘密文、BatchEncoder、额外降模或噪声淹没。规范原文按本算法更新，候选和失败路线仅保留工程证据。
